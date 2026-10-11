"""
Video ControlNet Model with Flow Matching (VI → Fusion)

架构 (策略 B — warp 在 ControlNet 之后):
- VI视频 (4帧) → VAE编码 → vi_latent (起点)
- GT视频 (4帧) → VAE编码 → gt_latent (终点)
- IR红外 (1帧→复制4帧, 静态) → VAE编码 → ir_latent → ControlNet (hint) → 控制信号
- VI像素帧 → RAFT → 帧间光流 → 累积光流 → warp 控制信号 (多尺度)
- 轨迹: velocity = gt_latent - vi_latent
- 插值: x_t = (1-t) * vi_latent + t * gt_latent
- 模型预测 velocity → Loss = MSE(pred_velocity, velocity)

策略 B 核心:
- ControlNet 始终接收静态 IR (4帧相同)，输入分布与 Fusion4 一致，无需解冻
- warp 发生在 ControlNet 输出之后、注入 UNet 之前
- UNet 空间模块学习吸收运动对齐的 skip connection

训练策略:
- ControlNet: ❄️ 冻结 (输入分布不变)
- UNet 时序模块: ❄️ 冻结 (保留预训练时序知识)
- UNet 空间模块 + HFP: 🔥 可训练
- VAE / RAFT: ❄️ 始终冻结
"""

import os
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
from einops import rearrange

from diffusers import AutoencoderKLTemporalDecoder

# 导入STAR的模块
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(this_dir, '..'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

from video_to_video.modules.unet_v2v import VideoControlNet, Vid2VidSDUNet
from video_to_video.modules.unet_v2v import sinusoidal_embedding, USE_TEMPORAL_TRANSFORMER
from video_to_video.modules.unet_v2v import TemporalTransformer

try:
    from unet_SGFE import SGFEVid2VidSDUNet
except ImportError:
    from Fusion_SGFE.unet_SGFE import SGFEVid2VidSDUNet
from video_to_video.utils.config import cfg
from video_to_video.utils.logger import get_logger

try:
    from flow_match import FlowMatchDiscreteScheduler
except ImportError:
    from Fusion.flow_match import FlowMatchDiscreteScheduler

try:
    from ir_flow_propagation import IRFlowPropagation
except ImportError:
    from Fusion_SGFE.ir_flow_propagation import IRFlowPropagation

logger = get_logger()


class VideoControlNetModel:
    """

    """
    
    def __init__(self, opt, device=torch.device('cuda:0'),
                 use_hfp: bool = False,
                 hfp_block_indices=(1, 4, 7),
                 hfp_ratio=(0.25, 0.25),
                 hfp_patch=(4, 4),
                 hfp_use_dct: bool = True,
                 hfp_serial: bool = False,
                 hfp_init: str = 'zeros',
                 raft_model_path: str = None,
                 latent_scale: int = 4,
                 use_spa_local: bool = False,
                 use_temp_local: bool = False):
        self.opt = opt
        self.device = device
        self.use_hfp = use_hfp
        
        # ============ 双条件分支 ============
        # video_controlnet: IR 条件。controlnet_vi: VI 条件。
        # 两者结构相同，残差相加后注入主 UNet。
        self.video_controlnet = VideoControlNet().to(self.device)
        self.controlnet_vi = VideoControlNet().to(self.device)
        logger.info('Built IR ControlNet and VI ControlNet')
        
        # ============ Vid2VidSDUNet (或 SGFEVid2VidSDUNet) ============
        if use_hfp:
            self.unet = SGFEVid2VidSDUNet(
                hfp_block_indices=hfp_block_indices,
                hfp_ratio=hfp_ratio,
                hfp_patch=hfp_patch,
                hfp_use_dct=hfp_use_dct,
                hfp_serial=hfp_serial,
                hfp_init=hfp_init,
                use_spa_local=use_spa_local,
                use_temp_local=use_temp_local,
            ).to(self.device)
            mode_str = "serial(直接串联)" if hfp_serial else "residual(残差)"
            logger.info(
                f'Built SGFEVid2VidSDUNet  '
                f'blocks={sorted(hfp_block_indices)}, mode={mode_str}, '
                f'init={hfp_init}, ratio={hfp_ratio}, patch={hfp_patch}, dct={hfp_use_dct}, '
                f'spa_local={use_spa_local}, temp_local={use_temp_local}'
            )
        else:
            self.unet = Vid2VidSDUNet(
                use_spa_local=use_spa_local,
                use_temp_local=use_temp_local,
            ).to(self.device)
            logger.info(f'Built Vid2VidSDUNet (spa_local={use_spa_local}, temp_local={use_temp_local})')
        
        # 加载预训练权重 (ControlNet + UNet)
        if hasattr(opt, 'model_path') and opt.model_path:
            self._load_pretrained(opt.model_path)
        
        # ============ Flow Matching Scheduler ============
        # 新设计: t=0 是 vi_latent, t=1 是 gt_latent
        # 推理时从 vi_latent 开始，积分 velocity 到 gt_latent
        self.noise_scheduler = FlowMatchDiscreteScheduler(
            num_train_timesteps=1000,
            reverse=False,
            solver="euler",
            device=self.device
        )
        logger.info('Built Flow Matching Scheduler (VI → Fusion)')

        # 扩散状态是带噪的可见光 latent。网络直接预测融合 latent x0。
        from diffusers import DDPMScheduler, DDIMScheduler
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
            clip_sample=False,
        )
        self.infer_scheduler = DDIMScheduler.from_config(self.train_scheduler.config)
        logger.info('Built DDPM/DDIM schedulers (noise on VI latent, predict fusion x0)')
        
        # ============ Temporal VAE ============
        # vae_path 为 None 时不加载。当前训练会立刻换成 4x VAE，用不到这个权重。
        vae_path = getattr(opt, 'vae_path', "stabilityai/stable-video-diffusion-img2vid")
        if not vae_path:
            self.vae = None
            logger.info('Skipped temporal VAE')
        else:
            if os.path.exists(vae_path):
                logger.info(f'Loading VAE from local path: {vae_path}')
                vae = AutoencoderKLTemporalDecoder.from_pretrained(
                    vae_path,
                    torch_dtype=torch.float16
                )
            else:
                logger.info(f'Loading VAE from Hugging Face: {vae_path}')
                vae = AutoencoderKLTemporalDecoder.from_pretrained(
                    vae_path,
                    subfolder="vae",
                    variant="fp16",
                    torch_dtype=torch.float16
                )

            vae.eval()
            vae.requires_grad_(False)
            vae.to(self.device, dtype=torch.float16)
            self.vae = vae
            logger.info('Built Temporal VAE')

        # ============ IR 光流传播模块 (可选) ============
        self.ir_flow_prop = None
        if raft_model_path is not None and os.path.exists(raft_model_path):
            self.ir_flow_prop = IRFlowPropagation(
                raft_model_path=raft_model_path,
                latent_scale=latent_scale,
            ).to(self.device)
            logger.info(f'Built IRFlowPropagation (RAFT: {raft_model_path}, scale={latent_scale}x)')
        elif raft_model_path is not None:
            logger.warning(f'RAFT weights not found: {raft_model_path}, IR flow propagation disabled')
        
        torch.cuda.empty_cache()
    
    def _load_pretrained(self, model_path):
        """加载预训练权重"""
        if not os.path.exists(model_path):
            logger.warning(f"Pretrained model not found: {model_path}")
            return
            
        logger.info(f"Loading pretrained weights from {model_path}")
        
        try:
            if model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                load_dict = load_file(model_path)
            else:
                load_dict = torch.load(model_path, map_location='cpu')
                if 'state_dict' in load_dict:
                    load_dict = load_dict['state_dict']
            
            print(f"  Loaded dict keys: {list(load_dict.keys())[:5]}...")
            
            # 分离 ControlNet 和 UNet 权重
            ctrl_dict = {}
            unet_dict = {}
            for k, v in load_dict.items():
                if k.startswith('VideoControlNet.'):
                    ctrl_dict[k.replace('VideoControlNet.', '')] = v
                else:
                    unet_dict[k] = v
            
            print(f"  ControlNet keys: {len(ctrl_dict)}, UNet keys: {len(unet_dict)}")
            
            ret1 = self.video_controlnet.load_state_dict(ctrl_dict, strict=False)
            ret2 = self.unet.load_state_dict(unet_dict, strict=False)
            if hasattr(self, "controlnet_vi"):
                ret3 = self.controlnet_vi.load_state_dict(ctrl_dict, strict=False)
                logger.info(f'Loaded VI ControlNet from the same checkpoint: missing={len(ret3.missing_keys)}')
            
            logger.info(f'Loaded IR ControlNet: missing={len(ret1.missing_keys)}')
            logger.info(f'Loaded UNet: missing={len(ret2.missing_keys)}')
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            import traceback
            traceback.print_exc()
    
    def freeze_for_training(self, train_controlnet=True, train_unet=True,
                            train_hfp_only=False, freeze_unet_temporal=False):
        """
        冻结策略

        Args:
            train_controlnet    : 是否训练 ControlNet
            train_unet          : 是否训练 UNet（含 HFP）
            train_hfp_only      : 仅训练 HFP，冻结 UNet 其余参数和 ControlNet
            freeze_unet_temporal: 额外冻结 UNet 内的时序模块
                                  (TemporalTransformer + TemporalConvBlock_v2)
                                  保留预训练时序知识，只训练空间模块 + HFP
                                  用于解决微调时时序一致性退化问题
        """
        if train_hfp_only and self.use_hfp:
            # ── 仅 HFP 训练模式 ──────────────────────────────────────────
            for ctrl in self._controlnets():
                for param in ctrl.parameters():
                    param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = False
            for param in self.unet.hfp_modules.parameters():
                param.requires_grad = True
            hfp_params = sum(p.numel() for p in self.unet.hfp_modules.parameters())
            print(f"VideoControlNet: 冻结")
            print(f"Vid2VidSDUNet (base): 冻结")
            print(f"  └─ HFP modules: 可训练  ({hfp_params/1e6:.2f}M 参数)")
            print(f"     (梯度可回传过冻结层, 但冻结参数不更新)")
        else:
            # ── 常规冻结策略 ─────────────────────────────────────────────
            for ctrl in self._controlnets():
                for param in ctrl.parameters():
                    param.requires_grad = train_controlnet
            status_ctrl = "可训练" if train_controlnet else "冻结"
            print(f"ControlNet-IR / ControlNet-VI: {status_ctrl}")

            for param in self.unet.parameters():
                param.requires_grad = train_unet
            status_unet = "可训练" if train_unet else "冻结"
            unet_name = "HFPVid2VidSDUNet" if self.use_hfp else "Vid2VidSDUNet"
            print(f"{unet_name}: {status_unet}")

            # ── 额外冻结 UNet 时序模块（防止微调破坏预训练时序知识）────────
            if freeze_unet_temporal and train_unet:
                from video_to_video.modules.unet_v2v import (
                    TemporalTransformer, TemporalConvBlock_v2
                )
                frozen_cnt = 0
                frozen_names = []
                for name, module in self.unet.named_modules():
                    if isinstance(module, (TemporalTransformer, TemporalConvBlock_v2)):
                        for param in module.parameters():
                            param.requires_grad = False
                        frozen_cnt += sum(p.numel() for p in module.parameters())
                        frozen_names.append(type(module).__name__)
                types_str = ", ".join(sorted(set(frozen_names)))
                print(f"  └─ 时序模块冻结 ({types_str}): {frozen_cnt/1e6:.2f}M 参数")
                spatial_cnt = sum(
                    p.numel() for p in self.unet.parameters() if p.requires_grad
                )
                print(f"  └─ 空间模块+HFP 可训练: {spatial_cnt/1e6:.2f}M 参数")

        # VAE 始终冻结
        print("VAE: 冻结")
    
    def _controlnets(self):
        nets = [self.video_controlnet]
        if getattr(self, "controlnet_vi", None) is not None:
            nets.append(self.controlnet_vi)
        return nets

    def get_trainable_parameters(self):
        """返回所有可训练参数 (两路 ControlNet + UNet，含 HFP)"""
        params = []
        for ctrl in self._controlnets():
            for param in ctrl.parameters():
                if param.requires_grad:
                    params.append(param)
        for param in self.unet.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def get_temporal_parameters(self):
        """返回 UNet 时序模块的可训练参数 (TemporalTransformer + TemporalConvBlock_v2)"""
        from video_to_video.modules.unet_v2v import (
            TemporalTransformer, TemporalConvBlock_v2
        )
        temporal_params = []
        temporal_param_ids = set()
        for module in self.unet.modules():
            if isinstance(module, (TemporalTransformer, TemporalConvBlock_v2)):
                for p in module.parameters():
                    if p.requires_grad and id(p) not in temporal_param_ids:
                        temporal_params.append(p)
                        temporal_param_ids.add(id(p))
        return temporal_params

    def get_non_temporal_trainable_parameters(self):
        """返回 UNet + ControlNet 中除时序模块外的可训练参数"""
        temporal_ids = {id(p) for p in self.get_temporal_parameters()}
        params = []
        for ctrl in self._controlnets():
            for p in ctrl.parameters():
                if p.requires_grad and id(p) not in temporal_ids:
                    params.append(p)
        for p in self.unet.parameters():
            if p.requires_grad and id(p) not in temporal_ids:
                params.append(p)
        return params

    def get_hfp_parameters(self):
        """单独返回 HFP 参数（用于差异化学习率 / 消融统计）"""
        if not self.use_hfp:
            return []
        return list(self.unet.hfp_modules.parameters())
    
    def to_half(self):
        """
        转换为半精度。
        HFP 模块保持 fp32：DCT/FFT 在 fp16 下极易溢出（fp16 最大值仅 65504），
        HFP 的计算路径包含大量累加操作，必须用 fp32 保证数值稳定。
        """
        for ctrl in self._controlnets():
            ctrl.half()
        self.unet.half()
        if self.use_hfp:
            self.unet.hfp_modules.float()  # HFP 恢复 fp32
    
    def eval_mode(self):
        """评估模式"""
        for ctrl in self._controlnets():
            ctrl.eval()
        self.unet.eval()
    
    def train_mode(self):
        """训练模式"""
        for ctrl in self._controlnets():
            if any(p.requires_grad for p in ctrl.parameters()):
                ctrl.train()
            else:
                ctrl.eval()
        
        has_trainable_unet = any(p.requires_grad for p in self.unet.parameters())
        if has_trainable_unet:
            self.unet.train()
        else:
            self.unet.eval()
    
    # ============ VAE 编码/解码 ============
    
    def vae_encode(self, video, chunk_size=1):
        """
        VAE编码
        video: (B, T, C, H, W) -> (B, C, T, H/8, W/8)
        """
        num_f = video.shape[1]
        video = rearrange(video, "b f c h w -> (b f) c h w")
        video = video.to(dtype=torch.float16)
        z_list = []
        for ind in range(0, video.shape[0], chunk_size):
            z_list.append(self.vae.encode(video[ind:ind+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        return z * self.vae.config.scaling_factor
    
    def vae_decode(self, z, num_f):
        """VAE解码"""
        z = z.to(dtype=torch.float16)
        return self.vae.decode(z / self.vae.config.scaling_factor, num_frames=num_f).sample
    
    def vae_decode_chunk(self, z, chunk_size=3):
        """分块VAE解码"""
        z = rearrange(z, "b c f h w -> (b f) c h w")
        z = z.to(dtype=torch.float16)
        video = []
        for ind in range(0, z.shape[0], chunk_size):
            num_f = z[ind:ind+chunk_size].shape[0]
            video.append(self.vae_decode(z[ind:ind+chunk_size], num_f))
        return torch.cat(video)
    
    # ============ 前向传播 ============
    
    def forward(self, x_t, t, ir_latent, return_block_features: bool = False,
                vi_frames=None, _precomputed_flows=None,
                ir_anchor_idx: int = 0, **kwargs):
        """
        前向传播 (Flow Matching: VI → Fusion, 策略 B, 支持 anchor 在中间帧)

        策略 B 数据流:
          1. ControlNet 接收静态 IR latent → 多尺度 skip connections
             (8 帧 skip 中 IR 信息空间位置都在 VI[ir_anchor_idx] 的位置)
          2. RAFT 估计 VI 帧间双向光流
          3. 以 ir_anchor_idx 为锚点, 双向累积光流 warp 每个 skip → 空间对齐
          4. UNet 接收对齐后的 skip → 预测 velocity

        Args:
            x_t          : (B, C, T, H/s, W/s) 当前插值状态
            t            : (B,) timestep (0~1)
            ir_latent    : (B, C, T, H/s, W/s) 静态 IR latent (ControlNet hint)
            return_block_features: 是否返回编码器中间特征
            vi_frames    : (B, T, 3, H_pix, W_pix) VI 像素帧, [-1, 1]
                           提供时启用 skip connection warp
            _precomputed_flows : 预计算光流, 推理时复用避免重复 RAFT
                                 接受三种格式 (向后兼容):
                                 - (flows_fwd, flows_bwd) tuple: 双向, 推荐
                                 - list of fwd flows: 旧接口, 内部按需补后向
                                 - None: 内部估计
            ir_anchor_idx: int, IR 在窗口内 0-indexed 锚点位置 (默认 0 兼容旧训练)
                           当数据加载用 middle 模式时应传 (T-1)//2 = 3

        Returns:
            velocity: (B, C, T, H/s, W/s)
            block_features (可选)
        """
        batch, _, f, _, _ = x_t.shape

        block_sample_stride  = kwargs.pop('block_sample_stride',  3)
        spatial_only         = kwargs.pop('spatial_only_diversity', True)

        first_param_ctrl = next(self.video_controlnet.parameters())
        if first_param_ctrl.dtype == torch.float16:
            x_t = x_t.half()
            ir_latent = ir_latent.half()
            t = t.half()

        # 1. ControlNet 处理静态 IR latent (hint)
        ctrl_signals = self.video_controlnet(x_t, t, None, hint=ir_latent, **kwargs)

        # 2. Warp skip connections (双向, anchor=ir_anchor_idx)
        if self.ir_flow_prop is not None and vi_frames is not None:
            # 仅光流估计冻结；warp 本身必须保留可导链路，
            # 否则会切断 ctrl_signals -> ControlNet 的反向传播。
            with torch.no_grad():
                flows_fwd, flows_bwd = self._resolve_flows(
                    vi_frames, _precomputed_flows, need_bwd=(ir_anchor_idx > 0)
                )
            H_pix, W_pix = vi_frames.shape[3], vi_frames.shape[4]
            ctrl_signals = self.ir_flow_prop.warp_ctrl_signals(
                ctrl_signals, flows_fwd, H_pix, W_pix, f,
                anchor_idx=int(ir_anchor_idx),
                flows_bwd=flows_bwd,
            )

        # 3. UNet 前向传播
        output = self._forward_unet(
            x_t, t, None, ctrl_signals, f,
            return_block_features=return_block_features,
            block_sample_stride=block_sample_stride,
            spatial_only_diversity=spatial_only,
            **kwargs
        )

        return output

    def _resolve_flows(self, vi_frames, _precomputed_flows, need_bwd: bool):
        """
        解析 _precomputed_flows 缓存, 按需估计后向光流, 统一返回 (flows_fwd, flows_bwd or None).
        - 输入是 tuple (fwd, bwd): 直接复用
        - 输入是 list (旧接口): 当作 fwd, 后向按需现算
        - 输入是 None: 全部现算
        """
        if isinstance(_precomputed_flows, tuple) and len(_precomputed_flows) == 2:
            return _precomputed_flows
        if _precomputed_flows is None:
            if need_bwd:
                return self.ir_flow_prop.estimate_bidirectional_flows(vi_frames)
            return self.ir_flow_prop.estimate_flow(vi_frames), None
        # list of forward flows (旧调用)
        flows_fwd = _precomputed_flows
        if need_bwd:
            _, flows_bwd = self.ir_flow_prop.estimate_bidirectional_flows(vi_frames)
            return flows_fwd, flows_bwd
        return flows_fwd, None
    
    @staticmethod
    def _block_has_temporal_attn(block) -> bool:
        """判断一个 UNet 块是否包含时间注意力模块（TemporalTransformer）"""
        for module in block.modules():
            if isinstance(module, TemporalTransformer):
                return True
        return False

    def _forward_unet(self, x, t, y, control, num_frames,
                      return_block_features: bool = False,
                      block_sample_stride: int = 3,
                      spatial_only_diversity: bool = True,
                      **kwargs):
        """
        UNet前向传播 (带控制信号注入)

        Args:
            return_block_features: 是否收集编码器块的中间特征用于多样性损失
            block_sample_stride: 每隔多少个编码器块采样一次特征（减少内存占用）

        Returns:
            velocity: (B, C, T, H/8, W/8)
            block_features (可选): 各采样块的特征列表，仅当 return_block_features=True 时返回
        """
        batch = x.shape[0]
        f = num_frames
        device = x.device
        
        self.unet.batch = batch
        
        # 确保输入与 UNet 精度匹配
        unet_dtype = next(self.unet.parameters()).dtype
        x = x.to(unet_dtype)
        t = t.to(unet_dtype)
        
        # Time embedding
        t_embed = t * 1000
        time_emb = sinusoidal_embedding(t_embed, self.unet.dim)
        time_emb = time_emb.to(self.unet.time_embed[0].weight.dtype)
        e = self.unet.time_embed(time_emb)
        e = e.repeat_interleave(repeats=f, dim=0)
        
        # Context: y=None 时 cross-attention 退化为 self-attention
        context = None
        if y is not None:
            context = y.repeat_interleave(repeats=f, dim=0)
        
        # Time relative position bias
        if self.unet.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.unet.time_rel_pos_bias(f, device=device)
        else:
            time_rel_pos_bias = None
        
        # Reshape x
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        
        # 收集块特征（仅编码器，隔行采样以控制内存）
        block_features = [] if return_block_features else None

        # 判断是否有 HFP 模块
        _has_hfp = self.use_hfp and hasattr(self.unet, 'hfp_modules')
        
        # Encoder
        xs = []
        for idx, block in enumerate(self.unet.input_blocks):
            need_hfp = _has_hfp and str(idx) in self.unet.hfp_modules

            if need_hfp and isinstance(block, nn.ModuleList):
                # ── SGFE并联注入 (2+1D) ──────────────────────────────────
                # 拆开 block 子模块，HFP 与 TemporalTransformer 并联：
                #   ResBlock → SpatialTransformer → x_spatial
                #     ├──→ TemporalTransformer(x_spatial) → x + Δtemporal
                #     └──→ TemporalHFP(x_spatial)         → Δhfp (时序一致)
                #   result = x + Δtemporal + Δhfp
                hfp_mod = self.unet.hfp_modules[str(idx)]
                hfp_injected = False

                for sub_module in block:
                    if isinstance(sub_module, TemporalTransformer):
                        x_pre = x  # 保存 TemporalTransformer 前的特征给 HFP
                        x = self.unet._forward_single(
                            sub_module, x, e, context,
                            time_rel_pos_bias, None, None
                        )
                        # 并联: HFP 3D 在同一输入上独立计算
                        orig_dtype = x.dtype
                        with torch.amp.autocast('cuda', enabled=False):
                            hfp_out = hfp_mod(x_pre.float(), batch_size=batch)
                        x = x + hfp_out.to(orig_dtype)
                        hfp_injected = True
                    else:
                        x = self.unet._forward_single(
                            sub_module, x, e, context,
                            time_rel_pos_bias, None, None
                        )

                # 兜底: block 无 TemporalTransformer 时，退回残差注入
                if not hfp_injected:
                    orig_dtype = x.dtype
                    with torch.amp.autocast('cuda', enabled=False):
                        hfp_out = hfp_mod(x.float(), batch_size=batch)
                    x = x + hfp_out.to(orig_dtype)
            else:
                # 无 HFP 或非 ModuleList 块（如 Downsample），正常执行
                x = self.unet._forward_single(
                    block, x, e, context, time_rel_pos_bias, None, None
                )

            xs.append(x)
            # 采集特征：每隔 stride 采一次，并根据策略过滤时间块
            if return_block_features and (idx % block_sample_stride == 0):
                is_temporal = self._block_has_temporal_attn(block)
                if not spatial_only_diversity or not is_temporal:
                    block_features.append(x)
        
        # Middle（中间块通常包含时间注意力，spatial_only 时跳过）
        for block in self.unet.middle_block:
            x = self.unet._forward_single(
                block, x, e, context, time_rel_pos_bias, None, None
            )
        
        if return_block_features and not spatial_only_diversity:
            # 仅在不限制空间时才采集中间块（中间块含时间注意力）
            block_features.append(x)
        
        # 注入 middle control
        if control is not None:
            control = [c.to(x.dtype) for c in control]
            x = control.pop() + x
        
        # Decoder
        for block in self.unet.output_blocks:
            if control is None:
                x = torch.cat([x, xs.pop()], dim=1)
            else:
                x = torch.cat([x, xs.pop() + control.pop()], dim=1)
            x = self.unet._forward_single(
                block, x, e, context, time_rel_pos_bias, None, None,
                reference=xs[-1] if len(xs) > 0 else None
            )
        
        # Head
        x = self.unet.out(x)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)
        
        if return_block_features:
            return x, block_features
        return x
    
    def forward_diffusion(self, x_t, t, vi_latent, ir_latent,
                          vi_frames=None, _precomputed_flows=None,
                          ir_anchor_idx: int = 0):
        """从带噪可见光状态预测融合 latent。

        x_t       : sqrt(ᾱ_t) z_VI + sqrt(1-ᾱ_t) ε，主 UNet 输入
        t         : (B,) 整数扩散步
        vi_latent : 保留接口。可见光已经在 x_t 里，不再走第二条 ControlNet
        ir_latent : 红外条件。ControlNet 之后用光流把残差对齐到各 VI 帧
        返回预测的融合 latent，形状与 x_t 相同。
        """
        del vi_latent
        batch, _, f, _, _ = x_t.shape
        t = t.to(device=x_t.device).float()
        # ControlNet 直接用整数步做时间嵌入。
        # 主 UNet 这条路径内部会再乘 1000，所以这里先除回去，两边看到的是同一步。
        t_unet = t / float(self.train_scheduler.config.num_train_timesteps)

        ctrl_ir = self.video_controlnet(x_t, t, None, hint=ir_latent)

        if self.ir_flow_prop is not None and vi_frames is not None:
            with torch.no_grad():
                flows_fwd, flows_bwd = self._resolve_flows(
                    vi_frames, _precomputed_flows, need_bwd=(ir_anchor_idx > 0)
                )
            h_pix, w_pix = vi_frames.shape[3], vi_frames.shape[4]
            ctrl_ir = self.ir_flow_prop.warp_ctrl_signals(
                ctrl_ir, flows_fwd, h_pix, w_pix, f,
                anchor_idx=int(ir_anchor_idx),
                flows_bwd=flows_bwd,
            )

        return self._forward_unet(x_t, t_unet, None, ctrl_ir, f)

    def _alpha_bar(self, timestep, ref):
        alphas = self.train_scheduler.alphas_cumprod.to(device=ref.device, dtype=torch.float32)
        index = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        alpha = alphas[index]
        while alpha.ndim < ref.ndim:
            alpha = alpha.unsqueeze(-1)
        return alpha

    def ddim_step_from_x0(self, x_t, t, t_prev, z0_pred):
        """用预测的融合 latent 做确定性 DDIM，不把网络输出当作噪声。

        ε̂ = (x_t - sqrt(ᾱ_t) ẑ_F) / sqrt(1-ᾱ_t)
        x_{t-1} = sqrt(ᾱ_{t-1}) ẑ_F + sqrt(1-ᾱ_{t-1}) ε̂

        时间表走完后，上一步用 ᾱ_0，不把 ᾱ 设成 1。
        """
        z0_pred = z0_pred.float()
        x_t = x_t.float()
        alpha_t = self._alpha_bar(t, x_t)
        alpha_prev = self._alpha_bar(0 if t_prev is None else t_prev, x_t)
        eps_pred = (x_t - alpha_t.sqrt() * z0_pred) / (1.0 - alpha_t).sqrt().clamp(min=1e-8)
        return alpha_prev.sqrt() * z0_pred + (1.0 - alpha_prev).sqrt() * eps_pred

    def predict_x0(self, x_t, t, eps):
        """从噪声预测还原 z0。t 是整数扩散步。"""
        alpha = self.train_scheduler.alphas_cumprod.to(device=x_t.device, dtype=torch.float32)[t.long()]
        while alpha.ndim < x_t.ndim:
            alpha = alpha.unsqueeze(-1)
        eps = eps.float()
        x_t = x_t.float()
        return (x_t - (1.0 - alpha).sqrt() * eps) / alpha.sqrt().clamp(min=1e-8)

    @torch.no_grad()
    def sample_diffusion(self, vi_latent, ir_latent, num_steps=20,
                         vi_frames=None, ir_anchor_idx: int = 0,
                         strength: float = 1.0):
        """从带噪的可见光 latent 做 DDIM。网络每步预测融合 latent x0。

        初始化和训练用同一个前向过程：x = sqrt(ᾱ) z_VI + sqrt(1-ᾱ) ε。
        strength=1 从推理表上最吵的一步开始。
        """
        batch = vi_latent.shape[0]
        device = vi_latent.device
        scheduler = self.infer_scheduler
        scheduler.set_timesteps(num_steps, device=device)
        strength = float(min(max(strength, 1e-3), 1.0))
        init_timestep = max(1, min(int(num_steps * strength), num_steps))
        t_start = max(num_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        precomputed_flows = None
        if self.ir_flow_prop is not None and vi_frames is not None:
            if ir_anchor_idx > 0:
                precomputed_flows = self.ir_flow_prop.estimate_bidirectional_flows(vi_frames)
            else:
                precomputed_flows = self.ir_flow_prop.estimate_flow(vi_frames)

        t0 = timesteps[0]
        t0_value = int(t0.item()) if torch.is_tensor(t0) else int(t0)
        t0_batch = torch.full((batch,), t0_value, device=device, dtype=torch.long)
        noise = torch.randn_like(vi_latent)
        x = self.train_scheduler.add_noise(vi_latent.float(), noise.float(), t0_batch)
        timestep_list = [int(t.item()) if torch.is_tensor(t) else int(t) for t in timesteps]
        for index, t_value in enumerate(timestep_list):
            t_prev = timestep_list[index + 1] if index + 1 < len(timestep_list) else None
            t_batch = torch.full((batch,), t_value, device=device, dtype=torch.long)
            z0_pred = self.forward_diffusion(
                x, t_batch, vi_latent, ir_latent,
                vi_frames=vi_frames,
                _precomputed_flows=precomputed_flows,
                ir_anchor_idx=int(ir_anchor_idx),
            )
            x = self.ddim_step_from_x0(x, t_value, t_prev, z0_pred)
        return x

    # ============ Flow Matching 采样 (从 VI 开始) ============
    
    @torch.no_grad()
    def sample(self, vi_latent, ir_latent, num_steps=20, time_shift=13.0,
               vi_frames=None, ir_anchor_idx: int = 0):
        """
        Flow Matching 采样 (从 vi_latent 开始积分, 支持双向 warp)

        Args:
            vi_latent    : (B, C, T, H/s, W/s) VI latent (起点)
            ir_latent    : (B, C, T, H/s, W/s) IR latent (ControlNet hint)
            num_steps    : 采样步数
            time_shift   : 时间偏移参数
            vi_frames    : (B, T, 3, H_pix, W_pix) VI 像素帧, [-1, 1]
            ir_anchor_idx: int, IR 锚点 0-indexed 位置 (默认 0 兼容旧推理)

        Returns:
            x: (B, C, T, H/s, W/s) 融合后的 latent
        """
        B, C, T, H, W = vi_latent.shape
        device = vi_latent.device

        self.noise_scheduler.set_timesteps(num_steps, time_shift=time_shift, device=device)

        # 预计算光流, 采样循环中复用. anchor 在中间帧 → 必须双向
        precomputed_flows = None
        if self.ir_flow_prop is not None and vi_frames is not None:
            if ir_anchor_idx > 0:
                precomputed_flows = self.ir_flow_prop.estimate_bidirectional_flows(vi_frames)
            else:
                precomputed_flows = self.ir_flow_prop.estimate_flow(vi_frames)

        x = vi_latent.clone()

        for t in self.noise_scheduler.timesteps:
            t_batch = t.unsqueeze(0).repeat(B)
            v_pred = self.forward(x, t_batch, ir_latent,
                                  vi_frames=vi_frames,
                                  _precomputed_flows=precomputed_flows,
                                  ir_anchor_idx=int(ir_anchor_idx))
            x = self.noise_scheduler.step(v_pred, t, x)

        return x
    
    # ============ 推理接口 ============
    
    @torch.no_grad()
    def test(self,
             input: Dict[str, Any],
             steps=20,
             time_shift=13.0):
        """
        推理/测试
        
        Args:
            input: 字典，包含:
                - 'vi_video': 可见光视频 (B, 3, T, H, W) - 4帧
                - 'ir_video': 红外视频 (B, 3, 1, H, W) - 1帧 (会复制到T帧)
                - 'target_res': (H, W) (可选)
        """
        vi_video = input['vi_video']
        ir_video = input['ir_video']
        target_h, target_w = input.get('target_res', vi_video.shape[-2:])
        
        if vi_video.dim() == 4:
            vi_video = vi_video.unsqueeze(0)
        
        B, C, T, H, W = vi_video.shape
        
        # Resize
        vi_video = self._resize_video(vi_video, target_h, target_w)
        
        # IR: 如果是1帧，复制到T帧
        if ir_video.dim() == 3:
            ir_video = ir_video.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
        elif ir_video.dim() == 4:
            ir_video = ir_video.unsqueeze(2)  # (B, C, 1, H, W)
        ir_video = ir_video.repeat(1, 1, T, 1, 1)  # (B, C, T, H, W)
        ir_video = self._resize_video(ir_video, target_h, target_w)
        
        # Padding
        _, _, _, h, w = vi_video.shape
        padding = self._compute_padding(h, w)
        vi_video = self._pad_video(vi_video, padding)
        ir_video = self._pad_video(ir_video, padding)
        
        vi_video = vi_video.to(self.device).half()
        ir_video = ir_video.to(self.device).half()
        
        # VAE 编码
        vi_video_perm = vi_video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        vi_latent = self.vae_encode(vi_video_perm)
        
        ir_video_perm = ir_video.permute(0, 2, 1, 3, 4)
        ir_latent = self.vae_encode(ir_video_perm)
        
        # VI 像素帧供 skip connection warp (策略 B)
        vi_frames = vi_video_perm.float()
        
        with amp.autocast(enabled=True):
            gen_latent = self.sample(vi_latent, ir_latent, num_steps=steps,
                                     time_shift=time_shift, vi_frames=vi_frames)
            
            # VAE 解码
            vid_gen = self.vae_decode_chunk(gen_latent, chunk_size=T)
        
        # 移除 padding
        w1, w2, h1, h2 = padding
        vid_gen = vid_gen[:, :, h1:h+h1, w1:w+w1]
        
        gen_video = rearrange(vid_gen, '(b f) c h w -> b c f h w', b=B)
        
        return gen_video.float().cpu()
    
    # ============ 工具函数 ============
    
    def _resize_video(self, video, target_h, target_w):
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video = F.interpolate(video, [target_h, target_w], mode='bilinear')
        return video.reshape(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)
    
    def _compute_padding(self, h, w):
        target_h = ((h + 63) // 64) * 64
        target_w = ((w + 63) // 64) * 64
        return (0, target_w - w, 0, target_h - h)
    
    def _pad_video(self, video, padding):
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video = F.pad(video, padding, 'constant', 0)
        _, _, H_new, W_new = video.shape
        return video.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Video ControlNet Model 测试 (Flow Matching: VI → Fusion)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 测试 VideoControlNet
    print("\n[1/2] 测试 VideoControlNet...")
    
    controlnet = VideoControlNet().to(device).half()
    
    B, T = 1, 4
    latent_h, latent_w = 32, 32
    
    # 模拟输入
    x_t = torch.randn(B, 4, T, latent_h, latent_w, device=device, dtype=torch.float16)
    ir_latent = torch.randn(B, 4, T, latent_h, latent_w, device=device, dtype=torch.float16)
    t = torch.rand(B, device=device)
    
    print(f"  x_t (插值状态): {x_t.shape}")
    print(f"  IR latent (hint): {ir_latent.shape}")
    print(f"  timestep: {t.shape}")
    
    with torch.no_grad():
        ctrl_signals = controlnet(x_t, t, None, hint=ir_latent)
    
    print(f"  ControlNet 输出数量: {len(ctrl_signals)}")
    for i, ctrl in enumerate(ctrl_signals):
        print(f"    [{i:2d}] {ctrl.shape}")
    
    params = sum(p.numel() for p in controlnet.parameters())
    print(f"  参数量: {params / 1e6:.2f}M")
    print("  ✓ VideoControlNet 测试通过!")
    
    # 测试 UNet
    print("\n[2/2] 测试 Vid2VidSDUNet...")
    
    unet = Vid2VidSDUNet().to(device).half()
    
    print(f"  输入 x_t: {x_t.shape}")
    
    with torch.no_grad():
        unet.batch = B
        time_emb = sinusoidal_embedding(t * 1000, unet.dim)
        time_emb = time_emb.to(unet.time_embed[0].weight.dtype)
        e = unet.time_embed(time_emb)
        e = e.repeat_interleave(repeats=T, dim=0)
        
        x = rearrange(x_t, 'b c f h w -> (b f) c h w')
        
        xs = []
        for block in unet.input_blocks:
            x = unet._forward_single(block, x, e, None, None, None, None)
            xs.append(x)
        
        print(f"  Encoder 输出数量: {len(xs)}")
        for i, feat in enumerate(xs):
            print(f"    [{i:2d}] {feat.shape}")
    
    params = sum(p.numel() for p in unet.parameters())
    print(f"  参数量: {params / 1e6:.2f}M")
    print("  ✓ Vid2VidSDUNet 测试通过!")
    
    # 清理
    del controlnet, unet
    torch.cuda.empty_cache()

    if device.type == 'cuda':
        print(f"\n  GPU显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

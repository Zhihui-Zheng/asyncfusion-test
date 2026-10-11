#!/usr/bin/env python
# coding=utf-8
"""
第一阶段 Flow Matching 训练脚本 (LLVIP 数据集) - Y通道 + TVT VAED4 (4x) 版
python Fusion2/train_latent_vae33.py --use_hfp 放开
特性:
1. 使用 TVT 论文 (ICCV2025) 预训练的 VAED4 (真正的 4x VAE), 大幅减少伪影
2. 所有图片转灰度(Y通道)后复制为3通道送入VAE → 输入输出均为灰度
3. latent 通道数保持 4, UNet/ControlNet 架构不变
4. ControlNet 和 UNet 都需要训练 (因为 latent 空间分辨率变化)

架构:
  - VI视频 (4帧) → 灰度Y通道→3ch复制 → VAED4编码 → vi_latent (B,4,T,H/4,W/4)
  - GT视频 (4帧) → 灰度Y通道→3ch复制 → VAED4编码 → gt_latent
  - IR红外 (1帧→复制4帧) → 灰度→3ch复制 → VAED4编码 → ir_latent → ControlNet (hint)
  - Flow Matching: velocity = gt_latent - vi_latent

颜色策略 (推理时):
  - 模型只负责预测灰度融合亮度 (Y 通道)
  - 颜色 (Cb, Cr) 直接从原始可见光 RGB 图像中提取，替换 Y 后转回 RGB

损失函数:
  - loss_velocity: Flow Matching 主损失 (MSE on velocity)
  - loss_diversity: 块间表征多样性损失 (鼓励各编码器块学习不同特征)

VAED4 来源: https://huggingface.co/Joypop/TVTSR/tree/main/ckp
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.cuda.amp as amp
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from einops import rearrange

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from Fusion4.dual_modal_fusion_model import VideoControlNetModel
from Fusion4.flow_match import FlowMatchDiscreteScheduler
from Fusion4.utils.loss import L_RepDiversity

# TVT VAED4 (预训练的 4x VAE)
tvt_path = os.path.join(base_path, 'TVT', 'TVT-main', 'TVT')
sys.path.insert(0, tvt_path)
from VAED4.autoencoder_kl import AutoencoderKL as VAED4AutoencoderKL

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import types

check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


# ============ 图片转灰度3通道 (Y通道) ============

def bgr_to_gray_3ch(img_bgr):
    """BGR图片 → 灰度(Y通道) → 复制为3通道 (用于VAE输入)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H, W)
    return np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)


# ============ TVT VAED4 加载 (预训练的 4x VAE, ICCV2025) ============

def load_vaed4(vae_ckpt_path):
    """
    加载 TVT 论文的预训练 VAED4 (4x 下采样 VAE)
    
    架构特点:
      - block_out_channels=(128, 256, 256), 2次下采样=4x
      - 去掉了 mid_block (无 self-attention), 更轻量
      - latent_channels=4, 与 SD UNet 兼容
      - 在 OpenImage + LSDIR 上端到端训练
    
    Args:
        vae_ckpt_path: TVT 的 vae.ckpt 路径
    
    Returns:
        VAED4 模型
    """
    print("=" * 60)
    print("加载 TVT VAED4 (预训练 4x VAE, ICCV2025)")
    print("=" * 60)
    
    # 1. 从 config 创建模型
    config_path = os.path.join(tvt_path, 'VAED4', 'config2.json')
    print(f"  Config: {config_path}")
    vae = VAED4AutoencoderKL.from_config(
        VAED4AutoencoderKL.load_config(config_path)
    )
    
    # 2. 加载预训练权重
    print(f"  权重: {vae_ckpt_path}")
    vae_ckp = torch.load(vae_ckpt_path, map_location='cpu', weights_only=False)
    
    # 3. 提取 VAED4 权重 (checkpoint 中以 'vaeDe4.' 为前缀)
    ckp_dict = {}
    for n in vae_ckp['state_dict'].keys():
        if "vaeDe4" in n:
            ckp_dict[n.replace('vaeDe4.', '')] = vae_ckp['state_dict'][n]
    
    vae.load_state_dict(ckp_dict, strict=True)
    
    param_count = sum(p.numel() for p in vae.parameters())
    print(f"  参数量: {param_count / 1e6:.2f}M")
    print(f"  架构: block_out_channels=(128, 256, 256), 无 mid_block")
    print(f"  下采样: 4x (128×128 → 32×32 latent, 4通道)")
    print("=" * 60)
    
    del vae_ckp
    torch.cuda.empty_cache()
    
    return vae


def replace_vae_to_4x(model, vae_ckpt_path):
    """
    将模型的 8x VAE 替换为 TVT VAED4 (预训练 4x VAE)
    
    Args:
        vae_ckpt_path: TVT 的 vae.ckpt 路径
    """
    # 加载 TVT VAED4
    vae_4x = load_vaed4(vae_ckpt_path)
    vae_4x.eval()
    vae_4x.requires_grad_(False)
    # 注意: TVT VAED4 目前只支持 fp32 (TODO: fp16 VAED4)
    vae_4x.to(model.device, dtype=torch.float32)
    
    # 替换
    del model.vae
    torch.cuda.empty_cache()
    model.vae = vae_4x
    
    # 修补 vae_encode: VAED4 不是 TemporalDecoder, 接口略有不同
    def vae_encode_4x(self, video, chunk_size=1):
        """VAE编码 (VAED4): (B, T, C, H, W) -> (B, C, T, H/4, W/4)"""
        num_f = video.shape[1]
        video = rearrange(video, "b f c h w -> (b f) c h w")
        video = video.to(dtype=torch.float32)
        z_list = []
        for ind in range(0, video.shape[0], chunk_size):
            z_list.append(self.vae.encode(video[ind:ind+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        return z * self.vae.config.scaling_factor
    
    def vae_decode_4x(self, z, num_f):
        """VAE解码 (VAED4)"""
        z = z.to(dtype=torch.float32)
        return self.vae.decode(z / self.vae.config.scaling_factor).sample
    
    def vae_decode_chunk_4x(self, z, chunk_size=3):
        """分块VAE解码 (VAED4)"""
        B = z.shape[0]
        z = rearrange(z, "b c f h w -> (b f) c h w")
        z = z.to(dtype=torch.float32)
        video = []
        for ind in range(0, z.shape[0], chunk_size):
            chunk = z[ind:ind+chunk_size]
            num_f = chunk.shape[0]
            video.append(self.vae_decode(chunk, num_f))
        return torch.cat(video)
    
    model.vae_encode = types.MethodType(vae_encode_4x, model)
    model.vae_decode = types.MethodType(vae_decode_4x, model)
    model.vae_decode_chunk = types.MethodType(vae_decode_chunk_4x, model)
    
    print("  ✓ VAE 已替换为 TVT VAED4 (预训练 4x VAE)")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Stage 1: Flow Matching Training (LLVIP)")
    
    # Model - 分别加载 ControlNet 和 UNet
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=r"E:\paper2\codebase\STAR\outputs\326_vae4x\checkpoint2-3000/model.safetensors",
        help="预训练权重路径 (用于加载 ControlNet)",
    )
    parser.add_argument(
        "--unet_checkpoint",
        type=str,
        default=r"E:\paper2\codebase\STAR\outputs\326_vae4x\checkpoint2-3000\model_1.safetensors",
        help="UNet checkpoint 路径 (用于覆盖 UNet 权重)",
    )
    parser.add_argument(
        "--resume_checkpoint_dir",
        type=str,
        default=None,
        help="从 accelerator checkpoint 恢复训练 (包含 model.safetensors + model_1.safetensors)",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid",
        help="VAE 权重路径",
    )
    
    # VAE
    parser.add_argument("--vae_4x_ckpt", type=str,
                        default=r"pretrained_weight/vae_4x_tvt.ckpt",
                        help="TVT VAED4 权重路径 (预训练 4x VAE)")
    
    # 冻结策略 (4x VAE 方案需要训练 ControlNet + UNet)
    parser.add_argument("--freeze_controlnet", action="store_true", default=False)
    parser.add_argument("--freeze_unet", action="store_true", default=False)
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/326_vae4x",
        help="输出目录",
    )
    
    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--start_step", type=int, default=0, help="起始 step (4x VAE 从头训练)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Learning rate
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    
    # Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")

    # Data - LLVIP 数据集
    parser.add_argument("--vi_data_dir", type=str, 
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\visible1")  #E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP2\visible
    parser.add_argument("--ir_data_dir", type=str, 
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\infrared")
    parser.add_argument("--gt_data_dir", type=str, 
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\Fusion")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--video_size", type=int, nargs=2, default=[256,256])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    
    # Flow Matching
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--time_shift", type=float, default=13.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    
    # 块间表征多样性损失
    parser.add_argument("--diversity_loss_weight", type=float, default=0.01,
                        help="块间表征多样性损失权重（建议 0.01~0.1，时间块自动排除）")
    parser.add_argument("--diversity_start_step", type=int, default=0,
                        help="从第几步开始加入多样性损失（前期热身）")
    parser.add_argument("--diversity_block_stride", type=int, default=3,
                        help="编码器块采样间隔，每隔多少块收集一次特征")
    parser.add_argument("--diversity_spatial_only", action="store_true", default=True,
                        help="仅对空间块计算多样性损失，跳过含时间注意力的块（维持时序一致性）")

    # ── HFP (High-Frequency Perception, AAAI 2025) ──────────────────────────
    parser.add_argument("--use_hfp", action="store_true", default=False,
                        help="在编码器指定块后插入 High-Frequency Perception 模块")
    parser.add_argument("--hfp_block_indices", type=int, nargs="+", default=[1, 4, 7],
                        help="要插入 HFP 的编码器块索引（默认: 1 4 7，对应 320/640/1280ch 各一个）")
    parser.add_argument("--hfp_ratio", type=float, nargs=2, default=[0.25, 0.25],
                        metavar=("H_RATIO", "W_RATIO"),
                        help="DCT 低频过滤比例，默认 0.25 0.25（过滤左上角25%%×25%%低频，保留高频）")
    parser.add_argument("--hfp_patch", type=int, nargs=2, default=[4, 4],
                        metavar=("H_PATCH", "W_PATCH"),
                        help="DctChannelInteraction 池化 patch 大小，默认 4 4")
    parser.add_argument("--hfp_no_dct", action="store_true", default=False,
                        help="HFP 退化为普通空间/通道注意力（不使用 DCT），用于消融实验")
    parser.add_argument("--hfp_residual", action="store_true", default=True,
                        help="HFP 以残差方式注入 x=x+HFP(x)（默认），加 --hfp_serial 切换为直接串联")
    parser.add_argument("--hfp_serial", action="store_true", default=False,
                        help="HFP 直接串联 x=HFP(x)（消融实验用，默认为残差模式）")
    parser.add_argument("--hfp_init", type=str, default="zeros",
                        choices=["ones", "zeros", "default"],
                        help="HFP 权重初始化策略: zeros=零初始化(残差模式默认，初始等效恒等映射), ones=dirac近似恒等, default=PyTorch默认")
    parser.add_argument("--hfp_train_only", action="store_true", default=False,
                        help="仅训练 HFP 参数，冻结 UNet 和 ControlNet（需配合 --use_hfp）")

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="权重保存间隔")
    parser.add_argument("--validation_steps", type=int, default=100, help="验证图片生成间隔")
    
    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    
    # Logging
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


class VideoDataset(torch.utils.data.Dataset):
    """
    视频数据集 (VI + IR → GT 融合任务)
    """
    def __init__(self, vi_dir, ir_dir, gt_dir, num_frames=4, video_size=(256, 256)):
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.gt_dir = gt_dir
        self.num_frames = num_frames
        self.video_size = video_size
        
        self.samples = self._find_samples()
        
        print(f"=" * 60)
        print(f"VideoDataset (LLVIP)")
        print(f"=" * 60)
        print(f"Found {len(self.samples)} video triplets")
        print(f"  VI (可见光): {vi_dir}")
        print(f"  IR (红外):   {ir_dir}")
        print(f"  GT (目标):   {gt_dir}")
        print(f"=" * 60)
    
    def _find_samples(self):
        """查找配对的视频数据样本（以 GT 为基准，匹配对应的 VI 和 IR）"""
        samples = []
        vi_root = Path(self.vi_dir)
        ir_root = Path(self.ir_dir)
        gt_root = Path(self.gt_dir)
        
        for root, label in [(vi_root, 'VI'), (ir_root, 'IR'), (gt_root, 'GT')]:
            if not root.exists():
                print(f"Warning: {label} dir {root} does not exist")
                return samples
        
        # 以 GT 文件夹为基准：GT 有多少个 clip 就配多少个，VI 多余的自动忽略
        for gt_sample_dir in sorted(gt_root.iterdir()):
            if not gt_sample_dir.is_dir():
                continue
            name = gt_sample_dir.name
            vi_sample_dir = vi_root / name
            ir_sample_dir = ir_root / name
            
            if not vi_sample_dir.exists() or not ir_sample_dir.exists():
                continue
            
            vi_frames = sorted(list(vi_sample_dir.glob('*.png')) + list(vi_sample_dir.glob('*.jpg')))
            ir_frames = sorted(list(ir_sample_dir.glob('*.png')) + list(ir_sample_dir.glob('*.jpg')))
            gt_frames = sorted(list(gt_sample_dir.glob('*.png')) + list(gt_sample_dir.glob('*.jpg')))
            
            if len(vi_frames) > 0 and len(ir_frames) > 0 and len(gt_frames) > 0:
                samples.append({
                    'vi_dir': vi_sample_dir,
                    'ir_dir': ir_sample_dir,
                    'gt_dir': gt_sample_dir,
                    'name': name,
                })
        
        return samples
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, idx):
        dummy = {
            'vi_video': torch.randn(3, self.num_frames, *self.video_size),
            'ir_video': torch.randn(3, self.num_frames, *self.video_size),
            'gt_video': torch.randn(3, self.num_frames, *self.video_size),
        }
        if len(self.samples) == 0:
            return dummy
        
        sample = self.samples[idx % len(self.samples)]
        
        vi_files_all = sorted(list(sample['vi_dir'].glob('*.png')) + list(sample['vi_dir'].glob('*.jpg')))
        ir_files_all = sorted(list(sample['ir_dir'].glob('*.png')) + list(sample['ir_dir'].glob('*.jpg')))
        gt_files_all = sorted(list(sample['gt_dir'].glob('*.png')) + list(sample['gt_dir'].glob('*.jpg')))
        
        # 按文件名 stem 建立映射，对 VI 和 GT 求交集，保证时间对齐
        vi_map = {f.stem: f for f in vi_files_all}
        gt_map = {f.stem: f for f in gt_files_all}
        ir_map = {f.stem: f for f in ir_files_all}
        
        common_stems = sorted(set(vi_map.keys()) & set(gt_map.keys()))
        
        if len(common_stems) == 0:
            # 文件名完全不同时退回按位置匹配（兜底），取 VI 和 GT 的最小长度防越界
            num_common = min(len(vi_files_all), len(gt_files_all))
            vi_files = vi_files_all[:num_common]
            gt_files = gt_files_all[:num_common]
            ir_files = ir_files_all
        else:
            vi_files = [vi_map[s] for s in common_stems]
            gt_files = [gt_map[s] for s in common_stems]
            # IR 按同名匹配，找不到则按位置兜底
            ir_files = [
                ir_map[s] if s in ir_map else ir_files_all[min(i, len(ir_files_all) - 1)]
                for i, s in enumerate(common_stems)
            ]
        
        num_files = len(vi_files)  # vi_files 和 gt_files 等长，由上方保证
        
        if num_files == 0:
            return dummy
        
        # 按 num_frames 分组选择 (0-3一组, 4-7一组, ...),避免跨组
        num_groups = num_files // self.num_frames
        if num_groups > 0:
            group_idx = np.random.randint(0, num_groups)
            start_idx = group_idx * self.num_frames
            indices = np.arange(start_idx, start_idx + self.num_frames)
        else:
            # 帧数不足一组，用所有帧均匀采样
            indices = np.linspace(0, num_files - 1, self.num_frames).astype(int)
        
        vi_frames = self._load_frames_by_indices(vi_files, indices)
        gt_frames = self._load_frames_by_indices(gt_files, indices)
        
        # IR: 取与 VI 序列起始帧同名的 IR 帧，复制 T 份
        ir_start_idx = indices[0]
        ir_frame = self._load_single_frame(ir_files, ir_start_idx)
        ir_frames = ir_frame.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        
        return {
            'vi_video': vi_frames,
            'ir_video': ir_frames,
            'gt_video': gt_frames,
        }
    
    def _load_frames_by_indices(self, frame_files, indices):
        frames = []
        for i in indices:
            img = cv2.imread(str(frame_files[i]))
            img = bgr_to_gray_3ch(img)  # 灰度(Y通道) → 3ch
            img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
            img = img.astype(np.float32) / 255.0
            img = img * 2 - 1
            frames.append(img)
        
        frames = np.stack(frames, axis=0)
        frames = np.transpose(frames, (3, 0, 1, 2))
        return torch.from_numpy(frames).float()
    
    def _load_single_frame(self, frame_files, idx):
        if idx >= len(frame_files):
            idx = len(frame_files) - 1
        img = cv2.imread(str(frame_files[idx]))
        img = bgr_to_gray_3ch(img)  # 灰度(Y通道) → 3ch
        img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
        img = img.astype(np.float32) / 255.0
        img = img * 2 - 1
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()


@torch.no_grad()
def run_validation(model, noise_scheduler, vi_video, ir_video, device, num_steps=20, time_shift=13.0):
    """运行验证推理"""
    model.eval_mode()
    
    B, C, T, H, W = vi_video.shape
    
    vi_video_device = vi_video.to(device)
    ir_video_device = ir_video.to(device)
    
    vi_video_perm = vi_video_device.permute(0, 2, 1, 3, 4)
    vi_latent = model.vae_encode(vi_video_perm)
    
    ir_video_perm = ir_video_device.permute(0, 2, 1, 3, 4)
    ir_latent = model.vae_encode(ir_video_perm)
    
    noise_scheduler.set_timesteps(num_steps, time_shift=time_shift, device=device)
    
    x = vi_latent.clone()
    
    for t_step in noise_scheduler.timesteps:
        t_batch = t_step.unsqueeze(0).repeat(B).to(device)
        v_pred = model.forward(x_t=x, t=t_batch, ir_latent=ir_latent)
        x = noise_scheduler.step(v_pred, t_step, x)
    
    pred_video = model.vae_decode_chunk(x, chunk_size=3)
    pred_video = rearrange(pred_video, '(b t) c h w -> b c t h w', b=B, t=T)
    pred_video = (pred_video + 1) / 2
    pred_video = torch.clamp(pred_video, 0, 1)
    
    model.train_mode()
    return pred_video


def save_validation_results(vi_video, ir_video, gt_video, pred_video, save_dir, step):
    """保存验证结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    T = pred_video.shape[2]
    num_frames_to_show = min(T, 4)
    indices = np.linspace(0, T - 1, num_frames_to_show).astype(int)
    
    fig, axes = plt.subplots(num_frames_to_show, 4, figsize=(12, num_frames_to_show * 2.5))
    
    if num_frames_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        # VI
        vi_frame = vi_video[0, :, idx].float().permute(1, 2, 0).cpu().numpy()
        vi_frame = (vi_frame + 1) / 2
        vi_frame = np.clip(vi_frame, 0, 1).astype(np.float32)
        axes[row, 0].imshow(vi_frame)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title('VI (起点)', fontsize=12, fontweight='bold')
        
        # IR（1帧 hint，所有行显示同一张）
        ir_frame = ir_video[0, :, 0].float().permute(1, 2, 0).cpu().numpy()
        ir_frame = (ir_frame + 1) / 2
        ir_frame = np.clip(ir_frame, 0, 1).astype(np.float32)
        axes[row, 1].imshow(ir_frame)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title('IR (hint)', fontsize=12, fontweight='bold')
        
        # GT
        gt_frame = gt_video[0, :, idx].float().permute(1, 2, 0).cpu().numpy()
        gt_frame = (gt_frame + 1) / 2
        gt_frame = np.clip(gt_frame, 0, 1).astype(np.float32)
        axes[row, 2].imshow(gt_frame)
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].set_title('GT (终点)', fontsize=12, fontweight='bold')
        
        # Pred
        pred_frame = pred_video[0, :, idx].float().permute(1, 2, 0).cpu().numpy()
        pred_frame = np.clip(pred_frame, 0, 1).astype(np.float32)
        axes[row, 3].imshow(pred_frame)
        axes[row, 3].axis('off')
        if row == 0:
            axes[row, 3].set_title('Pred', fontsize=12, fontweight='bold')
        
        axes[row, 0].set_ylabel(f'Frame {idx}', fontsize=10)
    
    plt.suptitle(f'Step {step} (VI → Fusion, LLVIP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, f"test_step{step}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  验证图片已保存: {comparison_path}")


def main(args):
    print("=" * 70)
    print("Stage 1: Flow Matching Training (LLVIP)")
    print("=" * 70)
    print(f"ControlNet 权重: {args.pretrained_model_path}")
    print(f"UNet 权重: {args.unet_checkpoint}")
    print(f"输出目录: {args.output_dir}")
    print(f"起始 step: {args.start_step}")
    print(f"目标 step: {args.max_train_steps}")
    if args.use_hfp:
        mode_str = "serial(直接串联)" if args.hfp_serial else "residual(残差+零初始化)"
        train_str = "仅HFP训练" if args.hfp_train_only else "全参数训练"
        print(f"HFP 模块: 启用  blocks={args.hfp_block_indices}  mode={mode_str}  "
              f"init={args.hfp_init}  ratio={args.hfp_ratio}  "
              f"dct={'是' if not args.hfp_no_dct else '否(消融)'}  训练策略={train_str}")
    else:
        print("HFP 模块: 禁用 (基线)")
    print("=" * 70)
    
    # ============ Accelerator ============
    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # ============ Set seed ============
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # ============ Build model ============
    logger.info("Building model...")
    
    class ModelConfig:
        def __init__(self):
            self.model_path = args.pretrained_model_path
            self.vae_path = args.vae_path
    
    opt = ModelConfig()
    # model_path=None：跳过 __init__ 里有前缀歧义的 _load_pretrained()
    # 权重在下方的"加载权重"块里按正确格式分别加载
    opt.model_path = None
    model = VideoControlNetModel(
        opt,
        device=accelerator.device,
        use_hfp=args.use_hfp,
        hfp_block_indices=tuple(args.hfp_block_indices),
        hfp_ratio=tuple(args.hfp_ratio),
        hfp_patch=tuple(args.hfp_patch),
        hfp_use_dct=not args.hfp_no_dct,
        hfp_serial=args.hfp_serial,   # 默认 False（残差模式）
        hfp_init=args.hfp_init,
    )
    
    # ============ 替换为 TVT VAED4 (预训练 4x VAE) ============
    replace_vae_to_4x(model, vae_ckpt_path=args.vae_4x_ckpt)
    
    # ============ 加载权重 ============
    if args.resume_checkpoint_dir and os.path.exists(args.resume_checkpoint_dir):
        # 从 accelerator checkpoint 恢复 (ControlNet + UNet)
        from safetensors.torch import load_file
        
        ctrl_path = os.path.join(args.resume_checkpoint_dir, "model.safetensors")
        unet_path = os.path.join(args.resume_checkpoint_dir, "model_1.safetensors")
        
        if os.path.exists(ctrl_path):
            ctrl_dict = load_file(ctrl_path)
            ctrl_dict = {k.replace('module.', ''): v for k, v in ctrl_dict.items()}
            ret = model.video_controlnet.load_state_dict(ctrl_dict, strict=False)
            logger.info(f"✓ ControlNet 从 checkpoint 恢复 (missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)})")
        
        if os.path.exists(unet_path):
            unet_dict = load_file(unet_path)
            unet_dict = {k.replace('module.', ''): v for k, v in unet_dict.items()}
            ret = model.unet.load_state_dict(unet_dict, strict=False)
            logger.info(f"✓ UNet 从 checkpoint 恢复 (missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)})")
    else:
        # 首次训练：分别加载 ControlNet 和 UNet
        from safetensors.torch import load_file as _load_file

        # ── ControlNet ────────────────────────────────────────────────────
        # accelerator 保存的 model.safetensors 里是纯 ControlNet 权重，无前缀
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            logger.info(f"加载 ControlNet checkpoint: {args.pretrained_model_path}")
            if args.pretrained_model_path.endswith('.safetensors'):
                ctrl_ckpt = _load_file(args.pretrained_model_path)
            else:
                ctrl_ckpt = torch.load(args.pretrained_model_path, map_location='cpu')
                if 'state_dict' in ctrl_ckpt:
                    ctrl_ckpt = ctrl_ckpt['state_dict']
            ctrl_ckpt = {k.replace('module.', ''): v for k, v in ctrl_ckpt.items()}
            ret = model.video_controlnet.load_state_dict(ctrl_ckpt, strict=False)
            logger.info(f"✓ ControlNet 权重加载完成 (missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)})")
        else:
            logger.warning(f"ControlNet checkpoint 不存在: {args.pretrained_model_path}")

        # ── UNet ──────────────────────────────────────────────────────────
        if args.unet_checkpoint and os.path.exists(args.unet_checkpoint):
            logger.info(f"加载 UNet checkpoint: {args.unet_checkpoint}")
            if args.unet_checkpoint.endswith('.safetensors'):
                ckpt = _load_file(args.unet_checkpoint)
            else:
                ckpt = torch.load(args.unet_checkpoint, map_location='cpu')
            unet_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
            ret = model.unet.load_state_dict(unet_dict, strict=False)
            logger.info(f"✓ UNet 权重加载完成 (missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)})")
        else:
            logger.warning(f"UNet checkpoint 不存在: {args.unet_checkpoint}")
    
    # ============ 冻结策略 ============
    train_controlnet = not args.freeze_controlnet
    train_unet = not args.freeze_unet
    train_hfp_only = args.use_hfp and args.hfp_train_only

    model.freeze_for_training(
        train_controlnet=train_controlnet,
        train_unet=train_unet,
        train_hfp_only=train_hfp_only,
    )

    if accelerator.is_main_process:
        trainable_count = sum(p.numel() for p in model.get_trainable_parameters())
        ctrl_count = sum(p.numel() for p in model.video_controlnet.parameters())
        unet_count = sum(p.numel() for p in model.unet.parameters())

        logger.info(f"Trainable: {trainable_count / 1e6:.2f}M")
        if train_hfp_only:
            hfp_count = sum(p.numel() for p in model.get_hfp_parameters())
            logger.info(f"  HFP only 模式: {hfp_count / 1e6:.2f}M  (UNet/ControlNet 冻结)")
        else:
            logger.info(f"  ControlNet: {ctrl_count / 1e6:.2f}M {'(训练)' if train_controlnet else '(冻结)'}")
            logger.info(f"  UNet: {unet_count / 1e6:.2f}M {'(训练)' if train_unet else '(冻结)'}")
    
    noise_scheduler = model.noise_scheduler
    
    # ============ 多样性损失 ============
    rep_diversity_loss_fn = L_RepDiversity(temperature=1.0).to(accelerator.device)
    use_diversity_loss = args.diversity_loss_weight > 0
    if use_diversity_loss:
        logger.info(f"块间表征多样性损失: weight={args.diversity_loss_weight}, "
                    f"start_step={args.diversity_start_step}, "
                    f"block_stride={args.diversity_block_stride}")
    
    # ============ Dataset ============
    train_dataset = VideoDataset(
        vi_dir=args.vi_data_dir,
        ir_dir=args.ir_data_dir,
        gt_dir=args.gt_data_dir,
        num_frames=args.num_frames,
        video_size=tuple(args.video_size)
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # ============ Optimizer ============
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
    
    trainable_params = model.get_trainable_parameters()
    
    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # ============ LR Scheduler ============
    # 从 start_step 开始，训练到 max_train_steps
    remaining_steps = args.max_train_steps - args.start_step
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=min(args.lr_warmup_steps, remaining_steps // 10),
        num_training_steps=remaining_steps * accelerator.num_processes,
    )
    
    # ============ Prepare with accelerator ============
    if train_controlnet and train_unet:
        model.video_controlnet, model.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model.video_controlnet, model.unet, optimizer, train_dataloader, lr_scheduler
        )
    elif train_unet:
        model.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model.unet, optimizer, train_dataloader, lr_scheduler
        )
    elif train_controlnet:
        model.video_controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model.video_controlnet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        train_dataloader, = accelerator.prepare(train_dataloader)
    
    # ============ Mixed precision ============
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    if not train_controlnet:
        model.video_controlnet.half()
    if not train_unet:
        model.unet.half()
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # ============ 从指定 step 开始 ============
    global_step = args.start_step
    
    # ============ Training loop ============
    logger.info("***** Running training *****")
    logger.info(f"  数据集样本数 = {len(train_dataset)}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  起始 step = {args.start_step}")
    logger.info(f"  目标 step = {args.max_train_steps}")
    logger.info(f"  剩余训练 steps = {remaining_steps}")
    logger.info(f"  验证图片间隔 = {args.validation_steps} steps")
    logger.info(f"  权重保存间隔 = {args.checkpointing_steps} steps")
    
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        total=args.max_train_steps,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    epoch = 0
    while global_step < args.max_train_steps:
        model.train_mode()
        
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break
            
            accumulate_modules = []
            if train_controlnet:
                accumulate_modules.append(model.video_controlnet)
            if train_unet:
                accumulate_modules.append(model.unet)
            
            accumulate_target = accumulate_modules[0] if accumulate_modules else model.unet
            
            with accelerator.accumulate(accumulate_target):
                vi_video = batch['vi_video'].to(accelerator.device, dtype=torch.float32)
                ir_video = batch['ir_video'].to(accelerator.device, dtype=torch.float32)
                gt_video = batch['gt_video'].to(accelerator.device, dtype=torch.float32)
                
                B, _, T, H, W = vi_video.shape
                
                with torch.no_grad():
                    vi_video_perm = vi_video.permute(0, 2, 1, 3, 4)
                    vi_latent = model.vae_encode(vi_video_perm).float()
                    
                    ir_video_perm = ir_video.permute(0, 2, 1, 3, 4)
                    ir_latent = model.vae_encode(ir_video_perm).float()
                    
                    gt_video_perm = gt_video.permute(0, 2, 1, 3, 4)
                    gt_latent = model.vae_encode(gt_video_perm).float()
                
                # Flow Matching
                velocity = gt_latent - vi_latent
                
                t = torch.rand(B, device=accelerator.device)
                t_expand = t.view(B, 1, 1, 1, 1)
                
                x_t = (1 - t_expand) * vi_latent + t_expand * gt_latent
                
                # 是否在本步启用多样性损失
                compute_diversity = (
                    use_diversity_loss and global_step >= args.diversity_start_step
                )
                
                with accelerator.autocast():
                    if compute_diversity:
                        pred_velocity, block_features = model.forward(
                            x_t=x_t, t=t, ir_latent=ir_latent,
                            return_block_features=True,
                            block_sample_stride=args.diversity_block_stride,
                            spatial_only_diversity=args.diversity_spatial_only,
                        )
                    else:
                        pred_velocity = model.forward(x_t=x_t, t=t, ir_latent=ir_latent)
                
                pred_velocity = pred_velocity.float()
                
                # ── 主损失：Flow Matching velocity MSE ──────────────────────
                loss_velocity = F.mse_loss(pred_velocity, velocity)
                
                # ── 块间表征多样性损失 ────────────────────────────────────────
                if compute_diversity:
                    loss_diversity = rep_diversity_loss_fn(block_features).float()
                else:
                    loss_diversity = torch.tensor(0.0, device=accelerator.device)
                
                loss = (loss_velocity
                        + args.diversity_loss_weight * loss_diversity)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[Warning] Step {global_step}: loss nan/inf, skipping...")
                    optimizer.zero_grad()
                    continue
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                logs = {
                    "loss":           loss.detach().item(),
                    "loss_vel":       loss_velocity.detach().item(),
                    "loss_diversity": loss_diversity.detach().item(),
                    "lr":             lr_scheduler.get_last_lr()[0],
                    "epoch":          epoch,
                }
                
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                accelerator.log(logs, step=global_step)
                
                # 每 1000 步生成验证图片
                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        try:
                            print(f"\n正在生成验证图片 (step {global_step})...")
                            pred_video = run_validation(
                                model=model,
                                noise_scheduler=noise_scheduler,
                                vi_video=vi_video.detach(),
                                ir_video=ir_video.detach(),
                                device=accelerator.device,
                                num_steps=args.num_inference_steps,
                                time_shift=args.time_shift,
                            )
                            
                            # 验证图片保存到 output_dir/validations/
                            val_save_dir = os.path.join(args.output_dir, "validations")
                            save_validation_results(
                                vi_video=vi_video.detach(),
                                ir_video=ir_video.detach(),
                                gt_video=gt_video.detach(),
                                pred_video=pred_video,
                                save_dir=val_save_dir,
                                step=global_step
                            )
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                
                # 每 5000 步保存权重
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
            
            if global_step >= args.max_train_steps:
                break
        
        epoch += 1
    
    # 保存最终 checkpoint
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        if not os.path.exists(final_save_path):
            accelerator.save_state(final_save_path)
            logger.info(f"Final checkpoint saved to {final_save_path}")
    
    accelerator.end_training()
    logger.info("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
使用方法:

# 从 step 10000 继续训练到 20000 (LLVIP 数据集)
python Fusion/train_latent23.py

# 或者自定义参数
python Fusion/train_latent23.py \
    --pretrained_model_path pretrained_weight/model.pt \
    --unet_checkpoint E:/paper2/codebase/STAR/outputs/Time_Fusion__training_by_latent/checkpoint-10000/model.safetensors \
    --vi_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/LLVIP/visible1 \
    --ir_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/LLVIP/infrared \
    --gt_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/LLVIP/Fusion \
    --output_dir outputs/23_latent \
    --start_step 10000 \
    --max_train_steps 20000 \
    --validation_steps 1000 \
    --checkpointing_steps 5000

说明:
- ControlNet 从 pretrained_weight/model.pt 加载
- UNet 从 checkpoint-10000/model.safetensors 加载  
- 从 step 10000 开始，训练到 step 20000
- 输出保存在 outputs/23_latent
- 每 1000 步生成验证图片 (保存在 outputs/23_latent/validations/)
- 每 5000 步保存权重 (保存在 outputs/23_latent/checkpoint-XXXX/)
"""


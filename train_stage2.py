#!/usr/bin/env python
# coding=utf-8
"""
训练脚本 — Fusion_SGFE 第二阶段（latent 前置增强 + 多步扩散 + 时序一致性损失）

数据流（纯一体）：
  VI/IR --VAE.encode(冻结)--> latent
     --(冻结)ControlNet+UNet 多步扩散积分--> pred_latent
     --LatentRestormer(可训练) 在 4 通道 latent 上增强/锐化--> pred_latent_sharp
     --VAE.decode(冻结, 梯度可穿过)--> 融合视频 [0,1]
     --无监督融合损失(SSIM/Max/Color/Grad) + 时序一致性损失 L_tc

要点：
  - 扩散 + VAE 全程冻结（不训练），仅训练 LatentRestormer；
  - 扩散用多步采样（默认 5 步，与 test_ablation.py 推理一致），在 no_grad 内完成；
  - L_tc 复用模型自带的 RAFT 光流（VI 帧间前向光流）对融合帧做 warp 一致性约束；
  - 显存：LatentRestormer 跑在 latent(低分辨率)，并对 VAE 解码器开梯度检查点。

用法:
  python Fusion_SGFE/train_stage2.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 缓解显存碎片
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from einops import rearrange
from tqdm.auto import tqdm

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

# 复用现有实现（导入即会把 Restormer archs 目录加入 sys.path）
from Fusion_SGFE.train_one_step import (
    SlidingRatioVideoDataset,
    fusion_unsupervised_loss,
    _load_safetensors_into,
    _to_01,
    LOSS_MAX_RATIO, LOSS_SSIM_RATIO, LOSS_SSIM_VIS_RATIO,
    LOSS_SSIM_IR_RATIO, LOSS_COLOR_RATIO, LOSS_GRAD_RATIO,
)
from Fusion_SGFE.dual_modal_fusion_model import VideoControlNetModel
from Fusion_SGFE.train_latent_vae48_tem import replace_vae_to_4x
from Fusion_SGFE.ir_flow_propagation import flow_warp

# Restormer 基础块（顶层导入，绕开 basicsr.models 的重型 __init__）
RESTORMER_ROOT = os.path.join(base_path, "Fusion_SGFE", "superresulation", "Restormer")
_ARCHS = os.path.join(RESTORMER_ROOT, "basicsr", "models", "archs")
for _p in (RESTORMER_ROOT, _ARCHS):
    if _p not in sys.path:
        sys.path.append(_p)
from restormer_arch import TransformerBlock, OverlapPatchEmbed

logger = get_logger(__name__)


# ============ latent 增强网络（Restormer 逻辑，跑在 4 通道 latent 上） ============

def _make_identity_temporal_conv3d(channels: int, kernel_t: int = 3) -> nn.Conv3d:
    """时间维 Conv3d，恒等(Dirac)初始化：中心抽头=单位阵，其余=0，bias=0。"""
    conv = nn.Conv3d(channels, channels, kernel_size=(kernel_t, 1, 1),
                     padding=(kernel_t // 2, 0, 0), bias=True)
    with torch.no_grad():
        nn.init.zeros_(conv.weight)
        center = kernel_t // 2
        conv.weight.data[:, :, center, 0, 0] = torch.eye(channels)
        nn.init.zeros_(conv.bias)
    return conv


class LatentRestormer(nn.Module):
    """在 4 通道 latent 上做增强/锐化的轻量 Restormer。

    - 用 Restormer 的 MDTA/GDFN TransformerBlock（latent 本身已是低分辨率，不再下采样）。
    - 内置恒等初始化的时间 Conv3d，做帧间一致，初始不改变单帧结果。
    - 输出卷积零初始化 + 全局残差：out = latent + delta，初始即恒等(=当前扩散结果)。
    - 不做 [0,1] clamp（latent 非图像，数值范围任意）。
    """

    def __init__(
        self,
        latent_ch: int = 4,
        dim: int = 128,
        num_blocks: int = 8,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        temporal_positions=(2, 4),
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_grad_checkpoint = bool(use_grad_checkpoint)

        self.patch_embed = OverlapPatchEmbed(latent_ch, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.temporal_positions = {int(p) for p in temporal_positions if 0 <= int(p) < num_blocks}
        self.temporal = nn.ModuleDict({
            str(p): _make_identity_temporal_conv3d(dim) for p in self.temporal_positions
        })
        self.output = nn.Conv2d(dim, latent_ch, kernel_size=3, padding=1, bias=bias)
        # 零初始化 → 初始 delta=0，全局残差等于恒等
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def _run_block(self, blk, x):
        if self.use_grad_checkpoint and self.training and x.requires_grad:
            return cp.checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (B, C=4, T, h, w)
        b, c, t, h, w = latent.shape
        x = rearrange(latent, "b c t h w -> (b t) c h w")
        x = self.patch_embed(x)
        for i, blk in enumerate(self.blocks):
            x = self._run_block(blk, x)
            if i in self.temporal_positions:
                xt = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
                xt = self.temporal[str(i)](xt)
                x = rearrange(xt, "b c t h w -> (b t) c h w")
        delta = self.output(x)
        delta = rearrange(delta, "(b t) c h w -> b c t h w", b=b, t=t)
        return latent + delta


# ============ 时序一致性损失 L_tc（论文式(10)） ============

def temporal_consistency_loss(pred_video, flows_fwd, eps=1e-3):
    """时序一致性损失：用相邻帧前向光流把上一融合帧 warp 到当前帧，惩罚有效区域差异。

    L_tc = mean_t || (Y_t - W(Y_{t-1}, O_{t-1->t})) ⊙ M_t ||_1 / (Σ M_t + eps)

    Args:
        pred_video: (B, C, T, H, W) 融合视频, 范围 [0,1]（可训练, 梯度回传到此）
        flows_fwd : list[(B, 2, H, W)] 长度 T-1, VI 帧间前向光流 O_{t-1->t}（已 detach）
        eps       : 数值稳定项
    Returns:
        标量损失
    """
    if flows_fwd is None or pred_video.shape[2] < 2:
        return pred_video.new_zeros(())

    B, C, T, H, W = pred_video.shape
    ones = pred_video.new_ones(B, 1, H, W)
    total = pred_video.new_zeros(())
    n = 0
    for t in range(1, T):
        flow = flows_fwd[t - 1].to(pred_video.dtype).permute(0, 2, 3, 1)  # (B,H,W,2)
        warped_prev = flow_warp(pred_video[:, :, t - 1], flow)            # 对齐 Y_{t-1} -> t
        mask = flow_warp(ones, flow)
        mask = (mask > 0.999).to(pred_video.dtype)                        # 二值有效掩码
        diff = (pred_video[:, :, t] - warped_prev).abs() * mask
        total = total + diff.sum() / (mask.sum() * C + eps)
        n += 1
    return total / max(1, n)


def _to_y_gray3_video(video_01: torch.Tensor) -> torch.Tensor:
    """RGB -> YCbCr 的 Y 通道，再复制 3 份，保持 (B,3,T,H,W)。"""
    y = (
        0.299 * video_01[:, 0:1]
        + 0.587 * video_01[:, 1:2]
        + 0.114 * video_01[:, 2:3]
    )
    return y.repeat(1, 3, 1, 1, 1)


# ============ 验证保存（连续多帧，便于看时序连续性） ============

def save_validation_video(vi_video, ir_video, pred_video, save_dir, step):
    """保存验证结果（视频）：

    - 把连续 T 帧的 [VI | IR | 融合] 拼成一张网格图（每行一帧），便于一眼看时序连续性；
    - 同时把 T 帧融合结果单独存到 step_xxxxxx/ 子目录（可按顺序当视频帧查看/合成）。

    vi_video, ir_video: (B,C,T,H,W) 范围 [-1,1]；pred_video: (B,C,T,H,W) 范围 [0,1]
    """
    os.makedirs(save_dir, exist_ok=True)
    T = pred_video.shape[2]
    h, w = pred_video.shape[-2], pred_video.shape[-1]

    vi_01 = _to_01(vi_video)
    ir_01 = _to_01(ir_video)
    if vi_01.shape[-2] != h or vi_01.shape[-1] != w:
        vi_01 = F.interpolate(vi_01[0].permute(1, 0, 2, 3), size=(h, w),
                              mode="bilinear", align_corners=False).permute(1, 0, 2, 3).unsqueeze(0)
        ir_01 = F.interpolate(ir_01[0].permute(1, 0, 2, 3), size=(h, w),
                              mode="bilinear", align_corners=False).permute(1, 0, 2, 3).unsqueeze(0)

    frame_dir = os.path.join(save_dir, f"step_{step:06d}")
    os.makedirs(frame_dir, exist_ok=True)

    rows = []
    for t in range(T):
        vi_t = vi_01[0, :, t].permute(1, 2, 0).cpu().numpy()
        ir_t = ir_01[0, :, t].permute(1, 2, 0).cpu().numpy()
        pred_t = pred_video[0, :, t].permute(1, 2, 0).cpu().numpy()
        rows.append(np.concatenate([vi_t, ir_t, pred_t], axis=1))
        cv2.imwrite(os.path.join(frame_dir, f"frame_{t:02d}.png"),
                    (np.clip(pred_t, 0, 1) * 255.0).astype(np.uint8))

    grid = np.concatenate(rows, axis=0)  # (T*h, 3*w, 3)，每行一帧
    cv2.imwrite(os.path.join(save_dir, f"step_{step:06d}_grid.png"),
                (np.clip(grid, 0, 1) * 255.0).astype(np.uint8))


# ============ 参数 ============

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Fusion_SGFE 第二阶段（latent 前置增强 + 多步扩散 + 时序损失）"
    )

    # 权重路径（与 train_one_step.py 默认对齐）
    parser.add_argument("--pretrained_model_path", type=str,
                        default=r"E:\paper2\codebase\STAR\outputs\33_vae4x\checkpoint-10000\model.safetensors")
    parser.add_argument("--unet_checkpoint", type=str,
                        default=r"E:\paper2\codebase\STAR\outputs\temporal8f_step_20260515_172242\checkpoint-1500\model_1.safetensors")
    parser.add_argument("--resume_checkpoint_dir", type=str, default=None)
    parser.add_argument("--vae_4x_ckpt", type=str, default=r"pretrained_weight/vae_4x_tvt.ckpt")
    parser.add_argument("--raft_model_path", type=str,
                        default=r"Fusion_SGFE/ProPainter/weights/raft-things.pth")

    # 数据
    parser.add_argument("--vi_data_dir", type=str,
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\visible3")
    parser.add_argument("--ir_data_dir", type=str,
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\infrared3")

    # 训练
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--vi_per_ir", type=int, default=8)
    parser.add_argument("--ir_select_mode", type=str, default="start",
                        choices=["middle", "random", "start"])
    parser.add_argument("--ir_middle_local_idx", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])

    # 多步扩散（与 test_ablation.py 推理设置对齐）
    parser.add_argument("--num_inference_steps", type=int, default=5,
                        help="扩散采样步数（默认 5，与多步推理一致）")
    parser.add_argument("--time_shift", type=float, default=3.0)

    # 时序一致性损失
    parser.add_argument("--temporal_loss_weight", type=float, default=5.0,
                        help="L_tc 权重 λ_tc（可调；建议 1~10）")
    parser.add_argument("--gray_color_reg_ratio", type=float, default=1.0,
                        help="灰度训练下的 color 正则权重（作用于 Y-gray3，建议 0.1~2.0）")

    # latent 网络 & 显存（默认 dim128/blocks8 ≈ 1.76M 参数）
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--latent_blocks", type=int, default=8)
    parser.add_argument("--latent_heads", type=int, default=4)
    parser.add_argument("--decode_chunk", type=int, default=1,
                        help="VAE 解码每次处理的帧数（越小峰值显存越低）")
    parser.add_argument("--net_grad_checkpoint", action="store_true",
                        help="对 latent 网络也开梯度检查点（一般不需要，latent 已很轻）")

    # 输出（独立目录，不覆盖现有训练）
    parser.add_argument("--output_dir", type=str, default=r"E:\paper2\codebase\STAR\Ablation1\Fusion_SGFE\train_stage2")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--validation_steps", type=int, default=100)# 1 

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


# ============ 主训练函数 ============

def main(args):
    logging_dir = Path(args.output_dir, "logs")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=logging_dir,
        ),
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(42)

    class _Opt:
        model_path = None
        vae_path = "stabilityai/stable-video-diffusion-img2vid"

    model = VideoControlNetModel(
        _Opt(), device=accelerator.device,
        use_hfp=True, hfp_block_indices=(1, 4, 7),
        hfp_ratio=(0.25, 0.25), hfp_patch=(4, 4), hfp_use_dct=True,
        hfp_serial=False, hfp_init='zeros',
        raft_model_path=args.raft_model_path, latent_scale=4,
        use_spa_local=True, use_temp_local=True,
    )

    if args.resume_checkpoint_dir and os.path.exists(args.resume_checkpoint_dir):
        ctrl_p = os.path.join(args.resume_checkpoint_dir, "model.safetensors")
        unet_p = os.path.join(args.resume_checkpoint_dir, "model_1.safetensors")
        _load_safetensors_into(model.video_controlnet, ctrl_p, "ControlNet")
        _load_safetensors_into(model.unet, unet_p, "UNet")
    else:
        _load_safetensors_into(model.video_controlnet, args.pretrained_model_path, "ControlNet")
        _load_safetensors_into(model.unet, args.unet_checkpoint, "UNet")

    replace_vae_to_4x(model, args.vae_4x_ckpt)
    model.vae.eval()
    model.vae.requires_grad_(False)

    # 整个扩散模型冻结，仅训练 latent 增强网络。
    model.freeze_for_training(train_controlnet=False, train_unet=False, train_hfp_only=False)
    model.eval_mode()

    # 关键：让 VAE 解码器在反传时重算激活省显存（VAE 参数仍冻结）。
    # VAE 仅含 GroupNorm，train/eval 行为一致，设 train() 只为启用检查点分支。
    model.vae.decoder.gradient_checkpointing = True
    model.vae.decoder.train()

    latent_net = LatentRestormer(
        latent_ch=4, dim=args.latent_dim, num_blocks=args.latent_blocks,
        num_heads=args.latent_heads, use_grad_checkpoint=args.net_grad_checkpoint,
    ).to(accelerator.device)

    if accelerator.is_main_process:
        trainable_net = sum(p.numel() for p in latent_net.parameters() if p.requires_grad)
        logger.info(f"可训练参数: latent_net={trainable_net / 1e6:.2f}M")

    noise_scheduler = model.noise_scheduler

    dataset = SlidingRatioVideoDataset(
        args.vi_data_dir, args.ir_data_dir,
        num_frames=args.num_frames, window_stride=args.window_stride,
        vi_per_ir=args.vi_per_ir, video_size=tuple(args.video_size),
        ir_select_mode=args.ir_select_mode, ir_middle_local_idx=args.ir_middle_local_idx,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0,
    )

    trainable_params = [p for p in latent_net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    remaining_steps = args.max_train_steps - args.start_step
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=min(200, remaining_steps // 10),
        num_training_steps=remaining_steps,
    )

    latent_net, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        latent_net, optimizer, dataloader, lr_scheduler,
    )
    model.video_controlnet.half()

    if args.ir_select_mode == "middle":
        default_anchor_idx = (args.num_frames - 1) // 2 if args.ir_middle_local_idx is None \
                             else int(args.ir_middle_local_idx)
    else:
        default_anchor_idx = 0
    default_anchor_idx = int(max(0, min(default_anchor_idx, args.num_frames - 1)))

    if accelerator.is_main_process:
        print("=" * 60)
        print("训练配置: Fusion_SGFE 第二阶段 (latent 前置增强 + 多步扩散 + L_tc)")
        print(f"输出:       {args.output_dir}")
        print(f"训练:       step {args.start_step} -> {args.max_train_steps}, "
              f"lr={args.learning_rate}, bs={args.train_batch_size}, frames={args.num_frames}")
        print(f"视频大小:   {args.video_size[0]}×{args.video_size[1]}")
        print(f"扩散:       多步采样 steps={args.num_inference_steps}, time_shift={args.time_shift}（冻结, no_grad）")
        print("冻结策略:   扩散 + VAE 冻结；仅训练 LatentRestormer")
        print(f"latent网络: dim={args.latent_dim}, blocks={args.latent_blocks}, heads={args.latent_heads}")
        print(f"显存:       VAE解码器梯度检查点=ON, decode_chunk={args.decode_chunk}")
        print(f"Loss:       max={LOSS_MAX_RATIO}, ssim={LOSS_SSIM_RATIO} "
              f"(vis={LOSS_SSIM_VIS_RATIO}, ir={LOSS_SSIM_IR_RATIO}), "
              f"color={args.gray_color_reg_ratio}(y-gray-reg), grad={LOSS_GRAD_RATIO}, "
              f"tc(L_tc)={args.temporal_loss_weight}")
        print("=" * 60)

    global_step = args.start_step
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step, total=args.max_train_steps,
        desc="Steps", disable=not accelerator.is_local_main_process,
    )

    epoch = 0
    while global_step < args.max_train_steps:
        model.eval_mode()
        latent_net.train()
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(latent_net):
                vi = batch['vi_video'].to(accelerator.device, dtype=torch.float32)
                ir = batch['ir_video'].to(accelerator.device, dtype=torch.float32)
                B = vi.shape[0]

                anchor_tensor = batch.get('ir_anchor_idx', None)
                sample_anchor = int(anchor_tensor[0].item()) if anchor_tensor is not None \
                                else default_anchor_idx

                # ── 扩散：冻结，no_grad 多步采样产出 pred_latent ──
                with torch.no_grad():
                    vi_latent = model.vae_encode(vi.permute(0, 2, 1, 3, 4)).float()
                    ir_latent = model.vae_encode(ir.permute(0, 2, 1, 3, 4)).float()

                    vi_frames = vi.permute(0, 2, 1, 3, 4).contiguous() \
                        if model.ir_flow_prop is not None else None
                    # VI 帧间前向光流：既给扩散 warp 复用，也给 L_tc 复用（RAFT 冻结）
                    flows_fwd = None
                    if model.ir_flow_prop is not None and vi_frames is not None:
                        flows_fwd = model.ir_flow_prop.estimate_flow(vi_frames)

                    noise_scheduler.set_timesteps(
                        args.num_inference_steps, time_shift=args.time_shift,
                        device=accelerator.device)
                    x = vi_latent.clone()
                    for t_step in noise_scheduler.timesteps:
                        t_batch = t_step.unsqueeze(0).repeat(B).to(accelerator.device)
                        v_pred = model.forward(
                            x_t=x, t=t_batch, ir_latent=ir_latent,
                            vi_frames=vi_frames, _precomputed_flows=flows_fwd,
                            ir_anchor_idx=sample_anchor,
                        )
                        x = noise_scheduler.step(v_pred, t_step, x)
                    pred_latent = x.to(torch.float32)

                # ── latent 增强(可训练) + VAE 解码(冻结但梯度可穿过) ──
                pred_latent_sharp = latent_net(pred_latent)
                pred_video = model.vae_decode_chunk(pred_latent_sharp, chunk_size=args.decode_chunk)
                pred_video = rearrange(pred_video, '(b t) c h w -> b c t h w', b=B, t=args.num_frames)
                pred_video = torch.clamp((pred_video + 1.0) * 0.5, 0.0, 1.0)
                pred_video_gray = _to_y_gray3_video(pred_video)
                pred_img = rearrange(pred_video_gray, "b c t h w -> (b t) c h w")

                vi_video_gray = _to_y_gray3_video(_to_01(vi))
                ir_video_gray = _to_y_gray3_video(_to_01(ir))
                vi_img = rearrange(vi_video_gray, "b c t h w -> (b t) c h w")
                ir_img = rearrange(ir_video_gray, "b c t h w -> (b t) c h w")

                loss_fuse, loss_ssim, loss_max, loss_color, loss_grad = fusion_unsupervised_loss(
                    vi_img, ir_img, pred_img,
                    max_ratio=LOSS_MAX_RATIO,
                    ssim_vis_ratio=LOSS_SSIM_VIS_RATIO,
                    ssim_ir_ratio=LOSS_SSIM_IR_RATIO,
                    ssim_ratio=LOSS_SSIM_RATIO,
                    color_ratio=args.gray_color_reg_ratio,
                    grad_ratio=LOSS_GRAD_RATIO,
                )

                # ── 时序一致性损失 L_tc（复用 VI 前向光流，作用在融合视频上） ──
                loss_tc = temporal_consistency_loss(pred_video_gray, flows_fwd)
                loss = loss_fuse + args.temporal_loss_weight * loss_tc

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                accelerator.log({
                    "loss": loss.item(),
                    "loss_ssim": loss_ssim.item(),
                    "loss_max": loss_max.item(),
                    "loss_color": loss_color.item(),
                    "loss_grad": loss_grad.item(),
                    "loss_tc": loss_tc.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ssim=f"{loss_ssim.item():.4f}",
                    max=f"{loss_max.item():.4f}",
                    color=f"{loss_color.item():.4f}",
                    grad=f"{loss_grad.item():.4f}",
                    tc=f"{loss_tc.item():.4f}",
                )

                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    try:
                        save_validation_video(
                            vi.detach(), ir.detach(), pred_video_gray.detach(),
                            os.path.join(args.output_dir, "validations"), global_step,
                        )
                    except Exception:
                        import traceback
                        traceback.print_exc()

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved: {save_path}")

            if global_step >= args.max_train_steps:
                break
        epoch += 1

    if accelerator.is_main_process:
        final = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        if not os.path.exists(final):
            accelerator.save_state(final)
            logger.info(f"Final checkpoint saved: {final}")

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())

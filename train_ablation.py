#!/usr/bin/env python
# coding=utf-8
"""
训练脚本 — Fusion_SGFE（block3 版 SGFE）

当前设置:
  - 使用 SGFE（use_hfp=True），SGFE 初始化为 zeros（block3 全0）
  - 打开空间/时间 local（use_spa_local=True, use_temp_local=True）
  - 开启 IR 光流传播（raft_model_path）
  - 训练阶段冻结 ControlNet + UNet 基座，仅训练 SGFE 分支
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from Fusion_SGFE.dual_modal_fusion_model import VideoControlNetModel
from Fusion_SGFE.train_latent_vae48_tem import (
    replace_vae_to_4x,
    run_validation,
    save_validation_results,
)

logger = get_logger(__name__)


# ============ 工具函数 ============

def _load_image_as_gray3(path, video_size):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"读取图像失败: {path}")
    img = cv2.resize(img, (video_size[1], video_size[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    gray3 = gray3 * 2 - 1
    return gray3


def _select_ir_index_for_vi_window(
    vi_window_files, ir_files, vi_start_idx, vi_per_ir,
    mode="middle", middle_local_idx=None,
):
    n = len(vi_window_files) if vi_window_files else 1
    if middle_local_idx is None:
        mid_local = (n - 1) // 2
    else:
        mid_local = max(0, min(int(middle_local_idx), n - 1))

    if len(ir_files) == 0:
        return 0, mid_local if mode == "middle" else 0

    ir_stem_to_idx = {p.stem: i for i, p in enumerate(ir_files)}
    matched = [
        (local_idx, ir_stem_to_idx[f.stem])
        for local_idx, f in enumerate(vi_window_files)
        if f.stem in ir_stem_to_idx
    ]

    if mode == "middle":
        if matched:
            local_idx, ir_idx = min(matched, key=lambda x: abs(x[0] - mid_local))
            return ir_idx, local_idx
        mid_global = vi_start_idx + mid_local
        ir_idx = min(mid_global // max(1, int(vi_per_ir)), len(ir_files) - 1)
        return ir_idx, mid_local

    if mode == "random":
        if matched:
            local_idx, ir_idx = matched[np.random.randint(0, len(matched))]
            return ir_idx, local_idx
        ir_idx = min(vi_start_idx // max(1, int(vi_per_ir)), len(ir_files) - 1)
        return ir_idx, 0

    # mode == "start":
    # 优先与窗口第 1 帧做同名匹配，避免仅按 vi_per_ir 比例索引导致错配。
    if vi_window_files:
        start_stem = vi_window_files[0].stem
        if start_stem in ir_stem_to_idx:
            return ir_stem_to_idx[start_stem], 0
    if matched:
        # 找不到第一帧同名时，回退到窗口内最靠前的可匹配帧
        local_idx, ir_idx = min(matched, key=lambda x: x[0])
        return ir_idx, local_idx

    ir_idx = min(vi_start_idx // max(1, int(vi_per_ir)), len(ir_files) - 1)
    return ir_idx, 0


class SlidingRatioVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, vi_dir, ir_dir, gt_dir,
        num_frames=8, window_stride=2, vi_per_ir=8,
        video_size=(256, 256), ir_select_mode="start", ir_middle_local_idx=None,
    ):
        self.vi_dir = Path(vi_dir)
        self.ir_dir = Path(ir_dir)
        self.gt_dir = Path(gt_dir)
        self.num_frames = int(num_frames)
        self.window_stride = int(max(1, window_stride))
        self.vi_per_ir = int(max(1, vi_per_ir))
        self.video_size = tuple(video_size)
        self.ir_select_mode = str(ir_select_mode).lower()
        self.ir_middle_local_idx = (
            None if ir_middle_local_idx is None else int(ir_middle_local_idx)
        )
        self.samples = self._collect_samples()
        print(
            f"SlidingRatioVideoDataset: {len(self.samples)} clips, "
            f"ir_select_mode={self.ir_select_mode}, "
            f"video_size={self.video_size}"
        )

    def _collect_samples(self):
        samples = []
        if not (self.vi_dir.exists() and self.ir_dir.exists() and self.gt_dir.exists()):
            return samples

        for gt_sub in sorted(self.gt_dir.iterdir()):
            if not gt_sub.is_dir():
                continue
            vi_sub = self.vi_dir / gt_sub.name
            ir_sub = self.ir_dir / gt_sub.name
            if not (vi_sub.exists() and ir_sub.exists()):
                continue

            vi_files = sorted(list(vi_sub.glob("*.png")) + list(vi_sub.glob("*.jpg")))
            gt_files = sorted(list(gt_sub.glob("*.png")) + list(gt_sub.glob("*.jpg")))
            ir_files = sorted(list(ir_sub.glob("*.png")) + list(ir_sub.glob("*.jpg")))
            if len(vi_files) < self.num_frames or len(gt_files) < self.num_frames or len(ir_files) == 0:
                continue

            vi_map = {f.stem: f for f in vi_files}
            gt_map = {f.stem: f for f in gt_files}
            common = sorted(set(vi_map) & set(gt_map))
            if len(common) >= self.num_frames:
                vi_seq = [vi_map[s] for s in common]
                gt_seq = [gt_map[s] for s in common]
            else:
                n = min(len(vi_files), len(gt_files))
                if n < self.num_frames:
                    continue
                vi_seq = vi_files[:n]
                gt_seq = gt_files[:n]

            if self.window_stride >= self.num_frames:
                starts = list(range(0, len(vi_seq) - self.num_frames + 1, self.num_frames))
            else:
                starts = list(range(0, len(vi_seq) - self.num_frames + 1, self.window_stride))
                if starts and starts[-1] != len(vi_seq) - self.num_frames:
                    starts.append(len(vi_seq) - self.num_frames)
            if not starts:
                continue

            samples.append({
                "vi_seq": vi_seq,
                "gt_seq": gt_seq,
                "ir_seq": ir_files,
                "starts": starts,
            })
        return samples

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx):
        dummy_anchor = (self.num_frames - 1) // 2 if self.ir_select_mode == "middle" else 0
        dummy = {
            "vi_video": torch.randn(3, self.num_frames, *self.video_size),
            "ir_video": torch.randn(3, self.num_frames, *self.video_size),
            "gt_video": torch.randn(3, self.num_frames, *self.video_size),
            "ir_anchor_idx": torch.tensor(int(dummy_anchor), dtype=torch.long),
        }
        if not self.samples:
            return dummy

        sample = self.samples[idx % len(self.samples)]
        start = sample["starts"][np.random.randint(0, len(sample["starts"]))]
        indices = np.arange(start, start + self.num_frames)

        vi_frames = [_load_image_as_gray3(sample["vi_seq"][i], self.video_size) for i in indices]
        gt_frames = [_load_image_as_gray3(sample["gt_seq"][i], self.video_size) for i in indices]
        vi_video = torch.from_numpy(np.transpose(np.stack(vi_frames), (3, 0, 1, 2))).float()
        gt_video = torch.from_numpy(np.transpose(np.stack(gt_frames), (3, 0, 1, 2))).float()

        vi_window = [sample["vi_seq"][i] for i in indices]
        ir_idx, ir_local_idx = _select_ir_index_for_vi_window(
            vi_window_files=vi_window,
            ir_files=sample["ir_seq"],
            vi_start_idx=start,
            vi_per_ir=self.vi_per_ir,
            mode=self.ir_select_mode,
            middle_local_idx=self.ir_middle_local_idx,
        )
        ir_frame = _load_image_as_gray3(sample["ir_seq"][ir_idx], self.video_size)
        ir_frame = torch.from_numpy(np.transpose(ir_frame, (2, 0, 1))).float()
        ir_video = ir_frame.unsqueeze(1).repeat(1, self.num_frames, 1, 1)

        return {
            "vi_video": vi_video,
            "ir_video": ir_video,
            "gt_video": gt_video,
            "ir_anchor_idx": torch.tensor(int(ir_local_idx), dtype=torch.long),
        }


def _load_safetensors_into(module, path, name):
    if not path or not os.path.exists(path):
        if path:
            logger.warning(f"{name} checkpoint 不存在，跳过: {path}")
        return

    from safetensors.torch import load_file
    raw_state = load_file(path)
    module_state = module.state_dict()
    state = {}
    shape_mismatch = 0
    for k, v in raw_state.items():
        key = k.replace('module.', '')
        # 兼容旧命名
        key = key.replace('.hfp_2d.', '.sgfe_2d.')
        if key not in module_state:
            continue
        if module_state[key].shape != v.shape:
            shape_mismatch += 1
            continue
        state[key] = v

    ret = module.load_state_dict(state, strict=False)
    logger.info(
        f"{name} 加载完成: loaded={len(state)}, missing={len(ret.missing_keys)}, "
        f"unexpected={len(ret.unexpected_keys)}, shape_mismatch_skipped={shape_mismatch}"
    )


def _make_gaussian_kernel(size=5, sigma=1.0):
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = g.outer(g)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def _low_pass_5d(x, kernel):
    b, c, t, h, w = x.shape
    x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    k = kernel.to(device=x.device, dtype=x.dtype).expand(c, -1, -1, -1)
    pad = kernel.shape[-1] // 2
    x_lf = F.conv2d(x_flat, k, padding=pad, groups=c)
    return x_lf.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def _pointwise_loss(pred, target, loss_type="l1"):
    lt = str(loss_type).lower()
    if lt in ("l1", "mae"):
        return F.l1_loss(pred, target)
    if lt in ("l2", "mse"):
        return F.mse_loss(pred, target)
    raise ValueError(f"不支持的 loss_type: {loss_type}")


# ============ 参数 ============

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Fusion_SGFE 训练（SGFE63 + spa/temp local）"
    )

    # 权重路径
    parser.add_argument(
        "--pretrained_model_path", type=str,
        default=r"E:\paper2\codebase\STAR\outputs\33_vae4x\checkpoint-10000\model.safetensors",
    )
    parser.add_argument(
        "--unet_checkpoint", type=str,
        default=r"E:\paper2\codebase\STAR\outputs\temporal8f_step_20260515_172242\checkpoint-1500\model_1.safetensors",
    )
    parser.add_argument(
        "--resume_checkpoint_dir", type=str, default=None,
    )
    parser.add_argument(
        "--vae_4x_ckpt", type=str,
        default=r"pretrained_weight/vae_4x_tvt.ckpt",
    )
    parser.add_argument(
        "--raft_model_path", type=str,
        default=r"Fusion_SGFE/ProPainter/weights/raft-things.pth",
    )

    # 数据
    parser.add_argument(
        "--vi_data_dir", type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\visible2",
    )
    parser.add_argument(
        "--ir_data_dir", type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\infrared",
    )
    parser.add_argument(
        "--gt_data_dir", type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\Fusion",
    )

    # 训练
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--window_stride", type=int, default=2)
    parser.add_argument("--vi_per_ir", type=int, default=8)
    parser.add_argument("--ir_select_mode", type=str, default="start",
                        choices=["middle", "random", "start"])
    parser.add_argument("--ir_middle_local_idx", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    # 默认仅开启 velocity loss：其余 loss 权重默认为 0。
    parser.add_argument("--anchor_rec_weight", type=float, default=0.0)
    parser.add_argument("--anchor_dt_weight", type=float, default=0.0)
    parser.add_argument("--temporal_loss_weight", type=float, default=0.0)
    parser.add_argument("--temporal_dt", type=float, default=0.05)
    parser.add_argument("--velocity_loss_weight", type=float, default=1.0)
    parser.add_argument("--velocity_loss_type", type=str, default="mse",
                        choices=["mse", "l1"])
    parser.add_argument("--anchor_rec_loss_type", type=str, default="l1",
                        choices=["l1", "mse"])
    parser.add_argument("--anchor_dt_loss_type", type=str, default="l1",
                        choices=["l1", "mse"])
    parser.add_argument("--temporal_loss_type", type=str, default="l1",
                        choices=["l1", "mse"])

    # 输出
    parser.add_argument(
        "--output_dir", type=str,
        default=r"Ablation1/Fusion_SGFE/full_loss",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=1500)
    parser.add_argument("--validation_steps", type=int, default=500)

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
            project_dir=args.output_dir,
            logging_dir=logging_dir,
        ),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(42)

    class _Opt:
        model_path = None
        vae_path = "stabilityai/stable-video-diffusion-img2vid"

    # 启用 SGFE(block3) + 打开 spa/temp local + 开启光流。
    model = VideoControlNetModel(
        _Opt(),
        device=accelerator.device,
        use_hfp=True,
        hfp_block_indices=(1, 4, 7),
        hfp_ratio=(0.25, 0.25),
        hfp_patch=(4, 4),
        hfp_use_dct=True,
        hfp_serial=False,
        hfp_init='zeros',
        raft_model_path=args.raft_model_path,
        latent_scale=4,
        use_spa_local=True,
        use_temp_local=True,
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

    # 按你的要求：关闭 ControlNet + UNet 基座，仅训练 SGFE 分支。
    model.freeze_for_training(train_controlnet=False, train_unet=False, train_hfp_only=True)

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.get_trainable_parameters())
        logger.info(f"可训练参数: {trainable / 1e6:.2f}M (仅 SGFE 分支)")

    noise_scheduler = model.noise_scheduler
    use_anchor = (args.anchor_rec_weight > 0) or (args.anchor_dt_weight > 0)
    gauss_kernel = _make_gaussian_kernel(size=5, sigma=1.0).to(accelerator.device)

    dataset = SlidingRatioVideoDataset(
        args.vi_data_dir, args.ir_data_dir, args.gt_data_dir,
        num_frames=args.num_frames,
        window_stride=args.window_stride,
        vi_per_ir=args.vi_per_ir,
        video_size=tuple(args.video_size),
        ir_select_mode=args.ir_select_mode,
        ir_middle_local_idx=args.ir_middle_local_idx,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0,
    )

    trainable_params = list(model.get_trainable_parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    remaining_steps = args.max_train_steps - args.start_step
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=min(200, remaining_steps // 10),
        num_training_steps=remaining_steps,
    )

    model.unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model.unet, optimizer, dataloader, lr_scheduler,
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
        print("训练配置: Fusion_SGFE")
        print("  使用: SGFE(block3) (hfp_init=zeros, 全0初始化)")
        print("  开关: spa_local=True, temp_local=True")
        print(f"  光流: raft_model_path={args.raft_model_path}")
        print(f"ControlNet: {args.pretrained_model_path}")
        print(f"UNet:       {args.unet_checkpoint}")
        print(f"输出:       {args.output_dir}")
        print(f"训练:       step {args.start_step} -> {args.max_train_steps}, "
              f"lr={args.learning_rate}, bs={args.train_batch_size}, frames={args.num_frames}")
        print(f"视频大小:   {args.video_size[0]}×{args.video_size[1]}，VI:IR = {args.vi_per_ir}:1")
        print("冻结策略:   ControlNet 冻结，UNet 基座冻结，仅 SGFE 可训练")
        print("Loss配置:")
        print(f"  velocity: weight={args.velocity_loss_weight}, type={args.velocity_loss_type}")
        print(f"  temporal: weight={args.temporal_loss_weight}, type={args.temporal_loss_type}")
        print(f"  anchor_rec: weight={args.anchor_rec_weight}, type={args.anchor_rec_loss_type}")
        print(f"  anchor_dt:  weight={args.anchor_dt_weight}, type={args.anchor_dt_loss_type}")
        print("=" * 60)

    global_step = args.start_step
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step, total=args.max_train_steps,
        desc="Steps", disable=not accelerator.is_local_main_process,
    )

    epoch = 0
    while global_step < args.max_train_steps:
        model.train_mode()
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(model.unet):
                vi = batch['vi_video'].to(accelerator.device, dtype=torch.float32)
                ir = batch['ir_video'].to(accelerator.device, dtype=torch.float32)
                gt = batch['gt_video'].to(accelerator.device, dtype=torch.float32)
                B = vi.shape[0]

                anchor_tensor = batch.get('ir_anchor_idx', None)
                sample_anchor = int(anchor_tensor[0].item()) if anchor_tensor is not None \
                                else default_anchor_idx

                with torch.no_grad():
                    vi_latent = model.vae_encode(vi.permute(0, 2, 1, 3, 4)).float()
                    ir_latent = model.vae_encode(ir.permute(0, 2, 1, 3, 4)).float()
                    gt_latent = model.vae_encode(gt.permute(0, 2, 1, 3, 4)).float()

                velocity = gt_latent - vi_latent
                t = torch.rand(B, device=accelerator.device)
                t_expand = t.view(B, 1, 1, 1, 1)
                x_t = (1 - t_expand) * vi_latent + t_expand * gt_latent

                # 开启光流 warp：传入 VI 像素帧用于估计帧间光流。
                vi_frames = vi.permute(0, 2, 1, 3, 4).contiguous() \
                    if model.ir_flow_prop is not None else None

                with accelerator.autocast():
                    pred_velocity = model.forward(
                        x_t=x_t, t=t, ir_latent=ir_latent,
                        vi_frames=vi_frames,
                        ir_anchor_idx=sample_anchor,
                    )
                pred_velocity = pred_velocity.float()

                if args.velocity_loss_weight > 0:
                    loss_vel = _pointwise_loss(
                        pred_velocity, velocity, args.velocity_loss_type
                    )
                else:
                    loss_vel = pred_velocity.sum() * 0.0
                pred_gt_latent = vi_latent + pred_velocity
                if args.temporal_loss_weight > 0 and args.num_frames > 1:
                    pred_dt_v = pred_velocity[:, :, 1:] - pred_velocity[:, :, :-1]
                    gt_dt_v = velocity[:, :, 1:] - velocity[:, :, :-1]
                    loss_tc = _pointwise_loss(
                        pred_dt_v, gt_dt_v.detach(), args.temporal_loss_type
                    )
                else:
                    loss_tc = pred_velocity.sum() * 0.0

                if use_anchor:
                    if args.anchor_rec_weight > 0:
                        z_hat_lf = _low_pass_5d(pred_gt_latent, gauss_kernel)
                        z_gt_lf = _low_pass_5d(gt_latent, gauss_kernel)
                        loss_rec = _pointwise_loss(
                            z_hat_lf, z_gt_lf.detach(), args.anchor_rec_loss_type
                        )
                    else:
                        loss_rec = pred_velocity.sum() * 0.0

                    if args.anchor_dt_weight > 0 and args.num_frames > 1:
                        pred_dt = pred_gt_latent[:, :, 1:] - pred_gt_latent[:, :, :-1]
                        gt_dt = gt_latent[:, :, 1:] - gt_latent[:, :, :-1]
                        loss_dt = _pointwise_loss(
                            pred_dt, gt_dt.detach(), args.anchor_dt_loss_type
                        )
                    else:
                        loss_dt = pred_velocity.sum() * 0.0
                else:
                    loss_rec = torch.tensor(0.0, device=accelerator.device)
                    loss_dt = torch.tensor(0.0, device=accelerator.device)

                loss = (
                    loss_vel
                    + args.temporal_loss_weight * loss_tc
                    + args.anchor_rec_weight * loss_rec
                    + args.anchor_dt_weight * loss_dt
                )

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
                    "loss_vel": loss_vel.item(),
                    "loss_temporal": loss_tc.item(),
                    "loss_rec_anchor": loss_rec.item(),
                    "loss_dt_anchor": loss_dt.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    vel=f"{loss_vel.item():.4f}",
                    tc=f"{loss_tc.item():.4f}",
                    rec=f"{loss_rec.item():.4f}",
                    dt=f"{loss_dt.item():.4f}",
                )

                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    try:
                        pred = run_validation(
                            model, noise_scheduler, vi.detach(), ir.detach(),
                            accelerator.device, ir_anchor_idx=sample_anchor,
                        )
                        save_validation_results(
                            vi.detach(), ir.detach(), gt.detach(), pred,
                            os.path.join(args.output_dir, "validations"),
                            global_step,
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

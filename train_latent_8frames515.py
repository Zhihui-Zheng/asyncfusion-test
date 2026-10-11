#!/usr/bin/env python
# coding=utf-8
"""
8-frame Flow Matching 时序一致性微调版 - step-consistent L_tc

数据设置:
  - VI/GT: 连续 8 帧
  - IR: 取窗口起始帧 1 张，复制到 8 帧作为 ControlNet hint
  - ControlNet 默认冻结，只训练 UNet/SGFE，节省算力并保留静态 IR 条件能力

时序正则:
  pred_next = x_t + dt * pred_velocity
  gt_next   = x_t + dt * (gt_latent - vi_latent)
  对 pred_next / gt_next 做低频帧间差一致性 L_tc
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

from Fusion5.dual_modal_fusion_model import VideoControlNetModel
from Fusion5.temporal_consistency_loss import LatentTemporalConsistencyLoss
from Fusion5.train_latent_vae48_tem import (
    replace_vae_to_4x,
    run_validation,
    save_validation_results,
)

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="8-frame Flow Matching 时序微调 - step-consistent temporal loss"
    )

    # 权重路径
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=r"E:\paper2\codebase\STAR\outputs\latent_20260408_103729\checkpoint-2000\model.safetensors",
    )
    parser.add_argument(
        "--unet_checkpoint",
        type=str,
        default=r"E:\paper2\codebase\STAR\outputs\temporal8f_step_20260508_121620\checkpoint-3000\model_1.safetensors",
    )
    parser.add_argument(
        "--resume_checkpoint_dir",
        type=str,
        default=None,
        help="从 accelerator checkpoint 目录恢复，优先级最高",
    )
    parser.add_argument(
        "--raft_model_path",
        type=str,
        default=r"Fusion5/ProPainter/weights/raft-things.pth",
    )
    parser.add_argument(
        "--vae_4x_ckpt",
        type=str,
        default=r"pretrained_weight/vae_4x_tvt.ckpt",
    )

    # 数据
    parser.add_argument(
        "--vi_data_dir",
        type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\visible2",
    )
    parser.add_argument(
        "--ir_data_dir",
        type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\infrared",
    )
    parser.add_argument(
        "--gt_data_dir",
        type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\Fusion",
    )

    # 训练  
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8, help="VI/GT 连续帧数；IR 起始帧会复制到该长度")
    parser.add_argument("--window_stride", type=int, default=8, help="训练窗口步长；默认8表示非重叠分组，<8为滑动")
    parser.add_argument("--vi_per_ir", type=int, default=8, help="VI:IR 帧率比（40:320 对应 8）")
    parser.add_argument(
        "--ir_select_mode",
        type=str,
        default="middle",
        choices=["middle", "random", "start"],
        help=(
            "IR 选取策略：middle=取窗口中心(默认第4帧, 1-indexed)对齐的IR；"
            "random=旧行为(窗口内同名IR随机)；start=对齐窗口起始帧"
        ),
    )
    parser.add_argument(
        "--ir_middle_local_idx",
        type=int,
        default=None,
        help="自定义中心位置 (0-indexed)，仅 ir_select_mode=middle 时生效；None 表示 (num_frames-1)//2",
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # 时序损失
    parser.add_argument(
        "--temporal_loss_weight",
        type=float,
        default=0.2,
        help="step-consistent L_tc 权重，建议先用 0.03~0.06",
    )
    parser.add_argument(
        "--temporal_dt",
        type=float,
        default=0.05,
        help="时序 loss 的一步步长；默认 0.05 对应约 20-step Euler 采样",
    )

    # 输出
    from datetime import datetime as _dt
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"outputs/temporal8f_step_{_dt.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=1500)
    parser.add_argument("--validation_steps", type=int, default=250)

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


def _load_image_as_gray3(path, video_size):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"读取图像失败: {path}")
    img = cv2.resize(img, (video_size[1], video_size[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    gray3 = gray3 * 2 - 1
    return gray3


def _parse_trailing_int(stem):
    m = re.search(r"(\d+)$", stem)
    if m is None:
        return None
    return int(m.group(1))


def _select_ir_index_for_vi_window(
    vi_window_files,
    ir_files,
    vi_start_idx,
    vi_per_ir,
    mode="middle",
    middle_local_idx=None,
):
    """
    IR 帧选取策略 (mode)：
      - "middle"  : 取窗口中间帧 (默认 (num_frames-1)//2，即 8 帧时的第 4 帧, 1-indexed)
                    对应的 IR；若窗口内没有同名 IR，则在窗口内取最接近中心的同名匹配，
                    再不行就用 (vi_start + middle) // vi_per_ir 兜底。
      - "random"  : 旧行为，在窗口内同名 IR 中随机选一个。
      - "start"   : 始终用窗口起始帧对齐的 IR。

    Returns:
        (ir_idx, ir_local_idx)
            ir_idx       : IR 文件列表中的全局索引
            ir_local_idx : 该 IR 在 VI 窗口内的实际 0-indexed 位置 (warp 锚点用)
                           - 同名匹配时 = 匹配 VI 帧的 local_idx (精确)
                           - 兜底比例映射时 = mid_local / 0 (近似)
    """
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

    # "start" 或未识别
    ir_idx = min(vi_start_idx // max(1, int(vi_per_ir)), len(ir_files) - 1)
    return ir_idx, 0


class SlidingRatioVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vi_dir,
        ir_dir,
        gt_dir,
        num_frames=8,
        window_stride=2,
        vi_per_ir=8,
        video_size=(256, 256),
        ir_select_mode="middle",
        ir_middle_local_idx=None,
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
            f"ir_middle_local_idx={self.ir_middle_local_idx}"
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
                # 非重叠分组：0-7, 8-15, ...
                starts = list(range(0, len(vi_seq) - self.num_frames + 1, self.num_frames))
            else:
                # 滑动窗口：例如 stride=2
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
    state = {}
    for k, v in load_file(path).items():
        key = k.replace('module.', '')
        # 兼容旧 HFP checkpoint：旧模块的 2D 高频分支名为 hfp_2d，
        # SGFE 中对应的可复用高频分支改名为 sgfe_2d。
        # 新增的低频 structure_gate / alpha 保持初始化，从零学习。
        key = key.replace('.hfp_2d.', '.sgfe_2d.')
        state[key] = v
    ret = module.load_state_dict(state, strict=False)
    logger.info(
        f"{name} 加载完成: missing={len(ret.missing_keys)}, "
        f"unexpected={len(ret.unexpected_keys)}"
    )


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

    # 8 帧训练阶段只训练 UNet/SGFE，ControlNet/VAE/RAFT 冻结。
    model.freeze_for_training(train_controlnet=False, train_unet=True)

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.get_trainable_parameters())
        logger.info(f"可训练参数: {trainable / 1e6:.2f}M (仅 UNet/SGFE)")
        logger.info(
            f"8-frame step-consistent L_tc: frames={args.num_frames}, "
            f"stride={args.window_stride}, vi_per_ir={args.vi_per_ir}, "
            f"weight={args.temporal_loss_weight}, dt={args.temporal_dt}"
        )

    noise_scheduler = model.noise_scheduler
    temporal_loss_fn = None
    use_temporal = args.temporal_loss_weight > 0
    if use_temporal:
        temporal_loss_fn = LatentTemporalConsistencyLoss().to(accelerator.device)

    dataset = SlidingRatioVideoDataset(
        args.vi_data_dir,
        args.ir_data_dir,
        args.gt_data_dir,
        num_frames=args.num_frames,
        window_stride=args.window_stride,
        vi_per_ir=args.vi_per_ir,
        ir_select_mode=args.ir_select_mode,
        ir_middle_local_idx=args.ir_middle_local_idx,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    trainable_params = list(model.get_trainable_parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    remaining_steps = args.max_train_steps - args.start_step
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(500, remaining_steps // 10),
        num_training_steps=remaining_steps,
    )

    model.unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model.unet,
        optimizer,
        dataloader,
        lr_scheduler,
    )
    model.video_controlnet.half()

    # 每个 batch 里 ir_anchor_idx 来自 dataset (per-sample), 这里只算一个用于打印 / 验证.
    if args.ir_select_mode == "middle":
        default_anchor_idx = (args.num_frames - 1) // 2 if args.ir_middle_local_idx is None \
                             else int(args.ir_middle_local_idx)
    else:
        default_anchor_idx = 0
    default_anchor_idx = int(max(0, min(default_anchor_idx, args.num_frames - 1)))

    if accelerator.is_main_process:
        print("=" * 60)
        print("8-frame Flow Matching 时序微调: step-consistent L_tc")
        print(f"ControlNet: {args.pretrained_model_path}")
        print(f"UNet:       {args.unet_checkpoint}")
        print(f"输出:       {args.output_dir}")
        print(
            f"训练:       step {args.start_step} -> {args.max_train_steps}, "
            f"lr={args.learning_rate}, bs={args.train_batch_size}, frames={args.num_frames}"
        )
        print("冻结策略:   ControlNet 冻结，仅训练 UNet/SGFE")
        print(f"时序损失:   λ={args.temporal_loss_weight}, dt={args.temporal_dt}")
        if args.window_stride >= args.num_frames:
            print(f"数据组织:   非重叠分组 ({args.num_frames}帧/组), IR 对齐比 VI:IR={args.vi_per_ir}:1")
        else:
            print(f"数据组织:   滑窗 stride={args.window_stride}, IR 对齐比 VI:IR={args.vi_per_ir}:1")
        if args.ir_select_mode == "middle":
            mid_disp = (args.num_frames - 1) // 2 if args.ir_middle_local_idx is None \
                       else args.ir_middle_local_idx
            print(f"IR 选取:    mode=middle, 窗口内 0-indexed={mid_disp} (即第{mid_disp + 1}帧, 1-indexed)")
        else:
            print(f"IR 选取:    mode={args.ir_select_mode}")
        print(f"Warp 锚点:  per-sample ir_anchor_idx (来自 dataset, 双向光流)")
        print(f"默认锚点:   {default_anchor_idx} (0-indexed, 仅用于验证 fallback / 打印)")
        print("=" * 60)

    global_step = args.start_step
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
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(model.unet):
                vi = batch['vi_video'].to(accelerator.device, dtype=torch.float32)
                ir = batch['ir_video'].to(accelerator.device, dtype=torch.float32)
                gt = batch['gt_video'].to(accelerator.device, dtype=torch.float32)
                B = vi.shape[0]

                # 取本 batch 的 anchor (bs=1 默认; bs>1 时取第 0 个, 同一 batch 应来自相近样本)
                anchor_tensor = batch.get('ir_anchor_idx', None)
                if anchor_tensor is None:
                    sample_anchor = default_anchor_idx
                else:
                    sample_anchor = int(anchor_tensor[0].item())

                with torch.no_grad():
                    vi_latent = model.vae_encode(vi.permute(0, 2, 1, 3, 4)).float()
                    ir_latent = model.vae_encode(ir.permute(0, 2, 1, 3, 4)).float()
                    gt_latent = model.vae_encode(gt.permute(0, 2, 1, 3, 4)).float()

                velocity = gt_latent - vi_latent
                t = torch.rand(B, device=accelerator.device)
                t_expand = t.view(B, 1, 1, 1, 1)
                x_t = (1 - t_expand) * vi_latent + t_expand * gt_latent

                vi_frames = vi.permute(0, 2, 1, 3, 4) if model.ir_flow_prop is not None else None
                with accelerator.autocast():
                    pred_velocity = model.forward(
                        x_t=x_t,
                        t=t,
                        ir_latent=ir_latent,
                        vi_frames=vi_frames,
                        ir_anchor_idx=sample_anchor,
                    )
                pred_velocity = pred_velocity.float()

                loss_vel = F.mse_loss(pred_velocity, velocity)

                if use_temporal:
                    dt = args.temporal_dt
                    pred_next = x_t + dt * pred_velocity
                    gt_next = x_t.detach() + dt * velocity
                    loss_tc = temporal_loss_fn(pred_next, gt_next)
                else:
                    loss_tc = torch.tensor(0.0, device=accelerator.device)

                loss = loss_vel + args.temporal_loss_weight * loss_tc

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
                    "loss_tc_step": loss_tc.item(),
                    "temporal_dt": args.temporal_dt,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    vel=f"{loss_vel.item():.4f}",
                    tc=f"{loss_tc.item():.4f}",
                )

                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    try:
                        pred = run_validation(
                            model,
                            noise_scheduler,
                            vi.detach(),
                            ir.detach(),
                            accelerator.device,
                            ir_anchor_idx=sample_anchor,
                        )
                        save_validation_results(
                            vi.detach(),
                            ir.detach(),
                            gt.detach(),
                            pred,
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
            logger.info(f"Final saved: {final}")

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())

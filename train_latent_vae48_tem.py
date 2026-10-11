#!/usr/bin/env python
# coding=utf-8
"""
Flow Matching 训练脚本 — 时序一致性微调版 (含 HFP)

在已训好的融合模型 (UNet + HFP) 上，加时序损失 L_tc 恢复帧间一致性。
ControlNet 冻结，只训 UNet (含 HFP 模块)。

损失: MSE + λ_tc * L_tc
  - MSE: Flow Matching velocity 主损失 (权重不变)
  - L_tc: 低频帧间差一致性 + 空间梯度一致性 (额外正则)

用法:
  python Fusion5/train_latent_vae48_tem.py --temporal_loss_weight 0.06
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from einops import rearrange

import sys
import types
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from Fusion5.dual_modal_fusion_model import VideoControlNetModel
from Fusion5.temporal_consistency_loss import LatentTemporalConsistencyLoss

import cv2
import numpy as np
import matplotlib.pyplot as plt

# TVT VAED4 (预训练的 4x VAE)
tvt_path = os.path.join(base_path, 'TVT', 'TVT-main', 'TVT')
sys.path.insert(0, tvt_path)
from VAED4.autoencoder_kl import AutoencoderKL as VAED4AutoencoderKL

logger = get_logger(__name__)


# ============ TVT VAED4 加载 ============

def load_vaed4(vae_ckpt_path):
    config_path = os.path.join(tvt_path, 'VAED4', 'config2.json')
    vae = VAED4AutoencoderKL.from_config(
        VAED4AutoencoderKL.load_config(config_path)
    )
    vae_ckp = torch.load(vae_ckpt_path, map_location='cpu', weights_only=False)
    ckp_dict = {n.replace('vaeDe4.', ''): vae_ckp['state_dict'][n]
                for n in vae_ckp['state_dict'] if "vaeDe4" in n}
    vae.load_state_dict(ckp_dict, strict=True)
    print(f"TVT VAED4 加载完成: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M params")
    del vae_ckp
    torch.cuda.empty_cache()
    return vae


def replace_vae_to_4x(model, vae_ckpt_path):
    vae_4x = load_vaed4(vae_ckpt_path)
    vae_4x.eval()
    vae_4x.requires_grad_(False)
    vae_4x.to(model.device, dtype=torch.float32)
    del model.vae
    torch.cuda.empty_cache()
    model.vae = vae_4x

    def vae_encode_4x(self, video, chunk_size=1):
        num_f = video.shape[1]
        video = rearrange(video, "b f c h w -> (b f) c h w").float()
        z_list = []
        for i in range(0, video.shape[0], chunk_size):
            z_list.append(self.vae.encode(video[i:i+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        return rearrange(z, "(b f) c h w -> b c f h w", f=num_f) * self.vae.config.scaling_factor

    def vae_decode_4x(self, z, num_f):
        return self.vae.decode(z.float() / self.vae.config.scaling_factor).sample

    def vae_decode_chunk_4x(self, z, chunk_size=3):
        B = z.shape[0]
        z = rearrange(z, "b c f h w -> (b f) c h w").float()
        video = []
        for i in range(0, z.shape[0], chunk_size):
            chunk = z[i:i+chunk_size]
            video.append(self.vae_decode(chunk, chunk.shape[0]))
        return torch.cat(video)

    model.vae_encode = types.MethodType(vae_encode_4x, model)
    model.vae_decode = types.MethodType(vae_decode_4x, model)
    model.vae_decode_chunk = types.MethodType(vae_decode_chunk_4x, model)


# ============ 图片转灰度3通道 ============

def bgr_to_gray_3ch(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return np.stack([gray, gray, gray], axis=-1)


# ============ 参数 ============
 
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Flow Matching 时序微调")

    # ── 权重路径 ──
    parser.add_argument("--pretrained_model_path", type=str,
                        default=r"E:\paper2\codebase\STAR\outputs\latent_20260408_103729\checkpoint-2000\model.safetensors")
    parser.add_argument("--unet_checkpoint", type=str,
                        default=r"E:\paper2\codebase\STAR\outputs\hfp_20260410_152654\checkpoint-3000\model1.safetensors")
    parser.add_argument("--resume_checkpoint_dir", type=str, default=None,
                        help="从 accelerator checkpoint 目录恢复 (优先级最高)")
    parser.add_argument("--raft_model_path", type=str,
                        default=r"Fusion5/ProPainter/weights/raft-things.pth")

    # ── 数据 ──
    parser.add_argument("--vi_data_dir", type=str,
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\visible2")
    parser.add_argument("--ir_data_dir", type=str,
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\infrared")
    parser.add_argument("--gt_data_dir", type=str,
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\Fusion")

    # ── 训练 ──
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--start_step", type=int, default=0)

    # ── 时序损失 ──
    parser.add_argument("--temporal_loss_weight", type=float, default=0.06,
                        help="时序损失权重 λ_tc，叠加在 MSE 之上 (建议 0.01~0.1)")

    # ── 输出 ──
    from datetime import datetime as _dt
    parser.add_argument("--output_dir", type=str,
                        default=f"outputs/temporal_{_dt.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--checkpointing_steps", type=int, default=3000)
    parser.add_argument("--validation_steps", type=int, default=250)

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


# ============ Dataset ============

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, vi_dir, ir_dir, gt_dir, num_frames=4, video_size=(256, 256)):
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.gt_dir = gt_dir
        self.num_frames = num_frames
        self.video_size = video_size
        self.samples = self._find_samples()
        print(f"VideoDataset: {len(self.samples)} clips")

    def _find_samples(self):
        samples = []
        vi_root, ir_root, gt_root = Path(self.vi_dir), Path(self.ir_dir), Path(self.gt_dir)
        for root in [vi_root, ir_root, gt_root]:
            if not root.exists():
                print(f"Warning: {root} does not exist")
                return samples

        for gt_dir in sorted(gt_root.iterdir()):
            if not gt_dir.is_dir():
                continue
            name = gt_dir.name
            vi_dir = vi_root / name
            ir_dir = ir_root / name
            if not vi_dir.exists() or not ir_dir.exists():
                continue

            has_vi = any(vi_dir.glob('*.png')) or any(vi_dir.glob('*.jpg'))
            has_ir = any(ir_dir.glob('*.png')) or any(ir_dir.glob('*.jpg'))
            has_gt = any(gt_dir.glob('*.png')) or any(gt_dir.glob('*.jpg'))
            if has_vi and has_ir and has_gt:
                samples.append({'vi_dir': vi_dir, 'ir_dir': ir_dir, 'gt_dir': gt_dir})
        return samples

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        dummy = {
            'vi_video': torch.randn(3, self.num_frames, *self.video_size),
            'ir_video': torch.randn(3, self.num_frames, *self.video_size),
            'gt_video': torch.randn(3, self.num_frames, *self.video_size),
        }
        if not self.samples:
            return dummy

        sample = self.samples[idx % len(self.samples)]
        get_files = lambda d: sorted(list(d.glob('*.png')) + list(d.glob('*.jpg')))
        vi_all, ir_all, gt_all = get_files(sample['vi_dir']), get_files(sample['ir_dir']), get_files(sample['gt_dir'])

        vi_map = {f.stem: f for f in vi_all}
        gt_map = {f.stem: f for f in gt_all}
        ir_map = {f.stem: f for f in ir_all}
        common = sorted(set(vi_map) & set(gt_map))

        if common:
            vi_files = [vi_map[s] for s in common]
            gt_files = [gt_map[s] for s in common]
            ir_files = [ir_map.get(s, ir_all[min(i, len(ir_all)-1)]) for i, s in enumerate(common)]
        else:
            n = min(len(vi_all), len(gt_all))
            vi_files, gt_files, ir_files = vi_all[:n], gt_all[:n], ir_all
            if n == 0:
                return dummy

        num_groups = len(vi_files) // self.num_frames
        if num_groups > 0:
            g = np.random.randint(0, num_groups)
            indices = np.arange(g * self.num_frames, (g + 1) * self.num_frames)
        else:
            indices = np.linspace(0, len(vi_files) - 1, self.num_frames).astype(int)

        vi_frames = self._load_frames(vi_files, indices)
        gt_frames = self._load_frames(gt_files, indices)

        ir_frame = self._load_single(ir_files, indices[0])
        ir_frames = ir_frame.unsqueeze(1).repeat(1, self.num_frames, 1, 1)

        return {'vi_video': vi_frames, 'ir_video': ir_frames, 'gt_video': gt_frames}

    def _load_frames(self, files, indices):
        frames = []
        for i in indices:
            img = bgr_to_gray_3ch(cv2.imread(str(files[i])))
            img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
            frames.append(img.astype(np.float32) / 255.0 * 2 - 1)
        return torch.from_numpy(np.transpose(np.stack(frames), (3, 0, 1, 2))).float()

    def _load_single(self, files, idx):
        idx = min(idx, len(files) - 1)
        img = bgr_to_gray_3ch(cv2.imread(str(files[idx])))
        img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
        return torch.from_numpy(np.transpose(img.astype(np.float32) / 255.0 * 2 - 1, (2, 0, 1))).float()


# ============ 验证 ============

@torch.no_grad()
def run_validation(model, noise_scheduler, vi_video, ir_video, device,
                   num_steps=20, time_shift=13.0, ir_anchor_idx: int = 0):
    model.eval_mode()
    B, C, T, H, W = vi_video.shape

    vi_perm = vi_video.to(device).permute(0, 2, 1, 3, 4)
    ir_perm = ir_video.to(device).permute(0, 2, 1, 3, 4)
    vi_latent = model.vae_encode(vi_perm)
    ir_latent = model.vae_encode(ir_perm)

    vi_frames = vi_perm.float() if model.ir_flow_prop is not None else None
    precomputed_flows = None
    if model.ir_flow_prop is not None and vi_frames is not None:
        if ir_anchor_idx > 0:
            precomputed_flows = model.ir_flow_prop.estimate_bidirectional_flows(vi_frames)
        else:
            precomputed_flows = model.ir_flow_prop.estimate_flow(vi_frames)

    noise_scheduler.set_timesteps(num_steps, time_shift=time_shift, device=device)
    x = vi_latent.clone()
    for t_step in noise_scheduler.timesteps:
        t_batch = t_step.unsqueeze(0).repeat(B).to(device)
        v_pred = model.forward(x_t=x, t=t_batch, ir_latent=ir_latent,
                               vi_frames=vi_frames,
                               _precomputed_flows=precomputed_flows,
                               ir_anchor_idx=int(ir_anchor_idx))
        x = noise_scheduler.step(v_pred, t_step, x)

    pred = model.vae_decode_chunk(x, chunk_size=T)
    pred = rearrange(pred, '(b t) c h w -> b c t h w', b=B, t=T)
    pred = torch.clamp((pred + 1) / 2, 0, 1)

    model.train_mode()
    return pred


def save_validation_results(vi_video, ir_video, gt_video, pred_video, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)
    T = pred_video.shape[2]
    n = min(T, 4)
    indices = np.linspace(0, T - 1, n).astype(int)

    fig, axes = plt.subplots(n, 4, figsize=(12, n * 2.5))
    if n == 1:
        axes = axes.reshape(1, -1)

    titles = ['VI', 'IR', 'GT', 'Pred']
    for row, idx in enumerate(indices):
        for col, (title, video) in enumerate(zip(titles, [vi_video, ir_video, gt_video, pred_video])):
            fidx = 0 if title == 'IR' else idx
            frame = video[0, :, fidx].float().permute(1, 2, 0).cpu().numpy()
            if title != 'Pred':
                frame = (frame + 1) / 2
            frame = np.clip(frame, 0, 1).astype(np.float32)
            axes[row, col].imshow(frame)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, 0].set_ylabel(f'F{idx}', fontsize=10)

    plt.suptitle(f'Step {step}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"test_step{step}.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ============ 训练 ============

def main(args):
    # ── Accelerator ──
    logging_dir = Path(args.output_dir, "logs")
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(42)

    # ── 模型 (含 HFP) ──
    class _Opt:
        model_path = None
        vae_path = "stabilityai/stable-video-diffusion-img2vid"

    model = VideoControlNetModel(
        _Opt(), device=accelerator.device,
        use_hfp=True,
        hfp_block_indices=(1, 4, 7),
        hfp_ratio=(0.25, 0.25),
        hfp_patch=(4, 4),
        hfp_use_dct=True,
        hfp_serial=False,
        hfp_init='zeros',
        raft_model_path=args.raft_model_path, latent_scale=4,
    )

    # ── 加载权重 ──
    from safetensors.torch import load_file
    if args.resume_checkpoint_dir and os.path.exists(args.resume_checkpoint_dir):
        ctrl_p = os.path.join(args.resume_checkpoint_dir, "model.safetensors")
        unet_p = os.path.join(args.resume_checkpoint_dir, "model_1.safetensors")
        if os.path.exists(ctrl_p):
            ret = model.video_controlnet.load_state_dict(
                {k.replace('module.', ''): v for k, v in load_file(ctrl_p).items()}, strict=False)
            logger.info(f"ControlNet 恢复 (missing={len(ret.missing_keys)})")
        if os.path.exists(unet_p):
            ret = model.unet.load_state_dict(
                {k.replace('module.', ''): v for k, v in load_file(unet_p).items()}, strict=False)
            logger.info(f"UNet 恢复 (missing={len(ret.missing_keys)})")
    else:
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            ret = model.video_controlnet.load_state_dict(
                {k.replace('module.', ''): v for k, v in load_file(args.pretrained_model_path).items()},
                strict=False)
            logger.info(f"ControlNet 加载 (missing={len(ret.missing_keys)})")
        if args.unet_checkpoint and os.path.exists(args.unet_checkpoint):
            ret = model.unet.load_state_dict(
                {k.replace('module.', ''): v for k, v in load_file(args.unet_checkpoint).items()},
                strict=False)
            logger.info(f"UNet 加载 (missing={len(ret.missing_keys)})")

    # ── VAE 替换 ──
    replace_vae_to_4x(model, r"pretrained_weight/vae_4x_tvt.ckpt")

    # ── 冻结：只训 UNet ──
    model.freeze_for_training(train_controlnet=False, train_unet=True)
    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.get_trainable_parameters())
        logger.info(f"可训练参数: {trainable / 1e6:.2f}M (仅 UNet)")

    noise_scheduler = model.noise_scheduler

    # ── 时序损失 ──
    use_temporal = args.temporal_loss_weight > 0
    if use_temporal:
        temporal_loss_fn = LatentTemporalConsistencyLoss().to(accelerator.device)
        logger.info(f"时序损失: λ={args.temporal_loss_weight}")

    # ── 数据 ──
    dataset = VideoDataset(args.vi_data_dir, args.ir_data_dir, args.gt_data_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0,
    )

    # ── 优化器 ──
    trainable_params = list(model.get_trainable_parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    remaining_steps = args.max_train_steps - args.start_step
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=min(500, remaining_steps // 10),
        num_training_steps=remaining_steps,
    )

    # ── Accelerator prepare ──
    model.unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model.unet, optimizer, dataloader, lr_scheduler
    )
    model.video_controlnet.half()

    # ── 打印配置 ──
    if accelerator.is_main_process:
        print("=" * 60)
        print(f"ControlNet: {args.pretrained_model_path}")
        print(f"UNet:       {args.unet_checkpoint}")
        print(f"输出:       {args.output_dir}")
        print(f"训练:       step {args.start_step} → {args.max_train_steps}, "
              f"lr={args.learning_rate}, bs={args.train_batch_size}")
        print(f"时序损失:   λ={args.temporal_loss_weight}" if use_temporal else "时序损失:   关闭")
        print("=" * 60)

    # ── 训练循环 ──
    global_step = args.start_step
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        initial=global_step, total=args.max_train_steps, desc="Steps",
                        disable=not accelerator.is_local_main_process)

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

                with torch.no_grad():
                    vi_latent = model.vae_encode(vi.permute(0, 2, 1, 3, 4)).float()
                    ir_latent = model.vae_encode(ir.permute(0, 2, 1, 3, 4)).float()
                    gt_latent = model.vae_encode(gt.permute(0, 2, 1, 3, 4)).float()

                velocity = gt_latent - vi_latent
                t = torch.rand(B, device=accelerator.device)
                x_t = (1 - t.view(B, 1, 1, 1, 1)) * vi_latent + t.view(B, 1, 1, 1, 1) * gt_latent

                vi_frames = vi.permute(0, 2, 1, 3, 4) if model.ir_flow_prop is not None else None

                with accelerator.autocast():
                    pred_velocity = model.forward(
                        x_t=x_t, t=t, ir_latent=ir_latent, vi_frames=vi_frames,
                    )
                pred_velocity = pred_velocity.float()

                loss_vel = F.mse_loss(pred_velocity, velocity)

                if use_temporal:
                    pred_fused = vi_latent + pred_velocity
                    loss_tc = temporal_loss_fn(pred_fused, gt_latent)
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
                    "loss_tc": loss_tc.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }, step=global_step)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    try:
                        pred = run_validation(model, noise_scheduler, vi.detach(), ir.detach(),
                                              accelerator.device)
                        save_validation_results(vi.detach(), ir.detach(), gt.detach(), pred,
                                                os.path.join(args.output_dir, "validations"),
                                                global_step)
                    except Exception:
                        import traceback; traceback.print_exc()

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
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())

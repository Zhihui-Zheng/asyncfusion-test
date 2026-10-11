#!/usr/bin/env python
# coding=utf-8
"""
Flow Matching 时序一致性微调版 - step-consistent L_tc

区别于 train_latent_vae48_tem.py:
  原脚本: pred_fused = vi_latent + pred_velocity，再和 gt_latent 做 L_tc
  本脚本: pred_next = x_t + dt * pred_velocity
          gt_next   = x_t + dt * (gt_latent - vi_latent)
          对 pred_next / gt_next 做 L_tc

这样时序正则约束的是“当前 x_t 沿速度场走一小步后的低频时序变化”，
更符合 Flow Matching 的训练语义。
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path

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
    VideoDataset,
    replace_vae_to_4x,
    run_validation,
    save_validation_results,
)

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Flow Matching 时序微调 - step-consistent temporal loss"
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
        default=r"E:\paper2\codebase\STAR\outputs\hfp_20260410_152654\checkpoint-3000\model1.safetensors",
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
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--start_step", type=int, default=0)
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
        default=f"outputs/temporal429_step_{_dt.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=3000)
    parser.add_argument("--validation_steps", type=int, default=250)

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


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

    # 时序微调阶段只训练 UNet/HFP，ControlNet/VAE/RAFT 冻结。
    model.freeze_for_training(train_controlnet=False, train_unet=True)

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.get_trainable_parameters())
        logger.info(f"可训练参数: {trainable / 1e6:.2f}M (仅 UNet/HFP)")
        logger.info(
            f"step-consistent L_tc: weight={args.temporal_loss_weight}, "
            f"dt={args.temporal_dt}"
        )

    noise_scheduler = model.noise_scheduler
    temporal_loss_fn = None
    use_temporal = args.temporal_loss_weight > 0
    if use_temporal:
        temporal_loss_fn = LatentTemporalConsistencyLoss().to(accelerator.device)

    dataset = VideoDataset(args.vi_data_dir, args.ir_data_dir, args.gt_data_dir)
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

    if accelerator.is_main_process:
        print("=" * 60)
        print("Flow Matching 时序微调: step-consistent L_tc")
        print(f"ControlNet: {args.pretrained_model_path}")
        print(f"UNet:       {args.unet_checkpoint}")
        print(f"输出:       {args.output_dir}")
        print(
            f"训练:       step {args.start_step} -> {args.max_train_steps}, "
            f"lr={args.learning_rate}, bs={args.train_batch_size}"
        )
        print(f"时序损失:   λ={args.temporal_loss_weight}, dt={args.temporal_dt}")
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

#!/usr/bin/env python
# coding=utf-8
# 双模态视频融合训练脚本 - 无监督训练
# 结合 VideoControlNet, Vid2VidSDUNet 和 TemCoCo 的交叉注意力机制

import argparse
import os
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from einops import rearrange

from typing import Tuple

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.logger import get_logger

# 双模态融合模型和损失函数
from Fusion.dual_modal_model import (
    DualModalControlledV2VUNet,
    CombinedLoss,
    print_model_info
)
from Fusion.image_encoder import FrozenCLIPImageEncoder

from diffusers import AutoencoderKLTemporalDecoder

check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Dual-Modal Video Fusion Training Script (Unsupervised)")
    
    # 模型路径
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=r"outputs\fusion_training\checkpoint-14000\model.safetensors",
        help="Path to pretrained STAR model weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dual_modal_fusion",
        help="The output directory for checkpoints.",
    )
    
    # 训练参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=2500)
    parser.add_argument("--max_train_steps", type=int, default=10000, help="Total training steps.")
    parser.add_argument("--checkpointing_steps", type=int, default=2000, help="Save checkpoint every X steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    
    # 数据参数
    parser.add_argument("--train_data_dir", type=str, required=False, default=r"Fusion\dataset\train\Fusion", help="Training data directory.")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames per video.")
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="Video size (H W).")
    
    # 模型参数
    parser.add_argument("--fusion_embed_dim", type=int, default=32, help="Fusion module embedding dim.")
    parser.add_argument("--fusion_num_blocks", type=int, default=2, help="Fusion encoder blocks.")
    
    # 损失权重
    parser.add_argument("--v_pred_weight", type=float, default=1.0, help="V-prediction loss weight.")
    parser.add_argument("--fusion_weight", type=float, default=0.1, help="Fusion loss weight.")
    parser.add_argument("--ssim_weight", type=float, default=1.0)
    parser.add_argument("--intensity_weight", type=float, default=5.0)
    parser.add_argument("--gradient_weight", type=float, default=5.0)
    parser.add_argument("--color_weight", type=float, default=10.0)
    parser.add_argument("--temporal_weight", type=float, default=1.0)
    
    # CLIP 编码器
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s32b_b79k")
    parser.add_argument("--clip_arch", type=str, default="ViT-H-14")
    
    # 其他
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class DualModalVideoDataset(torch.utils.data.Dataset):
    """
    双模态视频数据集
    
    数据目录结构:
    root_folder/
        video1/
            ir/
                frame_0000.png
                frame_0001.png
                ...
            vi/
                frame_0000.png
                frame_0001.png
                ...
        video2/
            ...
    
    或者:
    root_folder/
        ir/
            video1/
                frame_0000.png
                ...
        vi/
            video1/
                frame_0000.png
                ...
    """
    def __init__(self, 
                 root_folders,
                 num_frames=32,
                 video_size=(256, 256)):
        self.root_folders = root_folders if isinstance(root_folders, list) else [root_folders]
        self.num_frames = num_frames
        self.video_size = video_size
        
        self.samples = self._scan_videos()
        
        # 图像变换
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(video_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def _scan_videos(self):
        """扫描视频目录"""
        samples = []
        
        for root in self.root_folders:
            root = Path(root)
            
            # 检查目录结构
            if (root / 'ir').exists() and (root / 'vi').exists():
                # 结构: root/ir/video1/, root/vi/video1/
                ir_dir = root / 'ir'
                vi_dir = root / 'vi'
                
                for video_name in os.listdir(ir_dir):
                    ir_video_dir = ir_dir / video_name
                    vi_video_dir = vi_dir / video_name
                    
                    if ir_video_dir.is_dir() and vi_video_dir.is_dir():
                        ir_frames = sorted([f for f in os.listdir(ir_video_dir) if f.endswith(('.png', '.jpg', '.exr'))])
                        vi_frames = sorted([f for f in os.listdir(vi_video_dir) if f.endswith(('.png', '.jpg', '.exr'))])
                        
                        if len(ir_frames) >= self.num_frames and len(vi_frames) >= self.num_frames:
                            samples.append({
                                'ir_dir': str(ir_video_dir),
                                'vi_dir': str(vi_video_dir),
                                'ir_frames': ir_frames,
                                'vi_frames': vi_frames
                            })
            else:
                # 结构: root/video1/ir/, root/video1/vi/
                for video_name in os.listdir(root):
                    video_dir = root / video_name
                    ir_video_dir = video_dir / 'ir'
                    vi_video_dir = video_dir / 'vi'
                    
                    if ir_video_dir.is_dir() and vi_video_dir.is_dir():
                        ir_frames = sorted([f for f in os.listdir(ir_video_dir) if f.endswith(('.png', '.jpg', '.exr'))])
                        vi_frames = sorted([f for f in os.listdir(vi_video_dir) if f.endswith(('.png', '.jpg', '.exr'))])
                        
                        if len(ir_frames) >= self.num_frames and len(vi_frames) >= self.num_frames:
                            samples.append({
                                'ir_dir': str(ir_video_dir),
                                'vi_dir': str(vi_video_dir),
                                'ir_frames': ir_frames,
                                'vi_frames': vi_frames
                            })
        
        print(f"Found {len(samples)} video pairs")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_frames(self, frame_dir, frame_names, start_idx):
        """加载帧序列"""
        from PIL import Image
        import numpy as np
        
        frames = []
        for i in range(self.num_frames):
            idx = start_idx + i
            frame_path = os.path.join(frame_dir, frame_names[idx])
            
            if frame_path.endswith('.exr'):
                import cv2
                frame = cv2.imread(frame_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.clip(frame, 0, 1)
                frame = (frame * 255).astype(np.uint8)
                frame = Image.fromarray(frame)
            else:
                frame = Image.open(frame_path).convert('RGB')
            
            frame = self.transform(frame)
            frames.append(frame)
        
        return torch.stack(frames, dim=0)  # [T, C, H, W]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        ir_dir = sample['ir_dir']
        vi_dir = sample['vi_dir']
        ir_frames = sample['ir_frames']
        vi_frames = sample['vi_frames']
        
        # 随机选择起始帧
        max_start = min(len(ir_frames), len(vi_frames)) - self.num_frames
        start_idx = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
        
        # 加载帧
        ir_video = self._load_frames(ir_dir, ir_frames, start_idx)  # [T, C, H, W]
        vi_video = self._load_frames(vi_dir, vi_frames, start_idx)  # [T, C, H, W]
        
        # 使用第一帧作为条件图像
        condition_image = vi_video[0]  # [C, H, W]
        
        return {
            'ir': ir_video,  # [T, C, H, W]
            'vi': vi_video,  # [T, C, H, W]
            'condition_image': condition_image  # [C, H, W]
        }


def main():
    args = parse_args()
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,
    )
    
    # =========================================
    # 构建模型
    # =========================================
    
    # 图片编码器 (冻结)
    image_encoder = FrozenCLIPImageEncoder(
        pretrained=args.clip_pretrained,
        arch=args.clip_arch,
        device='cuda',
        freeze=True
    )
    image_encoder.requires_grad_(False)
    logger.info(f'Built image encoder with FrozenCLIPImageEncoder (arch: {args.clip_arch})')
    
    # 双模态融合模型
    model = DualModalControlledV2VUNet(
        fusion_embed_dim=args.fusion_embed_dim,
        fusion_num_blocks=args.fusion_num_blocks,
        pretrained_model_path=args.pretrained_model_path
    )
    model.freeze_pretrained()
    
    logger.info('Built DualModalControlledV2VUNet')
    print_model_info(model)
    
    # 噪声调度器
    sigmas = noise_schedule(
        schedule='logsnr_cosine_interp',
        n=1000,
        zero_terminal_snr=True,
        scale_min=2.0,
        scale_max=4.0
    )
    noise_scheduler = GaussianDiffusion(sigmas=sigmas)
    logger.info('Built noise_scheduler')
    
    # Temporal VAE (冻结)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
    )
    vae.eval()
    vae.requires_grad_(False)
    logger.info('Built Temporal VAE')
    
    # 损失函数
    loss_fn = CombinedLoss(
        v_pred_weight=args.v_pred_weight,
        fusion_weight=args.fusion_weight,
        ssim_weight=args.ssim_weight,
        intensity_weight=args.intensity_weight,
        gradient_weight=args.gradient_weight,
        color_weight=args.color_weight,
        temporal_weight=args.temporal_weight
    )
    logger.info('Built CombinedLoss')
    
    # =========================================
    # 数据集
    # =========================================
    train_dataset = DualModalVideoDataset(
        root_folders=[args.train_data_dir],
        num_frames=args.num_frames,
        video_size=tuple(args.video_size)
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    # =========================================
    # 优化器
    # =========================================
    params_to_optimize = model.get_trainable_parameters()
    total_params = sum(p.numel() for p in params_to_optimize)
    logger.info(f"Total trainable parameters: {total_params / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    model.train()
    
    # 精度设置
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # =========================================
    # VAE 编码/解码函数
    # =========================================
    def tensor2latent(t):
        """将视频编码为 latent"""
        # t: [B, T, C, H, W] -> [B, C, T, H, W]
        if t.dim() == 5 and t.shape[1] != t.shape[2]:
            # 检查是 [B, T, C, H, W] 还是 [B, C, T, H, W]
            if t.shape[2] == 3:  # [B, T, C, H, W]
                t = t.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
        
        video_length = t.shape[2]
        t = rearrange(t, "b c f h w -> (b f) c h w")
        
        latents_list = []
        for ind in range(0, t.shape[0], 1):
            latents_list.append(vae.encode(t[ind:ind+1]).latent_dist.sample())
        latents = torch.cat(latents_list, dim=0)
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = latents * vae.config.scaling_factor
        return latents
    
    def vae_decode_chunk(z, chunk_size=3):
        """解码 latent 为视频"""
        z = rearrange(z, "b c f h w -> (b f) c h w")
        video = []
        for ind in range(0, z.shape[0], chunk_size):
            num_f = z[ind:ind+chunk_size].shape[0]
            decoded = vae.decode(z[ind:ind+chunk_size] / vae.config.scaling_factor, num_frames=num_f).sample
            video.append(decoded)
        video = torch.cat(video)
        return video
    
    # =========================================
    # 训练循环
    # =========================================
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    
    for epoch in range(0, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            
            with accelerator.accumulate(model):
                # 获取数据: [B, T, C, H, W]
                ir_video = batch['ir'].to(accelerator.device, dtype=weight_dtype)
                vi_video = batch['vi'].to(accelerator.device, dtype=weight_dtype)
                condition_image = batch['condition_image'].to(accelerator.device, dtype=weight_dtype)
                
                actual_model = model.module if hasattr(model, 'module') else model
                
                # Step 1: 将视频转换为 [0, 1] 范围
                # ir_video, vi_video: [B, T, C, H, W], 范围 [-1, 1]
                ir_video_01 = (ir_video + 1) / 2  # [0, 1]
                vi_video_01 = (vi_video + 1) / 2  # [0, 1]
                ir_video_01 = torch.clamp(ir_video_01, 0, 1)
                vi_video_01 = torch.clamp(vi_video_01, 0, 1)
                
                # 转换为 [B, C, T, H, W] 格式
                ir_video_bct = rearrange(ir_video_01, 'b t c h w -> b c t h w')
                vi_video_bct = rearrange(vi_video_01, 'b t c h w -> b c t h w')
                
                # Step 2: 像素空间融合 (这个需要梯度)
                fused_video = actual_model.pixel_fusion(ir_video_bct, vi_video_bct)  # [B, C, T, H, W], [0, 1]
                
                # Step 3: 转换为 [-1, 1] 并 VAE 编码
                fused_video_norm = fused_video * 2 - 1  # [-1, 1]
                
                with torch.no_grad():
                    # VAE 编码融合结果
                    fused_latent = tensor2latent(fused_video_norm)  # [B, C, T, H/8, W/8]
                    
                    # 图片编码器
                    image_features = image_encoder(condition_image).detach()
                    if image_features.dim() == 2:
                        B, D = image_features.shape
                        y = image_features.unsqueeze(1).expand(B, 77, D)
                    else:
                        y = image_features
                
                # Step 4: 扩散过程
                bsz = fused_latent.shape[0]
                timesteps = torch.randint(0, 1000, (bsz,), device=fused_latent.device).long()
                noise = torch.randn_like(fused_latent)
                noised_fused = noise_scheduler.diffuse(fused_latent, timesteps, noise=noise)
                
                # Step 5: UNet 前向 (使用 fused_latent 作为 hint)
                v_pred = actual_model.controlled_unet(
                    x=noised_fused,
                    t=timesteps,
                    y=y,
                    hint=fused_latent
                )
                
                # Step 6: V-prediction 目标
                v_target = noise_scheduler.get_velocity(x0=fused_latent, xt=noised_fused, t=timesteps)
                
                # Step 7: 计算损失
                # 使用融合模块的直接输出 (fused_video) 计算像素空间损失
                loss_dict = loss_fn(
                    v_pred=v_pred,
                    v_target=v_target,
                    img_fused=fused_video,  # [B, C, T, H, W], [0, 1]
                    img_ir=ir_video_bct,    # [B, C, T, H, W], [0, 1]
                    img_vi=vi_video_bct     # [B, C, T, H, W], [0, 1]
                )
                
                total_loss = loss_dict['total_loss']
                
                # 反向传播
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    actual_model = model.module if hasattr(model, 'module') else model
                    params_to_clip = actual_model.get_trainable_parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
                
                # 保存检查点
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                
                # 日志
                logs = {
                    "loss_v": loss_dict['loss_v_pred'].detach().item(),
                    "loss_fusion": loss_dict.get('loss_fusion_total', torch.tensor(0.0)).detach().item(),
                    "total_loss": total_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                if global_step >= args.max_train_steps:
                    break
    
    accelerator.end_training()


if __name__ == "__main__":
    main()


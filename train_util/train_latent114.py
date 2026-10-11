#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Video ControlNet Training Script with Flow Matching (VI → Fusion)
# 
# 新训练策略 (从可见光到融合):
# ==================
# 架构:
#   - VI视频 (4帧) → VAE编码 → vi_latent (起点)
#   - GT视频 (4帧) → VAE编码 → gt_latent (终点)
#   - IR红外 (1帧→复制4帧) → VAE编码 → ir_latent → ControlNet (hint)
#   - 轨迹: velocity = gt_latent - vi_latent
#   - 插值: x_t = (1-t) * vi_latent + t * gt_latent
#   - 模型预测 velocity → Loss = MSE(pred_velocity, velocity)
#
# 推理时:
#   - 从 vi_latent 开始积分 velocity → 融合结果
#
# 默认全部放开训练:
#   🔥 可训练: VideoControlNet, Vid2VidSDUNet
#   ❄️ 冻结: VAE (始终)
#
# 可通过参数控制:
#   --freeze_controlnet: 冻结 ControlNet
#   --freeze_unet: 冻结 UNet

import os
# 解决 OMP 冲突
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

# 直接使用 dual_modal_fusion_model.py 中的模型
from Fusion.dual_modal_fusion_model import VideoControlNetModel
from Fusion.flow_match import FlowMatchDiscreteScheduler

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


# ============ 颜色一致性损失函数 ============

def color_mean_loss(pred, target):
    """
    颜色均值匹配损失 (非常轻量)
    
    匹配每个通道的全局均值，确保整体色调一致
    
    Args:
        pred: (B, C, T, H, W) 预测的 latent
        target: (B, C, T, H, W) 目标 latent
    
    Returns:
        loss: 颜色均值损失
    """
    pred_mean = pred.mean(dim=[2, 3, 4])    # (B, C)
    target_mean = target.mean(dim=[2, 3, 4])  # (B, C)
    return F.mse_loss(pred_mean, target_mean)


def get_model_numel(model: torch.nn.Module):
    """获取模型参数数量"""
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
    parser = argparse.ArgumentParser(description="Video ControlNet Training (VI → Fusion)")
    
    # Model
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=r"pretrained_weight/model.pt",
        help="Path to pretrained model weights (ControlNet + UNet). 用于初始化模型权重。",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid",
        help="Path to VAE weights (local directory or Hugging Face model name).",
    )
    parser.add_argument(
        "--train_controlnet",
        action="store_true",
        default=True,
        help="Whether to train ControlNet (default: True)",
    )
    parser.add_argument(
        "--freeze_controlnet",
        action="store_true",
        help="Freeze ControlNet (overrides --train_controlnet)",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        default=True,
        help="Whether to train UNet (default: True)",
    )
    parser.add_argument(
        "--freeze_unet",
        action="store_true",
        help="Freeze UNet (overrides --train_unet)",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/vi_to_fusion",
        help="The output directory for checkpoints and logs.",
    )
    
    # Training
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=2500)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
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

    # Data (VI + IR → GT)
    parser.add_argument("--vi_data_dir", type=str, default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\visible1", 
                        help="可见光输入目录 - num_frames 帧 (起点)")
    parser.add_argument("--ir_data_dir", type=str, default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\infrared", 
                        help="红外输入目录 - 1 帧 → 复制到 num_frames 帧 → ControlNet hint")
    parser.add_argument("--gt_data_dir", type=str, default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\Fusion", 
                        help="GT 目录 - num_frames 帧 (终点/监督目标)")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="帧数 (VI 和 GT 都是这么多帧，IR 1 帧复制)")
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    
    # Flow Matching
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--time_shift", type=float, default=13.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    
    # 额外损失权重
    parser.add_argument("--color_loss_weight", type=float, default=0.1,
                        help="颜色均值匹配损失权重")
    
    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
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
    
    输入条件: 
        - vi_dir: 可见光 - 取 num_frames 帧 (起点)
        - ir_dir: 红外 - 取 1 帧，复制到 num_frames 帧 (ControlNet hint)
    
    监督目标: GT (Ground Truth) - 取 num_frames 帧 (终点)
    
    轨迹: velocity = gt_latent - vi_latent
    """
    def __init__(self, vi_dir, ir_dir, gt_dir, num_frames=4, video_size=(512, 512)):
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.gt_dir = gt_dir
        self.num_frames = num_frames
        self.video_size = video_size
        
        # 查找配对数据
        self.samples = self._find_samples()
        
        print(f"Found {len(self.samples)} video triplets")
        print(f"  VI (可见光): {vi_dir} - {num_frames} 帧 (起点)")
        print(f"  IR (红外):   {ir_dir} - 1 帧 → 复制 {num_frames} 帧 (ControlNet hint)")
        print(f"  GT (目标):   {gt_dir} - {num_frames} 帧 (终点)")
    
    def _find_samples(self):
        """查找配对的视频数据样本"""
        samples = []
        vi_root = Path(self.vi_dir)
        ir_root = Path(self.ir_dir)
        gt_root = Path(self.gt_dir)
        
        if not vi_root.exists():
            print(f"Warning: VI dir {vi_root} does not exist")
            return samples
        
        if not ir_root.exists():
            print(f"Warning: IR dir {ir_root} does not exist")
            return samples
        
        if not gt_root.exists():
            print(f"Warning: GT dir {gt_root} does not exist")
            return samples
        
        for vi_sample_dir in sorted(vi_root.iterdir()):
            if vi_sample_dir.is_dir():
                name = vi_sample_dir.name
                ir_sample_dir = ir_root / name
                gt_sample_dir = gt_root / name
                
                if not ir_sample_dir.exists() or not gt_sample_dir.exists():
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
                        'num_vi_frames': len(vi_frames),
                        'num_ir_frames': len(ir_frames),
                        'num_gt_frames': len(gt_frames),
                    })
        
        return samples
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, idx):
        """
        返回:
        - vi_video: (3, T, H, W) VI视频 (起点)
        - ir_video: (3, T, H, W) IR视频 (1帧复制到T帧，ControlNet hint)
        - gt_video: (3, T, H, W) GT视频 (终点)
        """
        if len(self.samples) == 0:
            return {
                'vi_video': torch.randn(3, self.num_frames, *self.video_size),
                'ir_video': torch.randn(3, self.num_frames, *self.video_size),
                'gt_video': torch.randn(3, self.num_frames, *self.video_size),
            }
        
        sample = self.samples[idx % len(self.samples)]
        
        vi_files = sorted(list(sample['vi_dir'].glob('*.png')) + list(sample['vi_dir'].glob('*.jpg')))
        ir_files = sorted(list(sample['ir_dir'].glob('*.png')) + list(sample['ir_dir'].glob('*.jpg')))
        gt_files = sorted(list(sample['gt_dir'].glob('*.png')) + list(sample['gt_dir'].glob('*.jpg')))
        
        num_files = min(len(vi_files), len(gt_files))
        
        if num_files == 0:
            return {
                'vi_video': torch.randn(3, self.num_frames, *self.video_size),
                'ir_video': torch.randn(3, self.num_frames, *self.video_size),
                'gt_video': torch.randn(3, self.num_frames, *self.video_size),
            }
        
        # 随机选择连续帧的起始位置
        max_start = num_files - self.num_frames
        if max_start > 0:
            start_idx = np.random.randint(0, max_start + 1)
            indices = np.arange(start_idx, start_idx + self.num_frames)
        else:
            indices = np.linspace(0, num_files - 1, self.num_frames).astype(int)
        
        # 加载 VI 和 GT 的 T 帧
        vi_frames = self._load_frames_by_indices(vi_files, indices)   # (3, T, H, W)
        gt_frames = self._load_frames_by_indices(gt_files, indices)   # (3, T, H, W)
        
        # 加载 IR 的 1 帧，然后复制到 T 帧
        ir_start_idx = indices[0]
        ir_frame = self._load_single_frame(ir_files, ir_start_idx)  # (3, H, W)
        ir_frames = ir_frame.unsqueeze(1).repeat(1, self.num_frames, 1, 1)  # (3, T, H, W)
        
        return {
            'vi_video': vi_frames,   # (3, T, H, W) 起点
            'ir_video': ir_frames,   # (3, T, H, W) ControlNet hint
            'gt_video': gt_frames,   # (3, T, H, W) 终点
        }
    
    def _load_frames_by_indices(self, frame_files, indices):
        """按索引加载帧"""
        frames = []
        for i in indices:
            img = cv2.imread(str(frame_files[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
            img = img.astype(np.float32) / 255.0
            img = img * 2 - 1
            frames.append(img)
        
        frames = np.stack(frames, axis=0)
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        return torch.from_numpy(frames).float()
    
    def _load_single_frame(self, frame_files, idx):
        """加载单帧"""
        if idx >= len(frame_files):
            idx = len(frame_files) - 1
        img = cv2.imread(str(frame_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
        img = img.astype(np.float32) / 255.0
        img = img * 2 - 1
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()


def vae_decode_chunk(z, vae, chunk_size=3):
    """分块 VAE 解码"""
    z = rearrange(z, "b c f h w -> (b f) c h w")
    video = []
    for ind in range(0, z.shape[0], chunk_size):
        num_f = z[ind:ind+chunk_size].shape[0]
        video.append(vae.decode(z[ind:ind+chunk_size] / vae.config.scaling_factor, num_frames=num_f).sample)
    video = torch.cat(video)
    return video


@torch.no_grad()
def run_validation(
    model, 
    noise_scheduler, 
    vi_video,
    ir_video,
    device, 
    weight_dtype,
    num_steps=20,
    time_shift=13.0,
):
    """
    运行验证推理 (从 vi_latent 开始积分)
    
    Args:
        model: VideoControlNetModel
        noise_scheduler: Flow Matching 调度器
        vi_video: (B, 3, T, H, W) VI视频 (起点)
        ir_video: (B, 3, T, H, W) IR视频 (ControlNet hint)
    
    Returns:
        pred_video: (B, C, T, H, W) 融合后的视频
    """
    model.eval_mode()
    
    B, C, T, H, W = vi_video.shape
    
    # 移动到设备
    vi_video_device = vi_video.to(device)
    ir_video_device = ir_video.to(device)
    
    # VAE 编码 VI (起点)
    vi_video_perm = vi_video_device.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
    vi_latent = model.vae_encode(vi_video_perm)  # (B, 4, T, H/8, W/8)
    
    # VAE 编码 IR (ControlNet hint)
    ir_video_perm = ir_video_device.permute(0, 2, 1, 3, 4)
    ir_latent = model.vae_encode(ir_video_perm)
    
    # 设置推理步数
    noise_scheduler.set_timesteps(num_steps, time_shift=time_shift, device=device)
    
    # 从 vi_latent 开始 (不是纯噪声!)
    x = vi_latent.clone()
    
    # Flow Matching 采样 (积分 velocity)
    for t_step in noise_scheduler.timesteps:
        t_batch = t_step.unsqueeze(0).repeat(B).to(device)
        
        # 预测 velocity
        v_pred = model.forward(
            x_t=x,
            t=t_batch,
            ir_latent=ir_latent,
        )
        
        # 更新 x
        x = noise_scheduler.step(v_pred, t_step, x)
    
    # VAE 解码
    pred_video = model.vae_decode_chunk(x, chunk_size=3)  # (B*T, C, H, W)
    pred_video = rearrange(pred_video, '(b t) c h w -> b c t h w', b=B, t=T)
    pred_video = (pred_video + 1) / 2
    pred_video = torch.clamp(pred_video, 0, 1)
    
    model.train_mode()
    return pred_video


def save_validation_results(vi_video, ir_video, gt_video, pred_video, save_dir, step):
    """
    保存验证结果
    
    Args:
        vi_video: (B, 3, T, H, W) VI视频 (起点)
        ir_video: (B, 3, T, H, W) IR视频 (hint)
        gt_video: (B, 3, T, H, W) GT视频 (终点)
        pred_video: (B, C, T, H, W) 预测视频
    """
    os.makedirs(save_dir, exist_ok=True)
    
    T = pred_video.shape[2]
    num_frames_to_show = min(T, 4)
    indices = np.linspace(0, T - 1, num_frames_to_show).astype(int)
    
    # 布局: 4行4列 (VI, IR, GT, Pred)
    fig, axes = plt.subplots(num_frames_to_show, 4, figsize=(12, num_frames_to_show * 2.5))
    
    if num_frames_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        # VI (起点)
        vi_frame = vi_video[0, :, idx].float().permute(1, 2, 0).cpu().numpy()
        vi_frame = (vi_frame + 1) / 2
        vi_frame = np.clip(vi_frame, 0, 1).astype(np.float32)
        axes[row, 0].imshow(vi_frame)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title('VI (起点)', fontsize=12, fontweight='bold')
        
        # IR (hint) - 所有帧相同，只在第一行显示实际帧
        ir_frame = ir_video[0, :, 0].float().permute(1, 2, 0).cpu().numpy()  # 只取第1帧
        ir_frame = (ir_frame + 1) / 2
        ir_frame = np.clip(ir_frame, 0, 1).astype(np.float32)
        if row == 0:
            axes[row, 1].imshow(ir_frame)
        else:
            gray_placeholder = np.ones_like(ir_frame) * 0.3
            axes[row, 1].imshow(gray_placeholder)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title('IR (hint)', fontsize=12, fontweight='bold')
        
        # GT (终点)
        gt_frame = gt_video[0, :, idx].float().permute(1, 2, 0).cpu().numpy()
        gt_frame = (gt_frame + 1) / 2
        gt_frame = np.clip(gt_frame, 0, 1).astype(np.float32)
        axes[row, 2].imshow(gt_frame)
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].set_title('GT (终点)', fontsize=12, fontweight='bold')
        
        # Pred (预测)
        pred_frame = pred_video[0, :, idx].float().permute(1, 2, 0).cpu().numpy()
        pred_frame = np.clip(pred_frame, 0, 1).astype(np.float32)
        axes[row, 3].imshow(pred_frame)
        axes[row, 3].axis('off')
        if row == 0:
            axes[row, 3].set_title('Pred', fontsize=12, fontweight='bold')
        
        axes[row, 0].set_ylabel(f'Frame {idx}', fontsize=10)
    
    plt.suptitle(f'Step {step} (VI → Fusion, IR as hint)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, f"test_step{step}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  验证图片已保存: {comparison_path}")


def main(args):
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
            self.model_path = args.pretrained_model_path if args.pretrained_model_path else None
            self.vae_path = args.vae_path
    
    opt = ModelConfig()
    model = VideoControlNetModel(opt, device=accelerator.device)
    
    # 确定冻结策略
    train_controlnet = args.train_controlnet and not args.freeze_controlnet
    train_unet = args.train_unet and not args.freeze_unet
    
    # 应用冻结策略
    logger.info("Applying freeze strategy...")
    model.freeze_for_training(train_controlnet=train_controlnet, train_unet=train_unet)
    
    # 打印可训练参数信息
    if accelerator.is_main_process:
        trainable_count = sum(p.numel() for p in model.get_trainable_parameters())
        ctrl_count = sum(p.numel() for p in model.video_controlnet.parameters())
        unet_count = sum(p.numel() for p in model.unet.parameters())
        total_count = ctrl_count + unet_count
        
        logger.info(f"Trainable: {trainable_count / 1e6:.2f}M / Total: {total_count / 1e6:.2f}M")
        logger.info(f"  ControlNet: {ctrl_count / 1e6:.2f}M {'🔥' if train_controlnet else '❄️'}")
        logger.info(f"  UNet: {unet_count / 1e6:.2f}M {'🔥' if train_unet else '❄️'}")
    
    # 使用模型内置的 noise_scheduler
    noise_scheduler = model.noise_scheduler
    logger.info("Built Flow Matching Scheduler (from model)")
    
    # VAE 已在 model 中创建
    logger.info("Built Temporal VAE (from model)")
    
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
    logger.info(f"Built dataset with {len(train_dataset)} samples")
    
    # ============ Optimizer ============
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not found, using standard AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
    
    trainable_params = model.get_trainable_parameters()
    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Total trainable parameters: {total_trainable / 1e6:.2f}M")
    
    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # ============ LR Scheduler ============
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
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
    
    # ============ Move to device ============
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
    
    # ============ Resume from checkpoint ============
    global_step = 0
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            if os.path.exists(args.output_dir):
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint-")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None
            else:
                path = None
        else:
            path = args.resume_from_checkpoint
            if path and not os.path.isdir(path):
                parent = os.path.dirname(path)
                if os.path.isdir(parent) and "checkpoint-" in os.path.basename(parent):
                    logger.info(f"检测到文件路径，自动使用父目录: {parent}")
                    path = parent
        
        if path and os.path.exists(path) and os.path.isdir(path):
            logger.info(f"Resuming from checkpoint: {path}")
            accelerator.load_state(path)
            basename = os.path.basename(path.rstrip("/\\"))
            if basename.startswith("checkpoint-"):
                global_step = int(basename.split("-")[1])
                logger.info(f"  Resumed at step {global_step}")
            else:
                logger.warning(f"无法从目录名 {basename} 解析 step，从 0 开始")
        else:
            logger.warning(f"Checkpoint not found or not a directory: {path}")
            logger.info("Starting from scratch")
    
    # ============ Training loop ============
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        logger.info(f"  Resuming from step {global_step}")
    
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
                # ========== 获取数据 ==========
                # vi_video: (B, 3, T, H, W) - 起点
                # ir_video: (B, 3, T, H, W) - ControlNet hint (1帧复制)
                # gt_video: (B, 3, T, H, W) - 终点
                vi_video = batch['vi_video'].to(accelerator.device, dtype=torch.float32)
                ir_video = batch['ir_video'].to(accelerator.device, dtype=torch.float32)
                gt_video = batch['gt_video'].to(accelerator.device, dtype=torch.float32)
                
                B, _, T, H, W = vi_video.shape
                
                with torch.no_grad():
                    # VAE 编码 VI (起点)
                    vi_video_perm = vi_video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                    vi_latent = model.vae_encode(vi_video_perm)  # (B, 4, T, H/8, W/8)
                    vi_latent = vi_latent.float()
                    
                    # VAE 编码 IR (ControlNet hint)
                    ir_video_perm = ir_video.permute(0, 2, 1, 3, 4)
                    ir_latent = model.vae_encode(ir_video_perm)
                    ir_latent = ir_latent.float()
                    
                    # VAE 编码 GT (终点)
                    gt_video_perm = gt_video.permute(0, 2, 1, 3, 4)
                    gt_latent = model.vae_encode(gt_video_perm)
                    gt_latent = gt_latent.float()
                
                # ========== Flow Matching 训练 (VI → GT) ==========
                
                # 轨迹 velocity: gt_latent - vi_latent
                velocity = gt_latent - vi_latent
                
                # 随机采样 timestep: t ∈ [0, 1]
                t = torch.rand(B, device=accelerator.device)
                t_expand = t.view(B, 1, 1, 1, 1)
                
                # 插值: x_t = (1-t) * vi_latent + t * gt_latent
                x_t = (1 - t_expand) * vi_latent + t_expand * gt_latent
                
                # 前向传播: 预测 velocity
                with accelerator.autocast():
                    pred_velocity = model.forward(
                        x_t=x_t,
                        t=t,
                        ir_latent=ir_latent,  # IR 作为 ControlNet hint
                    )
                
                # ========== 计算损失 ==========
                pred_velocity = pred_velocity.float()
                
                # 1. Velocity MSE Loss (主损失)
                loss_velocity = F.mse_loss(pred_velocity, velocity)
                
                # 2. 从 velocity 恢复预测的 x0，用于计算颜色损失
                pred_x0 = x_t + (1 - t_expand) * pred_velocity
                
                # 3. 颜色均值匹配损失 (轻量级)
                loss_color = color_mean_loss(pred_x0, gt_latent)
                
                # 总损失
                loss = loss_velocity + args.color_loss_weight * loss_color
                
                # 检查 nan
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[Warning] Step {global_step}: loss nan/inf, skipping...")
                    optimizer.zero_grad()
                    continue
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 更新进度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                
                postfix_logs = {
                    "loss": f"{loss.item():.4f}",
                }
                
                progress_bar.set_postfix(**postfix_logs)
                accelerator.log(logs, step=global_step)
                
                # 保存 checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                        
                        # 保存验证结果
                        try:
                            print(f"\n正在生成验证图片 (VI → Fusion)...")
                            vi_video_val = vi_video.detach()
                            ir_video_val = ir_video.detach()
                            gt_video_val = gt_video.detach()
                            
                            pred_video_val = run_validation(
                                model=model,
                                noise_scheduler=noise_scheduler,
                                vi_video=vi_video_val,
                                ir_video=ir_video_val,
                                device=accelerator.device,
                                weight_dtype=weight_dtype,
                                num_steps=args.num_inference_steps,
                                time_shift=args.time_shift,
                            )
                            
                            save_validation_results(
                                vi_video=vi_video_val,
                                ir_video=ir_video_val,
                                gt_video=gt_video_val,
                                pred_video=pred_video_val,
                                save_dir=save_path,
                                step=global_step
                            )
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
            
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
        
        epoch += 1
    
    # 训练结束保存最终 checkpoint
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        if not os.path.exists(final_save_path):
            accelerator.save_state(final_save_path)
            logger.info(f"Final checkpoint saved to {final_save_path}")
            
            try:
                print(f"\n正在生成最终验证图片 (VI → Fusion)...")
                pred_video_final = run_validation(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    vi_video=vi_video.detach(),
                    ir_video=ir_video.detach(),
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    num_steps=args.num_inference_steps,
                    time_shift=args.time_shift,
                )
                save_validation_results(
                    vi_video=vi_video.detach(),
                    ir_video=ir_video.detach(),
                    gt_video=gt_video.detach(),
                    pred_video=pred_video_final,
                    save_dir=final_save_path,
                    step=global_step
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
    
    accelerator.end_training()
    logger.info("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
使用示例 (VI + IR → Fusion):

目录结构 (三个目录子文件夹名相同):
    vi_data_dir/       ir_data_dir/       gt_data_dir/
      video_001/         video_001/         video_001/
        0000.png           0000.png           0000.png
        0001.png           0001.png           0001.png
        ...                ...                ...

# 1. 完整训练 (ControlNet + UNet 都训练)
python Fusion/train_latent.py \
    --pretrained_model_path pretrained_weight/model.pt \
    --vi_data_dir Fusion/dataset/train/visible \
    --ir_data_dir Fusion/dataset/train/infrared \
    --gt_data_dir Fusion/dataset/train/fusion \
    --output_dir outputs/vi_to_fusion \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --max_train_steps 10000 \
    --checkpointing_steps 1000

# 2. 只训练 UNet (冻结 ControlNet)
python Fusion/train_latent.py \
    --pretrained_model_path pretrained_weight/model.pt \
    --vi_data_dir Fusion/dataset/train/visible \
    --ir_data_dir Fusion/dataset/train/infrared \
    --gt_data_dir Fusion/dataset/train/fusion \
    --output_dir outputs/unet_only \
    --freeze_controlnet

# 3. 使用本地 VAE 权重
python Fusion/train_latent.py \
    --pretrained_model_path pretrained_weight/model.pt \
    --vae_path E:/models/stable-video-diffusion/vae \
    --vi_data_dir Fusion/dataset/train/visible \
    --ir_data_dir Fusion/dataset/train/infrared \
    --gt_data_dir Fusion/dataset/train/fusion \
    --output_dir outputs/vi_to_fusion

新架构说明:
==================
VI (4帧) → VAE → vi_latent (起点)
                    ↓
GT (4帧) → VAE → gt_latent (终点)
                    ↓
            velocity = gt_latent - vi_latent
                    ↓
            x_t = (1-t) * vi_latent + t * gt_latent
                    ↓
IR (1帧→复制4帧) → VAE → ir_latent → ControlNet → 控制信号
                    ↓
            x_t + 控制信号 → UNet → pred_velocity
                    ↓
            Loss = MSE(pred_velocity, velocity)

推理时:
    从 vi_latent 开始，积分 velocity → 融合结果 (不是从纯噪声开始!)
"""

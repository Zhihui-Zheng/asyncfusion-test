#!/usr/bin/env python
# coding=utf-8
"""
DPO (Direct Preference Optimization) 训练脚本 - Y通道 + TVT 4x VAE 版
用于视频融合模型的偏好学习优化

阶段2训练：在 train_latent_vae331 产生的 checkpoint 上继续 DPO 偏好对齐。

冻结策略（与 train_latent_vae331 不同）：
  - ControlNet: 冻结（只提供空间条件）
  - UNet 时序模块: 可训练（DPO 梯度修正"热斑固定住"的时序行为）
  - UNet 空间模块 + HFP: 可训练

DPO 数据集设计：
  - Good: IR_t=4 + VI_t=4  → IR 与 VI 同一时刻，热斑位置对齐（正确）
  - Bad:  IR_t=0 + VI_t=4  → IR 来自早期帧，热斑滞后于 VI（错误）
  教模型："红外应该跟随可见光当前位置，而不是固定在原位"

核心思想:
=========
用两组不同质量的融合结果 (good vs bad) 来优化模型：
- Chosen (好样本): Fusion_good 目录下的融合结果
- Rejected (坏样本): Fusion_bad 目录下的融合结果

特殊设计:
- 用户数据中前3帧 good 和 bad 是一样的，只有最后一帧不同
- DPO loss 只关注最后一帧的差异

DPO Loss for Flow Matching:
===========================
1. 计算模型对 chosen 样本的重建误差: loss_good
2. 计算模型对 rejected 样本的重建误差: loss_bad  
3. DPO Loss = -log_sigmoid(β * (loss_bad - loss_good))
   让模型更倾向于生成好样本 (loss_good < loss_bad)

数据结构:
=========
Fusion_good/0111_1716/
    group00_frame_0003.png  (跳过)
    group01_frame_0039.png  → frame 0
    group02_frame_0076.png  → frame 1
    group03_frame_0113.png  → frame 2
    group04_frame_0150.png  → frame 3 (最后一帧，DPO关注点)

Fusion_bad/0111_1716/
    group00_frame_0003.png  (跳过)
    group01_frame_0039.png  → frame 0 (与good相同)
    group02_frame_0076.png  → frame 1 (与good相同)
    group03_frame_0113.png  → frame 2 (与good相同)
    group04_frame_0150.png  → frame 3 (与good不同！DPO关注点)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import re
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

from Fusion5.dual_modal_fusion_model import VideoControlNetModel
from Fusion4.flow_match import FlowMatchDiscreteScheduler

import cv2
import numpy as np
import types
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# TVT VAED4 (预训练的 4x VAE)
tvt_path = os.path.join(base_path, 'TVT', 'TVT-main', 'TVT')
sys.path.insert(0, tvt_path)
from VAED4.autoencoder_kl import AutoencoderKL as VAED4AutoencoderKL

check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


# ============ TVT VAED4 加载 (预训练的 4x VAE) ============

def load_vaed4(vae_ckpt_path):
    """加载 TVT 论文的预训练 VAED4 (4x 下采样 VAE)"""
    print("=" * 60)
    print("加载 TVT VAED4 (预训练 4x VAE)")
    print("=" * 60)
    config_path = os.path.join(tvt_path, 'VAED4', 'config2.json')
    vae = VAED4AutoencoderKL.from_config(VAED4AutoencoderKL.load_config(config_path))
    vae_ckp = torch.load(vae_ckpt_path, map_location='cpu', weights_only=False)
    ckp_dict = {n.replace('vaeDe4.', ''): v
                for n, v in vae_ckp['state_dict'].items() if 'vaeDe4' in n}
    vae.load_state_dict(ckp_dict, strict=True)
    print(f"  参数量: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M")
    print(f"  下采样: 4x (256×256 → 64×64 latent)")
    print("=" * 60)
    del vae_ckp
    torch.cuda.empty_cache()
    return vae


def replace_vae_to_4x(model, vae_ckpt_path):
    """将模型的 8x VAE 替换为 TVT VAED4 (预训练 4x VAE)"""
    vae_4x = load_vaed4(vae_ckpt_path)
    vae_4x.eval()
    vae_4x.requires_grad_(False)
    vae_4x.to(model.device, dtype=torch.float32)
    del model.vae
    torch.cuda.empty_cache()
    model.vae = vae_4x

    def vae_encode_4x(self, video, chunk_size=1):
        num_f = video.shape[1]
        video = rearrange(video, "b f c h w -> (b f) c h w").to(dtype=torch.float32)
        z_list = [self.vae.encode(video[i:i+chunk_size]).latent_dist.sample()
                  for i in range(0, video.shape[0], chunk_size)]
        z = torch.cat(z_list, dim=0)
        return rearrange(z, "(b f) c h w -> b c f h w", f=num_f) * self.vae.config.scaling_factor

    def vae_decode_4x(self, z, num_f):
        return self.vae.decode((z / self.vae.config.scaling_factor).to(torch.float32)).sample

    def vae_decode_chunk_4x(self, z, chunk_size=3):
        z = rearrange(z, "b c f h w -> (b f) c h w").to(torch.float32)
        out = [self.vae_decode(z[i:i+chunk_size], z[i:i+chunk_size].shape[0])
               for i in range(0, z.shape[0], chunk_size)]
        return torch.cat(out)

    model.vae_encode = types.MethodType(vae_encode_4x, model)
    model.vae_decode = types.MethodType(vae_decode_4x, model)
    model.vae_decode_chunk = types.MethodType(vae_decode_chunk_4x, model)
    print("  ✓ VAE 已替换为 TVT VAED4 (4x)")


# ============ 图片转灰度3通道 (Y通道) ============

def bgr_to_gray_3ch(img_bgr):
    """BGR图片 → 灰度(Y通道) → 复制为3通道 (用于VAE输入)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H, W)
    return np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)


# ============ Temporal Metrics (from metrics.py, 可微分版本) ============

class FeatureExtractor(torch.nn.Module):
    """用于提取特征的 ResNet (冻结参数，用于计算 reward)"""
    _instance = None
    
    @classmethod
    def get_instance(cls, device='cuda'):
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """x: (B, C, H, W), 范围 [-1, 1] → 返回 (B, 512) 特征"""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.model(x).squeeze(-1).squeeze(-1)


def compute_feaCD_loss(pred_video, target_vi, target_ir, feature_extractor):
    """
    特征运动一致性 Loss (可微分，直接优化)
    
    让预测视频的运动方向与 VI/IR 一致
    
    Args:
        pred_video: (B, C, T, H, W) 预测视频，范围 [-1, 1]
        target_vi: (B, C, T, H, W) 可见光，范围 [-1, 1]
        target_ir: (B, C, T, H, W) 红外，范围 [-1, 1]
    
    Returns:
        loss: 标量，越小越好
    """
    B, C, T, H, W = pred_video.shape
    if T < 2:
        return torch.tensor(0.0, device=pred_video.device, requires_grad=True)
    
    total_loss = 0.0
    count = 0
    
    for t in range(1, T):
        # 提取特征 (feature_extractor 冻结，但 pred_video 有梯度)
        feat_pred_curr = feature_extractor(pred_video[:, :, t])
        feat_pred_prev = feature_extractor(pred_video[:, :, t-1])
        
        with torch.no_grad():
            feat_vi_curr = feature_extractor(target_vi[:, :, t])
            feat_vi_prev = feature_extractor(target_vi[:, :, t-1])
            feat_ir_curr = feature_extractor(target_ir[:, :, t])
            feat_ir_prev = feature_extractor(target_ir[:, :, t-1])
        
        # 计算运动方向
        motion_pred = feat_pred_curr - feat_pred_prev
        motion_vi = feat_vi_curr - feat_vi_prev
        motion_ir = feat_ir_curr - feat_ir_prev
        
        # Loss = 1 - cos_sim (越小越好)
        cos_sim_vi = F.cosine_similarity(motion_pred, motion_vi, dim=1).mean()
        cos_sim_ir = F.cosine_similarity(motion_pred, motion_ir, dim=1).mean()
        
        loss_vi = 1 - cos_sim_vi
        loss_ir = 1 - cos_sim_ir
        
        total_loss = total_loss + (loss_vi + loss_ir) / 2
        count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_video.device, requires_grad=True)


def compute_flowD_loss(pred_video, target_video):
    """
    光流一致性 Loss (简化版，可微分)
    
    让预测视频的帧间运动与目标一致
    
    Args:
        pred_video: (B, C, T, H, W) 预测视频，范围 [-1, 1]
        target_video: (B, C, T, H, W) 目标视频，范围 [-1, 1]
    
    Returns:
        loss: 标量，越小越好
    """
    B, C, T, H, W = pred_video.shape
    if T < 2:
        return torch.tensor(0.0, device=pred_video.device, requires_grad=True)
    
    total_loss = 0.0
    count = 0
    
    for t in range(1, T):
        # 帧差 (运动表示)
        pred_diff = pred_video[:, :, t] - pred_video[:, :, t-1]
        target_diff = target_video[:, :, t] - target_video[:, :, t-1]
        
        # L1 loss
        loss = F.l1_loss(pred_diff, target_diff, reduction='mean')
        
        total_loss = total_loss + loss
        count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_video.device, requires_grad=True)


# ============ DPO Loss Function ============

def dpo_loss(
    loss_chosen: torch.FloatTensor,
    loss_rejected: torch.FloatTensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = True,
) -> torch.FloatTensor:
    """
    计算 DPO loss (Direct Preference Optimization)
    
    对于 Diffusion/Flow Matching 模型，我们使用重建误差作为隐式 reward:
    - reward = -loss (loss 越小，reward 越高)
    - 所以 reward_chosen - reward_rejected = loss_rejected - loss_chosen
    
    Args:
        loss_chosen: 模型对好样本的重建误差
        loss_rejected: 模型对坏样本的重建误差
        beta: DPO 温度参数 (控制偏好强度)
        label_smoothing: 标签平滑
        reference_free: 是否使用 reference-free DPO
    
    Returns:
        DPO loss
    """
    # logits = reward_chosen - reward_rejected
    #        = (-loss_chosen) - (-loss_rejected)
    #        = loss_rejected - loss_chosen
    logits = loss_rejected - loss_chosen
    
    # DPO loss: -log_sigmoid(β * logits)
    # 当 logits > 0 (即 loss_rejected > loss_chosen，模型更擅长预测好样本) 时，loss 较小
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    
    # 计算 reward (用于监控)
    chosen_rewards = -loss_chosen.detach()
    rejected_rewards = -loss_rejected.detach()
    
    return losses, chosen_rewards, rejected_rewards


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="8-frame DPO Training for Video Fusion Model")
    
    # Model
    # 仅用于保持接口兼容, DPO 阶段不会再用这里的权重初始化
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="",
        help="(DPO 阶段已弃用) 原始预训练权重路径, 当前不会使用",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid",
        help="SVD 8x VAE 路径 (use_tvt_vae=True 时忽略)",
    )
    parser.add_argument(
        "--resume_checkpoint_dir",
        type=str,
        default=r"E:\paper2\codebase\STAR\outputs\latent_20260408_103729\checkpoint-2000",
        help="Stage-1 latent 训练产生的 accelerator checkpoint 目录 (model.safetensors=ControlNet + model_1.safetensors=UNet)",
    )

    # TVT VAED4 (4x VAE)
    parser.add_argument("--use_tvt_vae", action=argparse.BooleanOptionalAction, default=True,
                        help="将 SVD 8x VAE 替换为 TVT VAED4 4x VAE（默认启用）")
    parser.add_argument("--vae_4x_ckpt", type=str,
                        default=r"pretrained_weight/vae_4x_tvt.ckpt",
                        help="TVT VAED4 权重路径")

    # HFP
    parser.add_argument("--use_hfp", action=argparse.BooleanOptionalAction, default=True,
                        help="启用 HFP 模块（需与 Stage-1 保持一致）")
    parser.add_argument("--hfp_block_indices", type=int, nargs="+", default=[1, 4, 7])
    parser.add_argument("--hfp_ratio", type=float, nargs=2, default=[0.25, 0.25])
    parser.add_argument("--hfp_patch", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--hfp_no_dct", action="store_true", default=False)
    parser.add_argument("--hfp_serial", action="store_true", default=False)
    parser.add_argument("--hfp_init", type=str, default="zeros",
                        choices=["ones", "zeros", "default"])
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=r"E:\paper2\codebase\STAR\outputs\temporal8f_step_20260509_115806\checkpoint-6000\model_1.safetensors",
        help="从单独的 UNet checkpoint 继续 DPO 训练 (例如上一轮 DPO 的 model_1.safetensors)",
    )
    
    # Freeze strategy
    parser.add_argument(
        "--freeze_controlnet",
        action="store_true",
        default=True,
        help="Freeze ControlNet (DPO 阶段只训练 UNet)",
    )
    parser.add_argument(
        "--freeze_unet",
        action="store_true",
        help="Freeze UNet",
    )
    
    # Output — 按启动时间自动生成目录名
    from datetime import datetime as _dt
    _default_dpo_dir = f"outputs/dpo_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_default_dpo_dir,
        help="输出目录（默认按训练启动时间自动生成）",
    )

    # Training
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Learning rate
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Data - DPO preference pairs
    parser.add_argument("--vi_data_dir", type=str, 
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD_cropped\visible1", 
                        help="可见光输入目录 (起点)")
    parser.add_argument("--ir_data_dir", type=str, 
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD_cropped\infrared", 
                        help="红外输入目录 (ControlNet hint)")
    parser.add_argument("--chosen_data_dir", type=str, 
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD_cropped\Fusion_good", 
                        help="好样本目录 (chosen)")
    parser.add_argument("--rejected_data_dir", type=str, 
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD_cropped\Fusion_bad", 
                        help="坏样本目录 (rejected)")
    
    parser.add_argument("--num_frames", type=int, default=8,
                        help="训练帧数，默认连续 8 帧")
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    
    # DPO specific
    parser.add_argument("--beta", type=float, default=1.0,
                        help="DPO 温度参数 β (控制偏好强度，越大梯度越强，建议 0.5~2.0)")
    parser.add_argument("--sft_weight", type=float, default=0.3,
                        help="SFT loss (前 T-1 帧监督) 权重，建议 0.2~0.5")
    parser.add_argument("--anchor_weight", type=float, default=0.5,
                        help="最后一帧 chosen-anchor MSE 权重，防止 DPO 把 latent 推出分布，建议 0.5~2.0")
    
    # Temporal Consistency Loss (直接作为 loss 优化)
    parser.add_argument("--use_temporal_loss", action="store_true", default=False,
                        help="使用时序一致性 loss (feaCD + flowD)")
    # Flow Matching
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--time_shift", type=float, default=13.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    
    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=750, help="权重保存间隔")
    parser.add_argument("--validation_steps", type=int, default=10, help="验证图片生成间隔")
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


class DPOPreferenceDataset(torch.utils.data.Dataset):
    """
    DPO 偏好数据集
    
    加载 chosen (好样本) 和 rejected (坏样本) 配对
    
    数据结构:
    - vi_dir: 可见光输入 (起点)
    - ir_dir: 红外输入 (ControlNet hint)
    - chosen_dir: 好样本 (Fusion_good)
    - rejected_dir: 坏样本 (Fusion_bad)
    
    配对逻辑 (按 group 加载连续帧):
    - 每个 group 内取连续 num_frames 帧 (默认 8 帧)
    - 跳过 group00，随机选择一个 group (group01-04)
    - 取该 group 的前 num_frames 帧
    - VI/IR 也取对应帧号的文件
    """
    def __init__(self, vi_dir, ir_dir, chosen_dir, rejected_dir, 
                 num_frames=8, video_size=(256, 256)):
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.chosen_dir = chosen_dir
        self.rejected_dir = rejected_dir
        self.num_frames = num_frames
        self.video_size = video_size
        
        # 查找配对数据
        self.samples = self._find_samples()
        
        print(f"=" * 60)
        print(f"DPO Preference Dataset (按 Group 加载连续帧)")
        print(f"=" * 60)
        print(f"Found {len(self.samples)} preference pairs")
        print(f"  VI (可见光): {vi_dir}")
        print(f"  IR (红外):   {ir_dir}")
        print(f"  Chosen:      {chosen_dir}")
        print(f"  Rejected:    {rejected_dir}")
        print(f"  帧数: {num_frames}")
        print(f"=" * 60)
    
    def _extract_group_and_frame(self, filename):
        """
        从文件名中提取 group 号和帧号
        例如: group01_frame_0039.png → ("group01", "frame_0039")
        """
        match = re.search(r"(group\d+)_frame_(\d+)", filename, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower(), int(match.group(2))
        return None, None

    def _extract_frame_index(self, filename):
        """
        从文件名提取帧序号，优先匹配 frame_XXXX，其次回退到末尾数字。
        """
        m = re.search(r"frame[_-]?(\d+)", filename, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)(?!.*\d)", Path(filename).stem)
        if m:
            return int(m.group(1))
        return None
    
    def _group_files_by_group(self, dir_path):
        """
        将目录中的文件按 group 分组
        返回: {"group00": [file1, file2, ...], "group01": [...], ...}
        """
        if not dir_path or not dir_path.exists():
            return {}
        
        all_files = sorted(list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg')))
        groups = {}
        
        for f in all_files:
            group_name, frame_num = self._extract_group_and_frame(f.name)
            if group_name:
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append((f, frame_num))
        
        # 按帧号排序每个 group 内的文件
        for group_name in groups:
            groups[group_name] = sorted(groups[group_name], key=lambda x: x[1])
        
        return groups
    
    def _find_file_by_frame_number(self, dir_path, frame_number):
        """在目录中查找包含指定帧号的文件"""
        if not dir_path or not dir_path.exists():
            return None

        target_idx = int(frame_number)
        all_files = list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg'))
        for f in all_files:
            idx = self._extract_frame_index(f.name)
            if idx is not None and idx == target_idx:
                return f
        return None

    def _build_windows(self, frame_indices):
        """
        按 num_frames 组织窗口；默认非重叠分组，保留尾部补齐窗口。
        """
        frame_indices = sorted(frame_indices)
        if len(frame_indices) < self.num_frames:
            return []

        starts = list(range(0, len(frame_indices) - self.num_frames + 1, self.num_frames))
        if starts and starts[-1] != len(frame_indices) - self.num_frames:
            starts.append(len(frame_indices) - self.num_frames)
        return [frame_indices[s:s + self.num_frames] for s in starts]
    
    def _find_samples(self):
        """
        查找配对的数据样本
        
        优先检查 Fusion_good 和 Fusion_bad 是否都存在，
        只有当两者都存在时，才去找 visible 和 infrared
        """
        samples = []
        chosen_root = Path(self.chosen_dir)
        rejected_root = Path(self.rejected_dir)
        
        if not chosen_root.exists():
            print(f"Warning: Chosen dir {chosen_root} does not exist")
            return samples
        
        if not rejected_root.exists():
            print(f"Warning: Rejected dir {rejected_root} does not exist")
            return samples
        
        skipped_no_rejected = 0
        skipped_no_vi = 0
        skipped_no_ir = 0
        skipped_no_group = 0
        
        for chosen_sample_dir in sorted(chosen_root.iterdir()):
            if chosen_sample_dir.is_dir():
                name = chosen_sample_dir.name
                rejected_sample_dir = rejected_root / name
                
                # 1. 首先检查 Fusion_bad 是否存在
                if not rejected_sample_dir.exists():
                    skipped_no_rejected += 1
                    continue
                
                # 2. 按 group 分组文件，检查两边是否都有
                chosen_groups = self._group_files_by_group(chosen_sample_dir)
                rejected_groups = self._group_files_by_group(rejected_sample_dir)
                
                # 3. 只有 Fusion_good/Fusion_bad 都存在，才检查 visible 和 infrared
                vi_sample_dir = Path(self.vi_dir) / name if self.vi_dir else None
                ir_sample_dir = Path(self.ir_dir) / name if self.ir_dir else None
                
                # 检查 visible 是否存在
                if vi_sample_dir and not vi_sample_dir.exists():
                    skipped_no_vi += 1
                    continue
                
                # 检查 infrared 是否存在
                if ir_sample_dir and not ir_sample_dir.exists():
                    skipped_no_ir += 1
                    continue

                # 严格按单 group 内组窗，禁止跨 group 合并
                # 跨 group 合并会把不同时间段的帧拼在一起，不是真正连续的 8 帧
                sample_windows = []
                common_groups = sorted(set(chosen_groups.keys()) & set(rejected_groups.keys()))
                max_common_frames_per_group = 0
                group_frame_info = []
                for group_name in common_groups:
                    if group_name == "group00":
                        continue
                    chosen_idx = [idx for (_, idx) in chosen_groups[group_name]]
                    rejected_idx = [idx for (_, idx) in rejected_groups[group_name]]
                    common_idx = sorted(set(chosen_idx) & set(rejected_idx))
                    max_common_frames_per_group = max(max_common_frames_per_group, len(common_idx))
                    group_frame_info.append(f"{group_name}:{len(common_idx)}帧")
                    windows = self._build_windows(common_idx)
                    sample_windows.extend(windows)

                if not sample_windows:
                    skipped_no_group += 1
                    frame_detail = ", ".join(group_frame_info) if group_frame_info else "无公共 group"
                    print(
                        f"  [跳过] {name}: 单 group 内公共帧不足 {self.num_frames} 帧 "
                        f"（{frame_detail}）。"
                        f"跨 group 合并已禁用（会引入不连续帧），请确认数据集帧数。"
                    )
                    continue

                for frame_numbers in sample_windows:
                    samples.append({
                        'name': name,
                        'chosen_dir': chosen_sample_dir,
                        'rejected_dir': rejected_sample_dir,
                        'vi_dir': vi_sample_dir,
                        'ir_dir': ir_sample_dir,
                        'frame_numbers': frame_numbers,
                    })
        
        print(f"  总共找到 {len(samples)} 个训练片段")
        if skipped_no_rejected > 0:
            print(f"  跳过 {skipped_no_rejected} 个样本 (Fusion_bad 不存在)")
        if skipped_no_group > 0:
            print(f"  跳过 {skipped_no_group} 个样本 (单 group 帧不足 {self.num_frames} 帧，拒绝跨 group 拼接)")
        if skipped_no_vi > 0:
            print(f"  跳过 {skipped_no_vi} 个样本 (visible 不存在)")
        if skipped_no_ir > 0:
            print(f"  跳过 {skipped_no_ir} 个样本 (infrared 不存在)")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回:
        - vi_video: (3, T, H, W) VI视频 (起点)
        - ir_video: (3, T, H, W) IR视频 (ControlNet hint)
        - chosen_video: (3, T, H, W) 好样本
        - rejected_video: (3, T, H, W) 坏样本
        """
        if len(self.samples) == 0:
            raise RuntimeError("DPOPreferenceDataset 为空，请检查数据路径和文件命名。")
        
        sample = self.samples[idx % len(self.samples)]
        frame_numbers = sample['frame_numbers']
        
        # 加载 chosen 和 rejected - 按帧号加载
        chosen_video = self._load_frames_by_frame_numbers(sample['chosen_dir'], frame_numbers)
        rejected_video = self._load_frames_by_frame_numbers(sample['rejected_dir'], frame_numbers)
        if chosen_video is None:
            chosen_video = torch.zeros(3, self.num_frames, *self.video_size)
        if rejected_video is None:
            rejected_video = torch.zeros(3, self.num_frames, *self.video_size)
        
        # 加载 VI - 通过帧号匹配
        vi_video = self._load_frames_by_frame_numbers(sample['vi_dir'], frame_numbers)
        if vi_video is None:
            vi_video = torch.zeros_like(chosen_video)
        
        # 加载 IR - 取第一帧对应的帧号，复制到 T 帧
        ir_video = self._load_ir_by_frame_number(sample['ir_dir'], frame_numbers[0])
        if ir_video is None:
            ir_video = torch.zeros_like(chosen_video)
        
        return {
            'vi_video': vi_video,
            'ir_video': ir_video,
            'chosen_video': chosen_video,
            'rejected_video': rejected_video,
        }
    
    def _load_group_frames(self, dir_path, group_name, frame_numbers):
        """加载指定 group 的帧"""
        groups = self._group_files_by_group(dir_path)
        
        if group_name not in groups:
            return torch.zeros(3, self.num_frames, *self.video_size)
        
        group_files = groups[group_name][:self.num_frames]
        
        frames = []
        for (file_path, _) in group_files:
            img = cv2.imread(str(file_path))
            img = bgr_to_gray_3ch(img)  # 灰度(Y通道) → 3ch
            img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
            img = img.astype(np.float32) / 255.0
            img = img * 2 - 1
            frames.append(img)
        
        # 如果不够，重复最后一帧
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        
        frames = np.stack(frames, axis=0)
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        return torch.from_numpy(frames).float()
    
    def _load_frames_by_frame_numbers(self, dir_path, frame_numbers):
        """通过帧号列表加载对应的帧"""
        if not dir_path or not dir_path.exists():
            return None
        
        frames = []
        for fn in frame_numbers:
            file_path = self._find_file_by_frame_number(dir_path, fn)
            if file_path is None:
                return None
            
            img = cv2.imread(str(file_path))
            if img is None:
                return None
            img = bgr_to_gray_3ch(img)  # 灰度(Y通道) → 3ch
            img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
            img = img.astype(np.float32) / 255.0
            img = img * 2 - 1
            frames.append(img)
        
        frames = np.stack(frames, axis=0)
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        return torch.from_numpy(frames).float()
    
    def _load_ir_by_frame_number(self, dir_path, frame_number):
        """通过帧号加载 IR 帧，并复制到 T 帧"""
        if not dir_path or not dir_path.exists():
            return None
        
        file_path = self._find_file_by_frame_number(dir_path, frame_number)
        if file_path is None:
            return None
        
        img = cv2.imread(str(file_path))
        if img is None:
            return None
        img = bgr_to_gray_3ch(img)  # 灰度(Y通道) → 3ch
        img = cv2.resize(img, (self.video_size[1], self.video_size[0]))
        img = img.astype(np.float32) / 255.0
        img = img * 2 - 1
        
        img = np.transpose(img, (2, 0, 1))
        ir_video = np.stack([img] * self.num_frames, axis=1)
        
        return torch.from_numpy(ir_video).float()


def save_dpo_validation(vi_video, ir_video, chosen_video, rejected_video, pred_video, 
                        save_dir, step, loss_info=None):
    """
    保存 DPO 验证结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    T = pred_video.shape[2]
    num_frames_to_show = min(T, 4)
    indices = np.linspace(0, T - 1, num_frames_to_show).astype(int)
    
    # 5列: VI, IR, Chosen, Rejected, Pred
    fig, axes = plt.subplots(num_frames_to_show, 5, figsize=(18, num_frames_to_show * 3))
    
    if num_frames_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        def _to_gray2d(tensor_chw, normalize_from_neg1=True):
            """(C,H,W) tensor → (H,W) float [0,1]，取均值通道"""
            arr = tensor_chw.float().cpu().numpy().mean(axis=0)
            if normalize_from_neg1:
                arr = (arr + 1) / 2
            return np.clip(arr, 0, 1)

        # VI
        axes[row, 0].imshow(_to_gray2d(vi_video[0, :, idx]), cmap='gray', vmin=0, vmax=1)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title('VI (起点)', fontsize=12, fontweight='bold')

        # IR
        if row == 0:
            axes[row, 1].imshow(_to_gray2d(ir_video[0, :, 0]), cmap='gray', vmin=0, vmax=1)
        else:
            axes[row, 1].imshow(np.ones((vi_video.shape[3], vi_video.shape[4])) * 0.3,
                                cmap='gray', vmin=0, vmax=1)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title('IR (hint)', fontsize=12, fontweight='bold')

        # Chosen (好样本)
        axes[row, 2].imshow(_to_gray2d(chosen_video[0, :, idx]), cmap='gray', vmin=0, vmax=1)
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].set_title('Chosen ✓', fontsize=12, fontweight='bold', color='green')

        # Rejected (坏样本)
        axes[row, 3].imshow(_to_gray2d(rejected_video[0, :, idx]), cmap='gray', vmin=0, vmax=1)
        axes[row, 3].axis('off')
        if row == 0:
            axes[row, 3].set_title('Rejected ✗', fontsize=12, fontweight='bold', color='red')

        # Pred (模型预测，已在 run_dpo_validation 中归一化到 [0,1])
        pred_frame = pred_video[0, :, idx].float().cpu().numpy().mean(axis=0)
        pred_frame = np.clip(pred_frame, 0, 1)
        axes[row, 4].imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
        axes[row, 4].axis('off')
        if row == 0:
            axes[row, 4].set_title('Pred', fontsize=12, fontweight='bold', color='blue')
        
        # 标注最后一帧 (DPO关注点)
        if idx == T - 1:
            for col in range(5):
                axes[row, col].set_xlabel('★ DPO Focus', fontsize=10, color='orange')
    
    title = f'DPO Step {step}'
    if loss_info:
        title += f' | loss_chosen={loss_info.get("loss_chosen", 0):.4f}, loss_rejected={loss_info.get("loss_rejected", 0):.4f}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"dpo_step{step}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  DPO 验证图片已保存: {save_path}")


@torch.no_grad()
def run_dpo_validation(model, noise_scheduler, vi_video, ir_video, device, weight_dtype,
                       num_steps=20, time_shift=13.0):
    """运行验证推理"""
    model.eval_mode()
    
    B, C, T, H, W = vi_video.shape
    
    vi_video_device = vi_video.to(device)
    ir_video_device = ir_video.to(device)
    
    # VAE 编码
    vi_video_perm = vi_video_device.permute(0, 2, 1, 3, 4)
    vi_latent = model.vae_encode(vi_video_perm)
    
    ir_video_perm = ir_video_device.permute(0, 2, 1, 3, 4)
    ir_latent = model.vae_encode(ir_video_perm)
    
    # Flow Matching 采样
    noise_scheduler.set_timesteps(num_steps, time_shift=time_shift, device=device)
    
    x = vi_latent.clone()
    
    for t_step in noise_scheduler.timesteps:
        t_batch = t_step.unsqueeze(0).repeat(B).to(device)
        v_pred = model.forward(x_t=x, t=t_batch, ir_latent=ir_latent)
        x = noise_scheduler.step(v_pred, t_step, x)
    
    # VAE 解码
    pred_video = model.vae_decode_chunk(x, chunk_size=T)
    pred_video = rearrange(pred_video, '(b t) c h w -> b c t h w', b=B, t=T)
    pred_video = (pred_video + 1) / 2
    pred_video = torch.clamp(pred_video, 0, 1)
    
    model.train_mode()
    return pred_video


def _save_model_weights(accelerator, model, save_path, freeze_controlnet, freeze_unet, logger):
    """
    根据冻结策略，直接保存可训练模型的 safetensors 权重。
    - freeze_controlnet=True  → 只保存 unet.safetensors
    - freeze_unet=True        → 只保存 controlnet.safetensors
    - 都训练                  → 同时保存两者
    """
    from safetensors.torch import save_file

    if not freeze_unet:
        unet_to_save = accelerator.unwrap_model(model.unet)
        unet_out = os.path.join(save_path, "unet.safetensors")
        save_file(unet_to_save.state_dict(), unet_out)
        logger.info(f"  ✓ UNet 权重已保存: {unet_out}")

    if not freeze_controlnet:
        ctrl_to_save = accelerator.unwrap_model(model.video_controlnet)
        ctrl_out = os.path.join(save_path, "controlnet.safetensors")
        save_file(ctrl_to_save.state_dict(), ctrl_out)
        logger.info(f"  ✓ ControlNet 权重已保存: {ctrl_out}")


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
            self.model_path = args.pretrained_model_path
            self.vae_path = args.vae_path

    opt = ModelConfig()
    opt.model_path = None  # 权重在下方分别加载
    model = VideoControlNetModel(
        opt,
        device=accelerator.device,
        use_hfp=args.use_hfp,
        hfp_block_indices=tuple(args.hfp_block_indices),
        hfp_ratio=tuple(args.hfp_ratio),
        hfp_patch=tuple(args.hfp_patch),
        hfp_use_dct=not args.hfp_no_dct,
        hfp_serial=args.hfp_serial,
        hfp_init=args.hfp_init,
    )
    
    # ============ 加载 Stage-1 latent checkpoint (ControlNet + UNet，一起恢复) ============
    if args.resume_checkpoint_dir and os.path.exists(args.resume_checkpoint_dir):
        from safetensors.torch import load_file
        ctrl_path = os.path.join(args.resume_checkpoint_dir, "model.safetensors")
        unet_path = os.path.join(args.resume_checkpoint_dir, "model_1.safetensors")

        if os.path.exists(ctrl_path):
            logger.info(f"Loading ControlNet checkpoint from: {ctrl_path}")
            ctrl_dict = load_file(ctrl_path)
            ctrl_dict = {k.replace('module.', ''): v for k, v in ctrl_dict.items()}
            ret = model.video_controlnet.load_state_dict(ctrl_dict, strict=False)
            logger.info(f"Loaded ControlNet: missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")

        if os.path.exists(unet_path):
            logger.info(f"Loading UNet checkpoint from: {unet_path}")
            unet_dict = load_file(unet_path)
            unet_dict = {k.replace('module.', ''): v for k, v in unet_dict.items()}
            ret = model.unet.load_state_dict(unet_dict, strict=False)
            logger.info(f"Loaded UNet: missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")
    
    # ============ 额外的 UNet checkpoint (仅 UNet，用于从上一轮 DPO 继续训练) ============
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        logger.info(f"Loading UNet checkpoint (DPO resume): {args.resume_checkpoint}")
        if args.resume_checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file
            ckpt = load_file(args.resume_checkpoint)
        else:
            ckpt = torch.load(args.resume_checkpoint, map_location='cpu')
        
        unet_dict = {}
        for k, v in ckpt.items():
            key = k[7:] if k.startswith('module.') else k
            unet_dict[key] = v
        
        ret = model.unet.load_state_dict(unet_dict, strict=False)
        logger.info(f"Loaded UNet (DPO resume): missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")
    
    # ============ TVT VAED4 替换 SVD VAE ============
    if args.use_tvt_vae:
        logger.info("[TVT] 正在替换 SVD 8x VAE → TVT VAED4 4x VAE ...")
        replace_vae_to_4x(model, args.vae_4x_ckpt)
        logger.info("[TVT] ✓ VAE 替换完成")

    # ============ 冻结策略 ============
    # DPO 阶段：ControlNet 冻结，UNet（含时序模块）全部可训练
    # 目的：让 DPO 梯度修正 UNet 时序模块的"热斑固定住"行为
    train_controlnet = not args.freeze_controlnet  # 默认 False（冻结）
    train_unet = not args.freeze_unet              # 默认 True（训练）

    model.freeze_for_training(
        train_controlnet=train_controlnet,
        train_unet=train_unet,
        freeze_unet_temporal=False,  # DPO 阶段不冻结时序模块，让 DPO 修正时序行为
    )

    if accelerator.is_main_process:
        trainable_count = sum(p.numel() for p in model.get_trainable_parameters())
        logger.info(f"Trainable parameters: {trainable_count / 1e6:.2f}M")
        logger.info("  ControlNet: 冻结")
        logger.info("  UNet (含时序模块): 可训练 → DPO 梯度修正热斑时序行为")
    
    noise_scheduler = model.noise_scheduler
    
    # ============ Feature Extractor for Temporal Loss ============
    feature_extractor = None
    if args.use_temporal_loss:
        logger.info("Initializing feature extractor for temporal loss (feaCD)...")
        feature_extractor = FeatureExtractor.get_instance(device=accelerator.device)
    
    # ============ Dataset ============
    train_dataset = DPOPreferenceDataset(
        vi_dir=args.vi_data_dir,
        ir_dir=args.ir_data_dir,
        chosen_dir=args.chosen_data_dir,
        rejected_dir=args.rejected_data_dir,
        num_frames=args.num_frames,
        video_size=tuple(args.video_size)
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "训练数据为空：未找到可用的 DPO 配对片段。"
            "请检查 chosen/rejected/visible/infrared 目录及文件命名是否一致。"
        )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # ============ Optimizer ============
    trainable_params = model.get_trainable_parameters()
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    
    # ============ LR Scheduler ============
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=200 * accelerator.num_processes,
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
    
    # ============ Resume ============
    global_step = 0
    
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if dirs else None
        
        if path and os.path.isdir(path):
            logger.info(f"Resuming from checkpoint: {path}")
            accelerator.load_state(path)
            global_step = int(os.path.basename(path).split("-")[1])
    
    # ============ Training loop ============
    logger.info("***** Running DPO Training *****")
    logger.info(f"  Num preference pairs = {len(train_dataset)}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num frames = {args.num_frames}")
    logger.info(f"  DPO β = {args.beta}")
    logger.info(f"  SFT weight = {args.sft_weight}")
    logger.info(f"  Anchor weight = {args.anchor_weight}")
    logger.info(f"  Use temporal loss = {args.use_temporal_loss}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  验证图片间隔 = {args.validation_steps} steps")
    logger.info(f"  权重保存间隔 = {args.checkpointing_steps} steps")
    
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        total=args.max_train_steps,
        desc="DPO Steps",
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
                vi_video = batch['vi_video'].to(accelerator.device, dtype=torch.float32)
                ir_video = batch['ir_video'].to(accelerator.device, dtype=torch.float32)
                chosen_video = batch['chosen_video'].to(accelerator.device, dtype=torch.float32)
                rejected_video = batch['rejected_video'].to(accelerator.device, dtype=torch.float32)
                
                B, _, T, H, W = vi_video.shape
                
                with torch.no_grad():
                    # VAE 编码
                    vi_video_perm = vi_video.permute(0, 2, 1, 3, 4)
                    vi_latent = model.vae_encode(vi_video_perm).float()
                    
                    ir_video_perm = ir_video.permute(0, 2, 1, 3, 4)
                    ir_latent = model.vae_encode(ir_video_perm).float()
                    
                    chosen_video_perm = chosen_video.permute(0, 2, 1, 3, 4)
                    chosen_latent = model.vae_encode(chosen_video_perm).float()
                    
                    rejected_video_perm = rejected_video.permute(0, 2, 1, 3, 4)
                    rejected_latent = model.vae_encode(rejected_video_perm).float()
                
                # ========== Flow Matching 训练 ==========
                
                # Velocity targets
                velocity_chosen = chosen_latent - vi_latent
                velocity_rejected = rejected_latent - vi_latent
                
                # 随机采样 timestep
                t = torch.rand(B, device=accelerator.device)
                t_expand = t.view(B, 1, 1, 1, 1)
                
                # 插值 (对于 chosen 和 rejected)
                x_t_chosen = (1 - t_expand) * vi_latent + t_expand * chosen_latent
                x_t_rejected = (1 - t_expand) * vi_latent + t_expand * rejected_latent
                
                # 前向传播
                with accelerator.autocast():
                    # 对 chosen 预测 velocity
                    pred_velocity_chosen = model.forward(
                        x_t=x_t_chosen,
                        t=t,
                        ir_latent=ir_latent,
                    )
                    
                    # 对 rejected 预测 velocity
                    pred_velocity_rejected = model.forward(
                        x_t=x_t_rejected,
                        t=t,
                        ir_latent=ir_latent,
                    )
                
                pred_velocity_chosen = pred_velocity_chosen.float()
                pred_velocity_rejected = pred_velocity_rejected.float()
                
                # ========== 计算 Loss ==========
                # 策略:
                #   帧 0..T-2 (前 T-1 帧): SFT — MSE(pred_chosen, velocity_chosen)
                #                           因为 good/bad 前几帧相同，DPO 无意义，
                #                           用监督学习保证时序轨迹正确
                #   帧 T-1    (最后一帧):   DPO — 对比 chosen vs rejected，
                #                           只在有真实差异的帧上学偏好

                # 前 T-1 帧：SFT 监督（对 chosen 的 MSE）
                if T > 1:
                    loss_sft = F.mse_loss(
                        pred_velocity_chosen[:, :, :-1],
                        velocity_chosen[:, :, :-1],
                        reduction='mean'
                    )
                else:
                    loss_sft = torch.tensor(0.0, device=accelerator.device)

                # 最后一帧：DPO loss
                loss_chosen = F.mse_loss(
                    pred_velocity_chosen[:, :, -1:],
                    velocity_chosen[:, :, -1:],
                    reduction='mean'
                )
                loss_rejected = F.mse_loss(
                    pred_velocity_rejected[:, :, -1:],
                    velocity_rejected[:, :, -1:],
                    reduction='mean'
                )

                dpo_losses, chosen_rewards, rejected_rewards = dpo_loss(
                    loss_chosen=loss_chosen,
                    loss_rejected=loss_rejected,
                    beta=args.beta,
                    label_smoothing=0.0,
                )

                loss_dpo = dpo_losses.mean()
                
                # ========== 时序一致性 Loss (可选，完整 20 步推理) ==========
                loss_feacd = torch.tensor(0.0, device=accelerator.device)
                loss_flowD = torch.tensor(0.0, device=accelerator.device)
                
                if args.use_temporal_loss and feature_extractor is not None:
                    # 完整推理（20 步）生成最终预测视频
                    with torch.no_grad():
                        noise_scheduler.set_timesteps(
                            args.num_inference_steps,  # 默认 20 步
                            time_shift=args.time_shift,
                            device=accelerator.device
                        )
                        
                        x = vi_latent.clone()
                        for t_step in noise_scheduler.timesteps:
                            t_batch = t_step.unsqueeze(0).repeat(B).to(accelerator.device)
                            v_pred = model.forward(x_t=x, t=t_batch, ir_latent=ir_latent)
                            x = noise_scheduler.step(v_pred, t_step, x)
                        
                        # 解码到像素空间
                        pred_video_decoded = model.vae_decode_chunk(x, chunk_size=T)
                        pred_video_decoded = rearrange(pred_video_decoded, '(b t) c h w -> b c t h w', b=B, t=T)
                        # 范围: [-1, 1]
                    
                    # 计算 feaCD loss (作为监控 + reward-weighted 调整)
                    loss_feacd = compute_feaCD_loss(
                        pred_video=pred_video_decoded,
                        target_vi=vi_video,
                        target_ir=ir_video,
                        feature_extractor=feature_extractor
                    )
                    
                    # 计算 flowD loss  
                    loss_flowD = compute_flowD_loss(
                        pred_video=pred_video_decoded,
                        target_video=chosen_video
                    )
                    
                    # 注意：由于完整推理是 no_grad，feaCD/flowD 不能直接反传梯度
                    # 使用 Reward-Weighted Regression: 用这些指标调整 SFT loss 的权重
                    # reward 越低（越好），SFT 权重越高（更多监督）
                    temporal_reward = 1.0 - (loss_feacd.detach() + loss_flowD.detach()) / 2
                    temporal_reward = torch.clamp(temporal_reward, 0.1, 2.0)  # 防止极端值
                
                # 总损失: DPO(最后一帧) + sft_weight * SFT(前T-1帧) + anchor_weight * chosen-anchor(最后一帧)
                # anchor 项：对最后一帧的 chosen 目标做 MSE，防止 DPO 把 latent 推出分布
                # (等价于 DPOP / SFT-anchored DPO 中的参考模型约束)
                loss = loss_dpo + args.sft_weight * loss_sft + args.anchor_weight * loss_chosen

                # 添加时序一致性约束
                if args.use_temporal_loss:
                    loss = loss_dpo + args.sft_weight * temporal_reward * loss_sft + args.anchor_weight * loss_chosen
                
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
                
                # 计算 reward margin
                reward_margin = (chosen_rewards - rejected_rewards).mean().item()
                reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
                
                logs = {
                    "loss": loss.detach().item(),
                    "loss_dpo": loss_dpo.detach().item(),
                    "loss_sft": loss_sft.detach().item(),
                    "loss_chosen": loss_chosen.detach().item(),
                    "loss_rejected": loss_rejected.detach().item(),
                    "reward_margin": reward_margin,
                    "reward_accuracy": reward_accuracy,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                
                # 添加时序 loss 日志
                if args.use_temporal_loss:
                    logs["loss_feacd"] = loss_feacd.detach().item()
                    logs["loss_flowD"] = loss_flowD.detach().item()
                
                postfix_logs = {
                    "loss": f"{loss.item():.4f}",
                    "dpo": f"{loss_dpo.item():.4f}",
                    "acc": f"{reward_accuracy:.2f}",
                }
                
                # 添加时序 loss 到进度条
                if args.use_temporal_loss:
                    postfix_logs["feacd"] = f"{loss_feacd.item():.3f}"
                    postfix_logs["flowD"] = f"{loss_flowD.item():.3f}"
                
                progress_bar.set_postfix(**postfix_logs)
                accelerator.log(logs, step=global_step)
                
                # 每 500 步生成验证图片
                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        try:
                            print(f"\n正在生成 DPO 验证图片 (step {global_step})...")
                            pred_video = run_dpo_validation(
                                model=model,
                                noise_scheduler=noise_scheduler,
                                vi_video=vi_video.detach(),
                                ir_video=ir_video.detach(),
                                device=accelerator.device,
                                weight_dtype=weight_dtype,
                                num_steps=args.num_inference_steps,
                                time_shift=args.time_shift,
                            )
                            
                            # 验证图片保存到 output_dir/validations/
                            val_save_dir = os.path.join(args.output_dir, "validations")
                            save_dpo_validation(
                                vi_video=vi_video.detach(),
                                ir_video=ir_video.detach(),
                                chosen_video=chosen_video.detach(),
                                rejected_video=rejected_video.detach(),
                                pred_video=pred_video,
                                save_dir=val_save_dir,
                                step=global_step,
                                loss_info={
                                    'loss_chosen': loss_chosen.item(),
                                    'loss_rejected': loss_rejected.item(),
                                }
                            )
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                
                # 每 checkpointing_steps 步保存权重
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        accelerator.save_state(save_path)
                        _save_model_weights(
                            accelerator, model, save_path,
                            args.freeze_controlnet, args.freeze_unet, logger
                        )
                        logger.info(f"Saved checkpoint to {save_path}")
            
            if global_step >= args.max_train_steps:
                break
        
        epoch += 1
    
    # 保存最终 checkpoint
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(final_save_path, exist_ok=True)
        if not os.path.exists(os.path.join(final_save_path, "random_states_0.pkl")):
            accelerator.save_state(final_save_path)
        _save_model_weights(
            accelerator, model, final_save_path,
            args.freeze_controlnet, args.freeze_unet, logger
        )
        logger.info(f"Final checkpoint saved to {final_save_path}")
    
    accelerator.end_training()
    logger.info("DPO Training completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
使用示例 (DPO Training):

# 1. 基本 DPO 训练 (只训练 UNet)
python Fusion/train_dpo.py \
    --pretrained_model_path pretrained_weight/model.pt \
    --vi_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/M3SVD/visible1 \
    --ir_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/M3SVD/infrared \
    --chosen_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/M3SVD/Fusion_good \
    --rejected_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/M3SVD/Fusion_bad \
    --output_dir outputs/dpo_fusion \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --max_train_steps 5000

# 2. 从 checkpoint 继续 DPO
python Fusion/train_dpo.py \
    --pretrained_model_path pretrained_weight/model.pt \
    --resume_checkpoint outputs/vi_to_fusion/checkpoint-10000/model.safetensors \
    --chosen_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/M3SVD/Fusion_good \
    --rejected_data_dir E:/paper2/codebase/STAR/Fusion/dataset/train/M3SVD/Fusion_bad \
    --output_dir outputs/dpo_fusion

# 3. 调整 DPO 参数
python Fusion/train_dpo.py \
    --beta 0.5 \              # 更强的偏好约束
    --sft_weight 0.3 \        # 降低 SFT loss 权重
    ...

DPO 核心概念:
=============
1. Chosen (好样本): Fusion_good 目录
2. Rejected (坏样本): Fusion_bad 目录
3. DPO Loss: -log_sigmoid(β * (loss_bad - loss_good))
   - 让模型更擅长预测好样本 (loss_good < loss_bad)
4. 默认使用 8 帧窗口训练
   - 前 T-1 帧：SFT 监督，稳定时序结构
   - 最后 1 帧：DPO + anchor，强化偏好区分

监控指标:
=========
- loss_chosen: 模型对好样本的重建误差
- loss_rejected: 模型对坏样本的重建误差  
- reward_margin: chosen_reward - rejected_reward (越大越好)
- reward_accuracy: 好样本 reward > 坏样本 reward 的比例 (目标接近 1.0)

时序一致性 Loss (--use_temporal_loss):
=====================================
来自 metrics.py 的指标，改成可微分版本直接作为 loss 优化：

1. feaCD (特征运动一致性):
   - 使用 ResNet18 提取特征
   - 计算预测视频运动方向与 VI/IR 的余弦相似度
   - Loss = 1 - cos_sim (越小越好)

2. flowD (光流一致性):
   - 计算预测视频的帧差与目标视频帧差的 L1 距离
   - 让运动模式一致

使用示例:
python Fusion/train_dpo.py \\
    --use_temporal_loss \\
    ...

总 Loss = loss_dpo + sft_weight * loss_sft + anchor_weight * loss_chosen
"""


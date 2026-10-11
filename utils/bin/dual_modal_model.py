# Copyright (c) Dual-Modal Video Fusion Model


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from typing import Any, Dict, Optional, Tuple, List
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from video_to_video.modules.unet_v2v import ControlledV2VUNet, VideoControlNet
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.config import cfg
from video_to_video.utils.logger import get_logger

# 处理相对导入和绝对导入
try:
    from .image_encoder import FrozenCLIPImageEncoder
    from .block.RTB import Encoder_Ir
except ImportError:
    from Fusion.image_encoder import FrozenCLIPImageEncoder
    from Fusion.block.RTB import Encoder_Ir

logger = get_logger()


# ============================================================================
# Part 1: 轻量级融合模块
# ============================================================================

class Conv3DModule(nn.Module):
    """3D卷积模块"""
    def __init__(self, in_channels=3, out_channels=64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)):
        super(Conv3DModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        return self.act(self.conv3d(x))


class Separable3DConv(nn.Module):
    """分离式3D卷积"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(Separable3DConv, self).__init__()
        self.spatial_conv = Conv3DModule(in_channels, out_channels, 
                                         kernel_size=(1, 3, 3), 
                                         stride=(1, stride, stride), 
                                         padding=(0, 1, 1))
        self.depth_conv = Conv3DModule(out_channels, out_channels, 
                                       kernel_size=(3, 1, 1), 
                                       stride=1, 
                                       padding=(1, 0, 0))

    def forward(self, x):
        return self.depth_conv(self.spatial_conv(x))


class ConvResBlock(nn.Module):
    """残差卷积块"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = Separable3DConv(in_ch, out_ch, stride=stride)
        self.conv2 = Separable3DConv(out_ch, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, stride=(1, stride, stride)) if in_ch != out_ch or stride != 1 else nn.Identity()
        
    def forward(self, x):
        return self.skip(x) + self.conv2(self.conv1(x))


class LightweightCrossModalFusion(nn.Module):
    """轻量级跨模态融合 - 使用卷积而非注意力"""
    def __init__(self, dim):
        super().__init__()
        self.ir_proj = nn.Conv3d(dim, dim, 1)
        self.vi_proj = nn.Conv3d(dim, dim, 1)
        self.gate = nn.Sequential(nn.Conv3d(dim * 2, dim, 1), nn.Sigmoid())
        self.out = Separable3DConv(dim * 2, dim)
    
    def forward(self, ir_feat, vi_feat):
        ir_proj = self.ir_proj(ir_feat)
        vi_proj = self.vi_proj(vi_feat)
        ir_enhanced = ir_feat + ir_proj * vi_proj
        vi_enhanced = vi_feat + vi_proj * ir_proj
        concat = torch.cat([ir_enhanced, vi_enhanced], dim=1)
        gate = self.gate(concat)
        fused = self.out(concat) * gate + (ir_feat + vi_feat) / 2 * (1 - gate)
        return fused, ir_enhanced, vi_enhanced


class TemporalAttention(nn.Module):
    """时序注意力 - 只在时间维度做"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x_pooled = x.mean(dim=(3, 4)).permute(0, 2, 1)
        x_pooled = self.norm(x_pooled)
        qkv = self.qkv(x_pooled).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return x + out


class PixelSpaceFusionModule(nn.Module):
    """
    像素空间融合模块
    输入: IR视频 [B, C, T, H, W], VI视频 [B, C, T, H, W]
    输出: Fused视频 [B, C, T, H, W]
    """
    def __init__(self, in_channels=3, embed_dim=32, num_blocks=2):
        super().__init__()
        self.encoder = nn.Sequential(
            Separable3DConv(in_channels, embed_dim),
            *[ConvResBlock(embed_dim, embed_dim) for _ in range(num_blocks)]
        )
        self.cross_fusion = LightweightCrossModalFusion(embed_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads=4)
        self.decoder = nn.Sequential(
            ConvResBlock(embed_dim, embed_dim),
            nn.Conv3d(embed_dim, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, ir_video, vi_video):
        # 确保格式为 [B, C, T, H, W]
        if ir_video.shape[2] == 3:
            ir_video = rearrange(ir_video, 'b t c h w -> b c t h w')
            vi_video = rearrange(vi_video, 'b t c h w -> b c t h w')
        
        ir_feat = self.encoder(ir_video)
        vi_feat = self.encoder(vi_video)
        fused_feat, _, _ = self.cross_fusion(ir_feat, vi_feat)
        fused_feat = self.temporal_attn(fused_feat)
        fused_video = self.decoder(fused_feat)
        return fused_video


# ============================================================================
# Part 1.5: 多尺度特征融合模块 (用于 latent space 融合)
# ============================================================================

class Separable3DConv2D(nn.Module):
    """
    分离式3D卷积 - 2D版本
    用于处理 (B*T, C, H, W) 格式的特征
    先做空间卷积，再做时间卷积
    """
    def __init__(self, in_channels, out_channels, batch_size=None):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        # 时间卷积用 1x1 conv 模拟（需要 reshape）
        self.temporal_conv = nn.Conv1d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, x, batch_size, num_frames):
        """
        x: (B*T, C, H, W)
        """
        # 空间卷积
        x = self.act(self.spatial_conv(x))
        
        # 时间卷积: (B*T, C, H, W) -> (B, C, T, H, W) -> 每个空间位置做时间卷积
        BT, C, H, W = x.shape
        x = x.view(batch_size, num_frames, C, H, W)  # (B, T, C, H, W)
        x = x.permute(0, 3, 4, 2, 1).reshape(batch_size * H * W, C, num_frames)  # (B*H*W, C, T)
        x = self.act(self.temporal_conv(x))
        x = x.view(batch_size, H, W, C, num_frames).permute(0, 4, 3, 1, 2)  # (B, T, C, H, W)
        x = x.reshape(batch_size * num_frames, C, H, W)  # (B*T, C, H, W)
        
        return x


class MultiScaleFusionBlock(nn.Module):
    """
    多尺度特征融合块
    融合 ControlNet 的 VI 特征和 IR 编码器的 IR 特征
    使用分离式3D卷积进行融合
    """
    def __init__(self, ctrl_channels, ir_channels):
        """
        Args:
            ctrl_channels: ControlNet 输出的通道数
            ir_channels: IR 编码器输出的通道数
        """
        super().__init__()
        
        # IR 特征通道适配
        self.ir_adapter = nn.Sequential(
            nn.Conv2d(ir_channels, ctrl_channels, 1),
            nn.SiLU()
        )
        
        # 跨模态融合 - 使用门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(ctrl_channels * 2, ctrl_channels, 1),
            nn.Sigmoid()
        )
        
        # 融合后的精炼
        self.refine = nn.Sequential(
            nn.Conv2d(ctrl_channels * 2, ctrl_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ctrl_channels, ctrl_channels, 3, padding=1)
        )
        
        # Zero conv for residual
        self.zero_conv = nn.Conv2d(ctrl_channels, ctrl_channels, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, ctrl_feat, ir_feat):
        """
        Args:
            ctrl_feat: (B*T, C_ctrl, H, W) - ControlNet 输出
            ir_feat: (B*T, C_ir, H, W) - IR 编码器输出
        
        Returns:
            fused_feat: (B*T, C_ctrl, H, W) - 融合后的特征
        """
        # 适配 IR 特征通道
        ir_adapted = self.ir_adapter(ir_feat)
        
        # 空间对齐 (如果尺寸不匹配)
        if ir_adapted.shape[-2:] != ctrl_feat.shape[-2:]:
            ir_adapted = F.interpolate(ir_adapted, size=ctrl_feat.shape[-2:], 
                                       mode='bilinear', align_corners=False)
        
        # 门控融合
        concat = torch.cat([ctrl_feat, ir_adapted], dim=1)
        gate = self.gate(concat)
        
        # 融合
        fused = self.refine(concat)
        
        # 残差连接 (zero-init)
        output = ctrl_feat + self.zero_conv(fused * gate)
        
        return output


class IRLatentEncoder(nn.Module):
    """
    红外 Latent 编码器
    将 IR latent 编码为多尺度特征，与 ControlNet 输出对齐
    
    ControlNet 输出尺度:
    - Scale 1 (H×W):     Block 0-2, 320 channels
    - Scale 1/2 (H/2×W/2): Block 3-5, 640 channels  
    - Scale 1/4 (H/4×W/4): Block 6-8, 1280 channels
    - Scale 1/8 (H/8×W/8): Block 9-12, 1280 channels
    """
    def __init__(self, 
                 in_channels=4,  # latent channels
                 dims=[320, 640, 1280, 1280],  # 与 ControlNet 对齐
                 use_rtb=True):
        super().__init__()
        
        self.use_rtb = use_rtb
        
        if use_rtb:
            # 使用 RTB 的 Encoder_Ir 架构
            # Encoder_Ir 输出: dim=32 -> [32, 64, 128, 256]
            self.encoder = Encoder_Ir(inp_channels=in_channels, dim=32)
            # 通道适配到 ControlNet 尺寸
            self.adapters = nn.ModuleList([
                nn.Conv2d(32, dims[0], 1),    # Level 1: 32 -> 320
                nn.Conv2d(64, dims[1], 1),    # Level 2: 64 -> 640
                nn.Conv2d(128, dims[2], 1),   # Level 3: 128 -> 1280
                nn.Conv2d(256, dims[3], 1),   # Level 4: 256 -> 1280
            ])
        else:
            # 简单卷积编码器
            self.encoder = nn.ModuleList([
                # Level 1: scale 1
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(dims[0], dims[0], 3, padding=1),
                ),
                # Level 2: scale 1/2
                nn.Sequential(
                    nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(dims[1], dims[1], 3, padding=1),
                ),
                # Level 3: scale 1/4
                nn.Sequential(
                    nn.Conv2d(dims[1], dims[2], 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(dims[2], dims[2], 3, padding=1),
                ),
                # Level 4: scale 1/8
                nn.Sequential(
                    nn.Conv2d(dims[2], dims[3], 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(dims[3], dims[3], 3, padding=1),
                ),
            ])
            self.adapters = None
    
    def forward(self, ir_latent):
        """
        Args:
            ir_latent: (B*T, C, H, W) - IR latent
        
        Returns:
            multi_scale_features: List of features at different scales
                - Level 1: (B*T, 320, H, W)
                - Level 2: (B*T, 640, H/2, W/2)
                - Level 3: (B*T, 1280, H/4, W/4)
                - Level 4: (B*T, 1280, H/8, W/8)
        """
        if self.use_rtb:
            # Encoder_Ir 返回 (out4, out3, out2, out1)
            # out1: Level 1 (最大尺度), out4: Level 4 (最小尺度)
            out4, out3, out2, out1 = self.encoder(ir_latent)
            
            # 适配通道
            feat1 = self.adapters[0](out1)  # Level 1
            feat2 = self.adapters[1](out2)  # Level 2
            feat3 = self.adapters[2](out3)  # Level 3
            feat4 = self.adapters[3](out4)  # Level 4
            
            return [feat1, feat2, feat3, feat4]
        else:
            features = []
            x = ir_latent
            for encoder in self.encoder:
                x = encoder(x)
                features.append(x)
            return features


class MultiScaleControlFusion(nn.Module):
    """
    多尺度控制信号融合模块
    
    将 IR 编码器的多尺度特征与 VideoControlNet 的输出融合
    
    ControlNet 输出结构 (共13个):
    - Block 0-2: Level 0, 320 channels, scale=1
    - Block 3: Downsample
    - Block 4-5: Level 1, 640 channels, scale=1/2
    - Block 6: Downsample
    - Block 7-8: Level 2, 1280 channels, scale=1/4
    - Block 9: Downsample  
    - Block 10-11: Level 3, 1280 channels, scale=1/8
    - Block 12: Middle, 1280 channels, scale=1/8
    """
    def __init__(self):
        super().__init__()
        
        # ControlNet 各 block 的通道数
        self.ctrl_channels = [
            320, 320, 320,  # Block 0-2 (Level 0)
            320,            # Block 3 (Downsample)
            640, 640,       # Block 4-5 (Level 1)
            640,            # Block 6 (Downsample)
            1280, 1280,     # Block 7-8 (Level 2)
            1280,           # Block 9 (Downsample)
            1280, 1280,     # Block 10-11 (Level 3)
            1280            # Block 12 (Middle)
        ]
        
        # IR 特征对应的 Level
        self.block_to_level = [
            0, 0, 0,  # Block 0-2 -> Level 0 (320 ch, scale 1)
            0,        # Block 3
            1, 1,     # Block 4-5 -> Level 1 (640 ch, scale 1/2)
            1,        # Block 6
            2, 2,     # Block 7-8 -> Level 2 (1280 ch, scale 1/4)
            2,        # Block 9
            3, 3,     # Block 10-11 -> Level 3 (1280 ch, scale 1/8)
            3         # Block 12
        ]
        
        # IR 编码器各 Level 的通道数 (适配后)
        self.ir_channels = [320, 640, 1280, 1280]
        
        # 为每个 ControlNet block 创建融合模块
        self.fusion_blocks = nn.ModuleList()
        for i, (ctrl_ch, level) in enumerate(zip(self.ctrl_channels, self.block_to_level)):
            ir_ch = self.ir_channels[level]
            self.fusion_blocks.append(MultiScaleFusionBlock(ctrl_ch, ir_ch))
    
    def forward(self, ctrl_signals: List[torch.Tensor], ir_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            ctrl_signals: List of 13 tensors from VideoControlNet
            ir_features: List of 4 tensors from IRLatentEncoder [Level0, Level1, Level2, Level3]
        
        Returns:
            fused_signals: List of 13 fused tensors
        """
        fused_signals = []
        
        for i, (ctrl, fusion_block) in enumerate(zip(ctrl_signals, self.fusion_blocks)):
            level = self.block_to_level[i]
            ir_feat = ir_features[level]
            fused = fusion_block(ctrl, ir_feat)
            fused_signals.append(fused)
        
        return fused_signals


# ============================================================================
# Part 2: 完整的双模态融合系统 (包含 VAE, UNet, 扩散过程)
# ============================================================================

class DualModalFusionModel:
    """
    双模态视频融合模型 - 完整系统
    
    架构:
    IR/VI视频 --> PixelSpaceFusionModule --> fused_video --> VAE编码 --> fused_latent
                                                                            |
                                                          +-----------------+
                                                          |                 |
                                                          v                 v
                                                      ControlNet          加噪
                                                          |                 |
                                                          v                 v
                                                       控制信号 -------> UNet
                                                                            |
                                                                            v
                                                                     pred_latent --> VAE解码 --> 融合视频
    """
    def __init__(self, opt, device=torch.device('cuda:0')):
        self.opt = opt
        self.device = device
        
        # ============ 图片编码器 (条件) ============
        image_encoder = FrozenCLIPImageEncoder(
            pretrained=getattr(opt, 'clip_pretrained', 'laion2b_s32b_b79k'),
            arch=getattr(opt, 'clip_arch', 'ViT-H-14'),
            device=self.device,
            freeze=True
        )
        image_encoder.model.to(self.device)
        self.image_encoder = image_encoder
        logger.info('Built image encoder with FrozenCLIPImageEncoder')
        
        # ============ 像素空间融合模块 ============
        fusion_embed_dim = getattr(opt, 'fusion_embed_dim', 32)
        fusion_num_blocks = getattr(opt, 'fusion_num_blocks', 2)
        self.pixel_fusion = PixelSpaceFusionModule(
            in_channels=3,
            embed_dim=fusion_embed_dim,
            num_blocks=fusion_num_blocks
        ).to(self.device)
        logger.info(f'Built PixelSpaceFusionModule (embed_dim={fusion_embed_dim}, num_blocks={fusion_num_blocks})')
        
        # ============ U-Net with ControlNet ============
        generator = ControlledV2VUNet()
        generator = generator.to(self.device)
        
        # 加载预训练权重
        if hasattr(opt, 'model_path') and opt.model_path:
            cfg.model_path = opt.model_path
            if cfg.model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                load_dict = load_file(cfg.model_path)
            else:
                load_dict = torch.load(cfg.model_path, map_location='cpu')
                if 'state_dict' in load_dict:
                    load_dict = load_dict['state_dict']
            
            ret = generator.load_state_dict(load_dict, strict=False)
            logger.info('Load model path {}, with local status {}'.format(cfg.model_path, ret))
        
        self.generator = generator.half()
        logger.info('Built UNet with ControlNet')
        
        # ============ 噪声调度器 ============
        sigmas = noise_schedule(
            schedule='logsnr_cosine_interp',
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0
        )
        self.diffusion = GaussianDiffusion(sigmas=sigmas)
        logger.info('Built diffusion scheduler')
        
        # ============ Temporal VAE ============
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", 
            subfolder="vae", 
            variant="fp16"
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(self.device)
        self.vae = vae
        logger.info('Built Temporal VAE')
        
        # ============ 负样本 (用于 classifier-free guidance) ============
        self.negative_y = torch.zeros(1, 1, 1024, device=self.device)
        
        torch.cuda.empty_cache()
    
    def freeze_for_training(self):
        """冻结预训练部分，只保留融合模块可训练"""
        # 冻结 UNet
        for param in self.generator.parameters():
            param.requires_grad = False
        
        # 融合模块可训练
        for param in self.pixel_fusion.parameters():
            param.requires_grad = True
        
        logger.info('Froze generator, pixel_fusion is trainable')
    
    def get_trainable_parameters(self):
        """返回需要训练的参数"""
        return list(self.pixel_fusion.parameters())
    
    def train_mode(self):
        """训练模式"""
        self.pixel_fusion.train()
        self.generator.eval()
    
    def eval_mode(self):
        """评估模式"""
        self.pixel_fusion.eval()
        self.generator.eval()
    
    # ============ VAE 编码/解码 ============
    
    def vae_encode(self, video, chunk_size=1):
        """
        VAE编码
        video: [B, T, C, H, W] 或 [B, C, T, H, W]
        返回: [B, C, T, H/8, W/8]
        """
        # 确保格式为 [B, T, C, H, W]
        if video.dim() == 5:
            if video.shape[2] == 3:  # [B, T, C, H, W]
                pass
            else:  # [B, C, T, H, W]
                video = video.permute(0, 2, 1, 3, 4)
        
        num_f = video.shape[1]
        video_flat = rearrange(video, "b f c h w -> (b f) c h w")
        
        z_list = []
        for ind in range(0, video_flat.shape[0], chunk_size):
            z_list.append(self.vae.encode(video_flat[ind:ind+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        return z * self.vae.config.scaling_factor
    
    def vae_decode(self, z, num_f):
        """VAE解码"""
        return self.vae.decode(z / self.vae.config.scaling_factor, num_frames=num_f).sample
    
    def vae_decode_chunk(self, z, chunk_size=3):
        """分块VAE解码"""
        z = rearrange(z, "b c f h w -> (b f) c h w")
        video = []
        for ind in range(0, z.shape[0], chunk_size):
            num_f = z[ind:ind+chunk_size].shape[0]
            video.append(self.vae_decode(z[ind:ind+chunk_size], num_f))
        video = torch.cat(video)
        return video
    
    # ============ 推理接口 ============
    
    def test(self, 
             input: Dict[str, Any], 
             total_noise_levels=1000,
             steps=50, 
             solver_mode='fast', 
             guide_scale=7.5, 
             max_chunk_len=32):
        """
        推理/测试函数
        
        Args:
            input: 字典，包含:
                - 'ir_video': 红外视频 [B, C, T, H, W]
                - 'vi_video': 可见光视频 [B, C, T, H, W]
                - 'condition_image': 条件图片 (PIL.Image或路径)
                - 'target_res': (target_h, target_w) 目标分辨率
        """
        ir_video = input['ir_video']
        vi_video = input['vi_video']
        condition_image = input['condition_image']
        (target_h, target_w) = input['target_res']
        
        # 处理条件图片
        if isinstance(condition_image, str):
            from PIL import Image
            condition_image = Image.open(condition_image).convert('RGB')
        
        # 确保有batch维度
        if ir_video.dim() == 4:
            ir_video = ir_video.unsqueeze(0)
            vi_video = vi_video.unsqueeze(0)
        
        B, C, T, H, W = ir_video.shape
        
        # 调整分辨率
        ir_video = self._resize_video(ir_video, target_h, target_w)
        vi_video = self._resize_video(vi_video, target_h, target_w)
        
        # Padding
        _, _, _, h, w = ir_video.shape
        padding = self._compute_padding(h, w)
        ir_video = self._pad_video(ir_video, padding)
        vi_video = self._pad_video(vi_video, padding)
        
        ir_video = ir_video.to(self.device).half()
        vi_video = vi_video.to(self.device).half()
        
        with torch.no_grad():
            # Step 1: 融合
            # 先归一化到 [0, 1] 范围给融合模块
            ir_01 = (ir_video + 1) / 2
            vi_01 = (vi_video + 1) / 2
            fused_video = self.pixel_fusion(ir_01.float(), vi_01.float())  # [B, C, T, H, W], [0, 1]
            
            # 转回 [-1, 1] 给 VAE
            fused_video_norm = fused_video * 2 - 1
            
            # Step 2: VAE 编码
            fused_latent = self.vae_encode(fused_video_norm)
            
            # Step 3: 编码条件图片
            image_features = self.image_encoder(condition_image).detach()
            if image_features.dim() == 2:
                B_img, D = image_features.shape
                y = image_features.unsqueeze(1).expand(B_img, 77, D)
            else:
                y = image_features
        
        with amp.autocast(enabled=True):
            # Step 4: 添加噪声
            t = torch.LongTensor([total_noise_levels - 1]).to(self.device)
            noised = self.diffusion.diffuse(fused_latent, t)
            
            # Step 5: 准备模型参数
            model_kwargs = [{'y': y}, {'y': self.negative_y}]
            model_kwargs.append({'hint': fused_latent})
            
            _, _, frames_num, _, _ = fused_latent.shape
            chunk_inds = self._make_chunks(frames_num, 0, max_chunk_len) if frames_num > max_chunk_len else None
            
            # Step 6: 采样
            gen_vid = self.diffusion.sample_sr(
                noise=noised,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver='dpmpp_2m_sde',
                solver_mode=solver_mode,
                return_intermediate=None,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing',
                chunk_inds=chunk_inds,
            )
            
            # Step 7: VAE 解码
            vid_tensor_gen = self.vae_decode_chunk(gen_vid, chunk_size=3)
        
        # 移除padding
        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:, :, h1:h+h1, w1:w+w1]
        
        gen_video = rearrange(vid_tensor_gen, '(b f) c h w -> b c f h w', b=B)
        
        torch.cuda.empty_cache()
        return gen_video.float().cpu()
    
    # ============ 工具函数 ============
    
    def _resize_video(self, video, target_h, target_w):
        """调整视频分辨率"""
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video = F.interpolate(video, [target_h, target_w], mode='bilinear', align_corners=False)
        video = video.reshape(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)
        return video
    
    def _compute_padding(self, h, w):
        """计算padding"""
        target_h = ((h + 63) // 64) * 64
        target_w = ((w + 63) // 64) * 64
        return (0, target_w - w, 0, target_h - h)
    
    def _pad_video(self, video, padding):
        """对视频进行padding"""
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video = F.pad(video, padding, 'constant', 0)
        _, _, H_new, W_new = video.shape
        video = video.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
        return video
    
    def _make_chunks(self, f_num, interp_f_num, max_chunk_len, chunk_overlap_ratio=0.5):
        """创建chunks"""
        chunk_len = int((max_chunk_len - 1) // (1 + interp_f_num) * (interp_f_num + 1) + 1)
        o_len = int((max_chunk_len * chunk_overlap_ratio - 1) // (1 + interp_f_num) * (interp_f_num + 1) + 1)
        return self._sliding_windows_1d(f_num, chunk_len, o_len)
    
    def _sliding_windows_1d(self, length, window_size, overlap_size):
        """滑动窗口"""
        stride = window_size - overlap_size
        ind = 0
        coords = []
        while ind < length:
            if ind + window_size * 1.25 >= length:
                coords.append((ind, length))
                break
            else:
                coords.append((ind, ind + window_size))
                ind += stride
        return coords


# ============================================================================
# Part 3: 用于训练的 nn.Module 包装器
# ============================================================================

class DualModalControlledV2VUNet(nn.Module):
    """
    用于训练的 nn.Module 版本 (旧版 - 像素空间融合)
    包含融合模块和 ControlledV2VUNet
    """
    def __init__(self, 
                 fusion_embed_dim=32,
                 fusion_num_blocks=2,
                 pretrained_model_path=None):
        super().__init__()
        
        # 融合模块
        self.pixel_fusion = PixelSpaceFusionModule(
            in_channels=3,
            embed_dim=fusion_embed_dim,
            num_blocks=fusion_num_blocks
        )
        
        # ControlledV2VUNet
        self.controlled_unet = ControlledV2VUNet()
        
        if pretrained_model_path is not None:
            self._load_pretrained(pretrained_model_path)
    
    def _load_pretrained(self, model_path):
        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            load_dict = load_file(model_path)
        else:
            load_dict = torch.load(model_path, map_location='cpu')
            if 'state_dict' in load_dict:
                load_dict = load_dict['state_dict']
        
        ret = self.controlled_unet.load_state_dict(load_dict, strict=False)
        print(f'Loaded pretrained model from {model_path}')
        print(f'Missing keys: {len(ret.missing_keys)}, Unexpected keys: {len(ret.unexpected_keys)}')
    
    def get_trainable_parameters(self):
        """返回需要训练的参数 - 只有融合模块"""
        return list(self.pixel_fusion.parameters())
    
    def freeze_pretrained(self):
        """冻结预训练参数"""
        for param in self.controlled_unet.parameters():
            param.requires_grad = False
        for param in self.pixel_fusion.parameters():
            param.requires_grad = True
    
    def forward(self, x, t, y, ir_video=None, vi_video=None, hint=None, **kwargs):
        """
        前向传播
        
        训练时: 提供 ir_video 和 vi_video
        推理时: 直接提供 hint
        """
        fused_video = None
        
        if ir_video is not None and vi_video is not None:
            fused_video = self.pixel_fusion(ir_video, vi_video)
        
        output = self.controlled_unet(x=x, t=t, y=y, hint=hint, **kwargs)
        
        if fused_video is not None:
            return output, fused_video
        return output


class DualModalV2VUNet(nn.Module):
    """
    双模态视频融合 UNet (新版 - Latent 空间多尺度融合)
    
    架构:
    ======
    VI latent ──> VideoControlNet ──> 多尺度控制信号 (13个)
                                            │
                                            ↓
    IR latent ──> IRLatentEncoder ──> 多尺度IR特征 ──> MultiScaleControlFusion
                     (4个尺度)                              │
                                                           ↓
                                                    融合后的控制信号
                                                           │
                                                           ↓
                                                    Vid2VidSDUNet
                                                           │
                                                           ↓
                                                      输出 latent
    
    训练策略:
    - VideoControlNet: 🔥 可训练
    - IRLatentEncoder: 🔥 可训练
    - MultiScaleControlFusion: 🔥 可训练
    - Vid2VidSDUNet: ❄️ 冻结 (但 local 层 🔥 训练)
    """
    def __init__(self, 
                 use_rtb=True,
                 pretrained_model_path=None):
        super().__init__()
        
        # IR Latent 编码器
        self.ir_encoder = IRLatentEncoder(
            in_channels=4,
            dims=[320, 640, 1280, 1280],
            use_rtb=use_rtb
        )
        
        # 多尺度融合模块
        self.multi_scale_fusion = MultiScaleControlFusion()
        
        # VideoControlNet (处理 VI)
        self.video_controlnet = VideoControlNet()
        
        # 主 UNet
        from video_to_video.modules.unet_v2v import Vid2VidSDUNet
        self.vid2vid_unet = Vid2VidSDUNet()
        
        if pretrained_model_path is not None:
            self._load_pretrained(pretrained_model_path)
    
    def _load_pretrained(self, model_path):
        """加载预训练权重到 VideoControlNet 和 Vid2VidSDUNet"""
        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            load_dict = load_file(model_path)
        else:
            load_dict = torch.load(model_path, map_location='cpu')
            if 'state_dict' in load_dict:
                load_dict = load_dict['state_dict']
        
        # 分离 VideoControlNet 和 UNet 的权重
        ctrl_dict = {}
        unet_dict = {}
        for k, v in load_dict.items():
            if k.startswith('VideoControlNet.'):
                ctrl_dict[k.replace('VideoControlNet.', '')] = v
            else:
                unet_dict[k] = v
        
        ret1 = self.video_controlnet.load_state_dict(ctrl_dict, strict=False)
        ret2 = self.vid2vid_unet.load_state_dict(unet_dict, strict=False)
        
        print(f'Loaded VideoControlNet: missing={len(ret1.missing_keys)}, unexpected={len(ret1.unexpected_keys)}')
        print(f'Loaded Vid2VidSDUNet: missing={len(ret2.missing_keys)}, unexpected={len(ret2.unexpected_keys)}')
    
    def freeze_for_training(self):
        """
        冻结预训练部分，保留融合相关模块可训练
        
        可训练:
        - ir_encoder
        - multi_scale_fusion
        - video_controlnet
        - vid2vid_unet 中的 local 层
        """
        # 冻结主 UNet
        for name, param in self.vid2vid_unet.named_parameters():
            if 'local' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # VideoControlNet 可训练
        for param in self.video_controlnet.parameters():
            param.requires_grad = True
        
        # IR 编码器可训练
        for param in self.ir_encoder.parameters():
            param.requires_grad = True
        
        # 多尺度融合可训练
        for param in self.multi_scale_fusion.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """返回需要训练的参数"""
        params = []
        params.extend(list(self.ir_encoder.parameters()))
        params.extend(list(self.multi_scale_fusion.parameters()))
        params.extend(list(self.video_controlnet.parameters()))
        # 加上 UNet 中的 local 层
        for name, param in self.vid2vid_unet.named_parameters():
            if 'local' in name and param.requires_grad:
                params.append(param)
        return params
    
    def forward(self, 
                x,          # 加噪后的 latent (目标)
                t,          # timestep
                y,          # condition (CLIP features)
                ir_latent,  # IR latent
                vi_latent,  # VI latent (用作 hint)
                **kwargs):
        """
        前向传播
        
        Args:
            x: (B, C, T, H, W) - 加噪后的 latent
            t: (B,) - timestep
            y: (B, 77, 1024) - CLIP 条件
            ir_latent: (B, C, T, H, W) - IR latent
            vi_latent: (B, C, T, H, W) - VI latent
        
        Returns:
            output: (B, C, T, H, W) - 预测的 velocity
        """
        batch, _, f, _, _ = x.shape
        device = x.device
        
        # 1. VideoControlNet 处理 VI
        # 输入 hint 是 vi_latent
        ctrl_signals = self.video_controlnet(x, t, y, hint=vi_latent, **kwargs)
        
        # 2. IR 编码器提取多尺度 IR 特征
        # 先 reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        ir_flat = rearrange(ir_latent, 'b c t h w -> (b t) c h w')
        ir_features = self.ir_encoder(ir_flat)
        
        # 3. 多尺度融合
        fused_ctrl = self.multi_scale_fusion(ctrl_signals, ir_features)
        
        # 4. 主 UNet 前向传播 (注入融合后的控制信号)
        output = self._forward_unet_with_control(x, t, y, fused_ctrl, **kwargs)
        
        return output
    
    def _forward_unet_with_control(self, x, t, y, control, **kwargs):
        """
        带控制信号注入的 UNet 前向传播
        模仿 ControlledV2VUNet.forward 但使用外部提供的控制信号
        """
        from video_to_video.modules.unet_v2v import (
            sinusoidal_embedding, prob_mask_like, default, USE_TEMPORAL_TRANSFORMER
        )
        
        batch, _, f, _, _ = x.shape
        device = x.device
        self.vid2vid_unet.batch = batch
        
        # Time embedding
        e = self.vid2vid_unet.time_embed(sinusoidal_embedding(t, self.vid2vid_unet.dim))
        e = e.repeat_interleave(repeats=f, dim=0)
        
        # Context
        context = y.repeat_interleave(repeats=f, dim=0)
        
        # Time relative position bias
        if self.vid2vid_unet.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.vid2vid_unet.time_rel_pos_bias(x.shape[2], device=device)
        else:
            time_rel_pos_bias = None
        
        focus_present_mask = None
        video_mask = None
        
        # Reshape x
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        
        # Encoder
        xs = []
        for block in self.vid2vid_unet.input_blocks:
            x = self.vid2vid_unet._forward_single(
                block, x, e, context, time_rel_pos_bias, 
                focus_present_mask, video_mask
            )
            xs.append(x)
        
        # Middle
        for block in self.vid2vid_unet.middle_block:
            x = self.vid2vid_unet._forward_single(
                block, x, e, context, time_rel_pos_bias,
                focus_present_mask, video_mask
            )
        
        # 注入 middle control
        if control is not None:
            x = control.pop() + x
        
        # Decoder with control injection
        for block in self.vid2vid_unet.output_blocks:
            if control is None:
                x = torch.cat([x, xs.pop()], dim=1)
            else:
                x = torch.cat([x, xs.pop() + control.pop()], dim=1)
            x = self.vid2vid_unet._forward_single(
                block, x, e, context, time_rel_pos_bias,
                focus_present_mask, video_mask,
                reference=xs[-1] if len(xs) > 0 else None
            )
        
        # Head
        x = self.vid2vid_unet.out(x)
        
        # Reshape back
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)
        
        return x


# ============================================================================
# Part 4: 无监督融合损失函数
# ============================================================================

def RGB2YCrCb(rgb_image):
    """RGB转YCbCr"""
    R, G, B = rgb_image[:, 0:1], rgb_image[:, 1:2], rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 0.5
    return Y.clamp(0, 1), Cb.clamp(0, 1), Cr.clamp(0, 1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


class DualModalFusionLoss(nn.Module):
    """无监督融合损失"""
    def __init__(self, 
                 ssim_weight=1.0,
                 intensity_weight=5.0,
                 gradient_weight=5.0,
                 color_weight=10.0,
                 temporal_weight=1.0,
                 window_size=11):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.intensity_weight = intensity_weight
        self.gradient_weight = gradient_weight
        self.color_weight = color_weight
        self.temporal_weight = temporal_weight
        self.window_size = window_size
        self.register_buffer('window', create_window(window_size, 1))
        
        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=1)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=1)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=1) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()
    
    def _gradient(self, img):
        img = F.pad(img, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(img, self.sobel_x)
        grad_y = F.conv2d(img, self.sobel_y)
        return torch.abs(grad_x), torch.abs(grad_y)
    
    def forward(self, img_fused, img_ir, img_vi):
        # 确保格式 [B, C, T, H, W]
        if img_fused.dim() == 5 and img_fused.shape[2] == 3:
            img_fused = rearrange(img_fused, 'b t c h w -> b c t h w')
        if img_ir.dim() == 5 and img_ir.shape[2] == 3:
            img_ir = rearrange(img_ir, 'b t c h w -> b c t h w')
        if img_vi.dim() == 5 and img_vi.shape[2] == 3:
            img_vi = rearrange(img_vi, 'b t c h w -> b c t h w')
        
        B, C, T, H, W = img_fused.shape
        
        img_fused_2d = rearrange(img_fused, 'b c t h w -> (b t) c h w')
        img_ir_2d = rearrange(img_ir, 'b c t h w -> (b t) c h w')
        img_vi_2d = rearrange(img_vi, 'b c t h w -> (b t) c h w')
        
        Y_f, Cb_f, Cr_f = RGB2YCrCb(img_fused_2d)
        Y_ir, _, _ = RGB2YCrCb(img_ir_2d)
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi_2d)
        
        loss_ssim = self.ssim_weight * (self._ssim(Y_vi, Y_f) + self._ssim(Y_ir, Y_f))
        
        Y_max = torch.max(Y_vi, Y_ir)
        loss_intensity = self.intensity_weight * F.l1_loss(Y_f, Y_max)
        
        grad_vi_x, grad_vi_y = self._gradient(Y_vi)
        grad_ir_x, grad_ir_y = self._gradient(Y_ir)
        grad_f_x, grad_f_y = self._gradient(Y_f)
        loss_gradient = self.gradient_weight * (
            F.l1_loss(grad_f_x, torch.max(grad_vi_x, grad_ir_x)) +
            F.l1_loss(grad_f_y, torch.max(grad_vi_y, grad_ir_y))
        )
        
        loss_color = self.color_weight * (F.l1_loss(Cb_f, Cb_vi) + F.l1_loss(Cr_f, Cr_vi))
        
        if T > 1:
            Y_f_t = Y_f.view(B, T, 1, H, W)
            Y_ir_t = Y_ir.view(B, T, 1, H, W)
            loss_temp = 0
            for t in range(1, T):
                f_diff = (Y_f_t[:, t] - Y_f_t[:, t-1]).view(B, -1)
                ir_diff = (Y_ir_t[:, t] - Y_ir_t[:, t-1]).view(B, -1)
                loss_temp += (1 - F.cosine_similarity(f_diff, ir_diff, dim=1)).mean()
            loss_temporal = self.temporal_weight * loss_temp / (T - 1)
        else:
            loss_temporal = torch.tensor(0.0, device=img_fused.device)
        
        total = loss_ssim + loss_intensity + loss_gradient + loss_color + loss_temporal
        
        return {
            'loss_ssim': loss_ssim,
            'loss_intensity': loss_intensity,
            'loss_gradient': loss_gradient,
            'loss_color': loss_color,
            'loss_temporal': loss_temporal,
            'loss_fusion_total': total
        }


class CombinedLoss(nn.Module):
    """组合损失 = v-prediction + 融合损失"""
    def __init__(self, v_pred_weight=1.0, fusion_weight=0.1, **fusion_kwargs):
        super().__init__()
        self.v_pred_weight = v_pred_weight
        self.fusion_weight = fusion_weight
        self.fusion_loss = DualModalFusionLoss(**fusion_kwargs)
    
    def forward(self, v_pred, v_target, img_fused=None, img_ir=None, img_vi=None):
        loss_v = F.mse_loss(v_pred.float(), v_target.float())
        loss_dict = {'loss_v_pred': loss_v * self.v_pred_weight}
        
        if img_fused is not None and img_ir is not None and img_vi is not None:
            fusion_losses = self.fusion_loss(img_fused, img_ir, img_vi)
            for k, v in fusion_losses.items():
                loss_dict[k] = v * self.fusion_weight
        
        loss_dict['total_loss'] = sum(loss_dict.values())
        return loss_dict


# ============================================================================
# Part 5: 工具函数
# ============================================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_info(model):
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable / 1e6:.2f}M")


def run_dual_modal_inference_test():
    """
    双模态融合完整推理测试
    包含：加载模型 -> 加载数据 -> 融合 -> VAE编码 -> 扩散采样 -> VAE解码 -> 保存
    """
    import argparse
    import cv2
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    import glob
    
    # ===== 参数配置 =====
    parser = argparse.ArgumentParser(description="双模态融合推理测试")
    parser.add_argument("--pretrained_model_path", type=str, 
                        default=r"E:\paper2\codebase\STAR\pretrained_weight\model.pt",
                        help="预训练模型权重路径")
    parser.add_argument("--ir_video_path", type=str, default=None,
                        help="红外视频路径（图片序列文件夹）")
    parser.add_argument("--vi_video_path", type=str, default=None,
                        help="可见光视频路径（图片序列文件夹）")
    parser.add_argument("--output_path", type=str, default="./dual_modal_test.mp4",
                        help="输出视频路径")
    parser.add_argument("--num_frames", type=int, default=8, help="帧数")
    parser.add_argument("--video_size", type=int, nargs=2, default=[1024, 1024], help="视频尺寸 (H W)")
    parser.add_argument("--num_steps", type=int, default=15, help="采样步数")
    parser.add_argument("--fusion_only", action="store_true", help="仅测试融合模块")
    parser.add_argument("--quick_test", action="store_true", help="快速测试（不加载大模型）")
    
    args, _ = parser.parse_known_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    
    print("=" * 70)
    print("双模态视频融合 - 推理测试")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"精度: {weight_dtype}")
    
    # ===== Step 1: 测试融合模块 =====
    print("\n" + "=" * 70)
    print("[Step 1/5] 测试 PixelSpaceFusionModule...")
    print("=" * 70)
    
    fusion = PixelSpaceFusionModule(in_channels=3, embed_dim=32, num_blocks=2)
    fusion = fusion.to(device)
    
    # 测试不同分辨率
    test_resolutions = [(256, 256), (512, 512)]
    for h, w in test_resolutions:
        print(f"\n  测试分辨率: {h}x{w}")
        ir_video = torch.randn(1, 3, 8, h, w).to(device)
        vi_video = torch.randn(1, 3, 8, h, w).to(device)
        
        with torch.no_grad():
            fused = fusion(ir_video, vi_video)
        
        print(f"    IR shape: {ir_video.shape}")
        print(f"    VI shape: {vi_video.shape}")
        print(f"    Fused shape: {fused.shape}")
        print(f"    输出范围: [{fused.min():.4f}, {fused.max():.4f}]")
        
        if device.type == 'cuda':
            print(f"    GPU显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("\n  融合模块参数:")
    print_model_info(fusion)
    
    # ===== Step 2: 测试损失函数 =====
    print("\n" + "=" * 70)
    print("[Step 2/5] 测试 DualModalFusionLoss...")
    print("=" * 70)
    
    loss_fn = DualModalFusionLoss().to(device)
    
    img_fused = torch.rand(1, 3, 8, 256, 256).to(device)
    img_ir = torch.rand(1, 3, 8, 256, 256).to(device)
    img_vi = torch.rand(1, 3, 8, 256, 256).to(device)
    
    losses = loss_fn(img_fused, img_ir, img_vi)
    
    print("\n  损失分量:")
    for k, v in losses.items():
        print(f"    {k}: {v.item():.6f}")
    
    # ===== Step 3: 测试完整模型结构 =====
    print("\n" + "=" * 70)
    print("[Step 3/5] 测试 DualModalControlledV2VUNet 结构...")
    print("=" * 70)
    
    model = DualModalControlledV2VUNet(
        fusion_embed_dim=32,
        fusion_num_blocks=2,
        pretrained_model_path=None  # 不加载权重，只测试结构
    )
    
    print("\n  完整模型参数:")
    print_model_info(model)
    
    trainable_params = model.get_trainable_parameters()
    print(f"\n  可训练参数: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    
    if args.quick_test:
        print("\n" + "=" * 70)
        print("快速测试完成！(跳过完整推理)")
        print("=" * 70)
        return
    
    # ===== Step 4: 加载完整模型进行推理测试 =====
    print("\n" + "=" * 70)
    print("[Step 4/5] 加载完整模型...")
    print("=" * 70)
    
    # 检查预训练权重
    if not os.path.exists(args.pretrained_model_path):
        print(f"  警告: 预训练权重不存在: {args.pretrained_model_path}")
        print("  跳过完整推理测试")
        print("\n" + "=" * 70)
        print("测试完成！(基础功能正常)")
        print("=" * 70)
        return
    
    print(f"  加载预训练权重: {args.pretrained_model_path}")
    
    # 加载完整模型
    model_full = DualModalControlledV2VUNet(
        fusion_embed_dim=32,
        fusion_num_blocks=2,
        pretrained_model_path=args.pretrained_model_path
    )
    model_full.to(device)
    model_full.freeze_pretrained()
    model_full.pixel_fusion.to(weight_dtype)
    model_full.controlled_unet.half()
    
    print("  模型加载完成！")
    
    # 加载 VAE
    print("  加载 Temporal VAE...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", 
        subfolder="vae", 
        variant="fp16"
    )
    vae.to(device, dtype=weight_dtype)
    vae.eval()
    print("  VAE 加载完成！")
    
    # 加载噪声调度器
    print("  初始化噪声调度器...")
    sigmas = noise_schedule(
        schedule='logsnr_cosine_interp',
        n=1000,
        zero_terminal_snr=True,
        scale_min=2.0,
        scale_max=4.0
    )
    diffusion = GaussianDiffusion(sigmas=sigmas)
    print("  噪声调度器初始化完成！")
    
    # ===== Step 5: 运行推理 =====
    print("\n" + "=" * 70)
    print("[Step 5/5] 运行推理测试...")
    print("=" * 70)
    
    # 创建测试数据 (如果没有提供真实数据)
    H, W = args.video_size
    T = args.num_frames
    
    print(f"  视频尺寸: {W}x{H}, 帧数: {T}")
    
    # 模拟 IR 和 VI 视频 (范围 [0, 1])
    ir_video = torch.rand(1, 3, T, H, W).to(device, dtype=weight_dtype)
    vi_video = torch.rand(1, 3, T, H, W).to(device, dtype=weight_dtype)
    
    print(f"  IR 视频: {ir_video.shape}, 范围: [{ir_video.min():.2f}, {ir_video.max():.2f}]")
    print(f"  VI 视频: {vi_video.shape}, 范围: [{vi_video.min():.2f}, {vi_video.max():.2f}]")
    
    with torch.no_grad():
        # Step 5.1: 像素空间融合
        print("\n  [5.1] 像素空间融合...")
        fused_video = model_full.pixel_fusion(ir_video, vi_video)  # [B, C, T, H, W], [0, 1]
        print(f"    融合结果: {fused_video.shape}, 范围: [{fused_video.min():.2f}, {fused_video.max():.2f}]")
        
        # Step 5.2: VAE 编码
        print("\n  [5.2] VAE 编码...")
        fused_norm = fused_video * 2 - 1  # [-1, 1]
        
        # 编码
        video_flat = rearrange(fused_norm, "b c t h w -> (b t) c h w")
        latents_list = []
        for ind in range(0, video_flat.shape[0], 1):
            latents_list.append(vae.encode(video_flat[ind:ind+1]).latent_dist.sample())
        latents = torch.cat(latents_list, dim=0)
        fused_latent = rearrange(latents, "(b t) c h w -> b c t h w", t=T)
        fused_latent = fused_latent * vae.config.scaling_factor
        
        print(f"    Latent: {fused_latent.shape}")
        
        # Step 5.3: 创建条件 (模拟 CLIP 输出)
        print("\n  [5.3] 创建条件...")
        y = torch.zeros(1, 77, 1024, device=device, dtype=weight_dtype)  # 模拟 CLIP 输出
        
        # Step 5.4: 扩散采样
        print(f"\n  [5.4] 扩散采样 ({args.num_steps} 步)...")
        
        # 添加噪声
        t_start = torch.LongTensor([999]).to(device)
        noised = diffusion.diffuse(fused_latent, t_start)
        
        # DDPM 采样 - 使用 autocast 确保类型一致
        xt = noised.to(weight_dtype)
        fused_latent_half = fused_latent.to(weight_dtype)
        timesteps = torch.linspace(999, 0, args.num_steps, device=device).long()
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(1)
            
            with torch.cuda.amp.autocast(dtype=weight_dtype):
                v_pred = model_full.controlled_unet(
                    x=xt,
                    t=t_batch,
                    y=y,
                    hint=fused_latent_half
                )
            
            x0_pred = diffusion.get_x0(v=v_pred, xt=xt, t=t_batch)
            
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = diffusion.alphas[t.item()].view(1, 1, 1, 1, 1).to(device, dtype=weight_dtype)
                alpha_next = diffusion.alphas[t_next.item()].view(1, 1, 1, 1, 1).to(device, dtype=weight_dtype)
                sigma_t = diffusion.sigmas[t.item()].view(1, 1, 1, 1, 1).to(device, dtype=weight_dtype)
                sigma_next = diffusion.sigmas[t_next.item()].view(1, 1, 1, 1, 1).to(device, dtype=weight_dtype)
                
                eps = (xt - alpha_t * x0_pred) / (sigma_t + 1e-8)
                xt = alpha_next * x0_pred + sigma_next * eps
            else:
                xt = x0_pred
            
            if (i + 1) % 5 == 0 or i == len(timesteps) - 1:
                print(f"    步骤 {i + 1}/{args.num_steps} 完成")
        
        # Step 5.5: VAE 解码
        print("\n  [5.5] VAE 解码...")
        z = rearrange(xt, "b c t h w -> (b t) c h w")
        z = z.to(weight_dtype)  # 确保与 VAE 权重类型一致
        video_out = []
        for ind in range(0, z.shape[0], 3):
            num_f = min(3, z.shape[0] - ind)
            z_batch = z[ind:ind+num_f] / vae.config.scaling_factor
            video_out.append(vae.decode(z_batch, num_frames=num_f).sample)
        video_out = torch.cat(video_out)  # [T, C, H, W]
        
        # 反归一化
        video_out = (video_out + 1) / 2
        video_out = torch.clamp(video_out, 0, 1)
        
        print(f"    输出视频: {video_out.shape}, 范围: [{video_out.min():.2f}, {video_out.max():.2f}]")
    
    # 保存结果
    print(f"\n  保存结果到: {args.output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.', exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, 30, (W, H))
    
    for t_idx in range(video_out.shape[0]):
        frame = video_out[t_idx].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"  视频已保存！")
    
    # 清理
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)


def test_multi_scale_fusion():
    """
    测试新的多尺度融合架构
    """
    print("\n" + "=" * 70)
    print("多尺度融合架构测试")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 测试参数
    B, C, T = 1, 4, 8  # latent space
    H, W = 256, 256  # latent 尺寸 (注意：如果原图是1024x1024，latent应该是128x128)
    
    # ===== 1. 测试 IRLatentEncoder =====
    print("\n[1/4] 测试 IRLatentEncoder...")
    
    ir_encoder = IRLatentEncoder(in_channels=4, use_rtb=True).to(device)
    ir_latent = torch.randn(B * T, C, H, W).to(device)
    
    with torch.no_grad():
        ir_features = ir_encoder(ir_latent)
    
    print(f"  输入: {ir_latent.shape}")
    print(f"  输出 (4个尺度):")
    for i, feat in enumerate(ir_features):
        print(f"    Level {i}: {feat.shape}")
    
    # ===== 2. 测试 MultiScaleFusionBlock =====
    print("\n[2/4] 测试 MultiScaleFusionBlock...")
    
    fusion_block = MultiScaleFusionBlock(ctrl_channels=320, ir_channels=320).to(device)
    ctrl_feat = torch.randn(B * T, 320, H, W).to(device)
    ir_feat = torch.randn(B * T, 320, H, W).to(device)
    
    with torch.no_grad():
        fused = fusion_block(ctrl_feat, ir_feat)
    
    print(f"  ControlNet 特征: {ctrl_feat.shape}")
    print(f"  IR 特征: {ir_feat.shape}")
    print(f"  融合输出: {fused.shape}")
    
    # ===== 3. 测试 MultiScaleControlFusion =====
    print("\n[3/4] 测试 MultiScaleControlFusion...")
    
    ms_fusion = MultiScaleControlFusion().to(device)
    
    # 模拟 ControlNet 输出 (13个)
    ctrl_signals = []
    ctrl_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280]
    scales = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.125, 0.125, 0.125]
    for ch, scale in zip(ctrl_channels, scales):
        h_s, w_s = int(H * scale), int(W * scale)
        ctrl_signals.append(torch.randn(B * T, ch, h_s, w_s).to(device))
    
    print(f"  ControlNet 输出数量: {len(ctrl_signals)}")
    print(f"  IR 特征数量: {len(ir_features)}")
    
    with torch.no_grad():
        fused_signals = ms_fusion(ctrl_signals, ir_features)
    
    print(f"  融合后输出数量: {len(fused_signals)}")
    for i, fused in enumerate(fused_signals):
        print(f"    Signal {i}: {fused.shape}")
    
    # ===== 4. 参数统计 =====
    print("\n[4/4] 参数统计...")
    
    print(f"  IRLatentEncoder: {sum(p.numel() for p in ir_encoder.parameters()) / 1e6:.2f}M")
    print(f"  MultiScaleControlFusion: {sum(p.numel() for p in ms_fusion.parameters()) / 1e6:.2f}M")
    
    if device.type == 'cuda':
        print(f"\n  GPU 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("\n" + "=" * 70)
    print("多尺度融合测试完成！")
    print("=" * 70)


def test_dual_modal_v2v_unet():
    """
    测试完整的 DualModalV2VUNet
    """
    print("\n" + "=" * 70)
    print("DualModalV2VUNet 完整测试")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 注意: 这个测试需要较大显存
    print("\n⚠️ 注意: 此测试需要约 20GB+ GPU 显存")
    print("  如果显存不足，请使用 --quick_test 参数")
    
    # 创建模型 (不加载预训练权重，只测试结构)
    print("\n创建模型...")
    model = DualModalV2VUNet(use_rtb=True, pretrained_model_path=None)
    model.to(device)
    model.half()  # 使用 FP16 节省显存
    model.eval()
    
    # 测试输入
    B, C, T = 1, 4, 4  # 减少帧数节省显存
    H, W = 256, 512
    
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)
    t = torch.LongTensor([500]).to(device)
    y = torch.randn(B, 77, 1024, device=device, dtype=torch.float16)
    ir_latent = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)
    vi_latent = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)
    
    print(f"\n输入:")
    print(f"  x (noised latent): {x.shape}")
    print(f"  t (timestep): {t.shape}")
    print(f"  y (condition): {y.shape}")
    print(f"  ir_latent: {ir_latent.shape}")
    print(f"  vi_latent: {vi_latent.shape}")
    
    # 前向传播
    print("\n前向传播...")
    try:
        with torch.no_grad():
            output = model(x, t, y, ir_latent, vi_latent)
        
        print(f"\n输出: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n总参数: {total_params / 1e6:.2f}M")
        
        if device.type == 'cuda':
            print(f"GPU 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        print("\n✓ DualModalV2VUNet 测试通过！")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("  请使用更小的测试参数或更大显存的 GPU")
        else:
            raise e
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    # 检查是否只运行快速测试
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick_test", action="store_true", help="快速测试（不加载大模型）")
    parser.add_argument("--test_multiscale", action="store_true", help="测试多尺度融合模块")
    parser.add_argument("--test_full", action="store_true", help="测试完整 DualModalV2VUNet")
    args, _ = parser.parse_known_args()
    
    if args.test_multiscale:
        # 测试多尺度融合模块
        test_multi_scale_fusion()
    elif args.test_full:
        # 测试完整模型
        test_dual_modal_v2v_unet()
    elif args.quick_test:
        # 快速测试 - 只测试融合模块和损失函数
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        print("\n" + "="*70)
        print("快速测试模式 - 仅测试像素空间融合模块")
        print("="*70)
        
        fusion = PixelSpaceFusionModule(in_channels=3, embed_dim=32, num_blocks=2).to(device)
        ir = torch.randn(1, 3, 8, 256, 256).to(device)
        vi = torch.randn(1, 3, 8, 256, 256).to(device)
        
        with torch.no_grad():
            fused = fusion(ir, vi)
        
        print(f"融合测试: IR {ir.shape} + VI {vi.shape} -> Fused {fused.shape}")
        print_model_info(fusion)
        
        loss_fn = DualModalFusionLoss().to(device)
        losses = loss_fn(fused, ir, vi)
        print("\n损失:")
        for k, v in losses.items():
            print(f"  {k}: {v.item():.4f}")
        
        print("\n" + "="*70)
        print("快速测试完成！")
        print("="*70)
    else:
        # 默认: 先测试多尺度融合，再运行完整推理测试
        print("运行默认测试流程...")
        print("使用 --quick_test 进行快速测试")
        print("使用 --test_multiscale 测试多尺度融合模块")
        print("使用 --test_full 测试完整 DualModalV2VUNet")
        print()
        
        # 先测试多尺度融合
        test_multi_scale_fusion()


"""
先看一下可见光在 输入Vid2VidSDUNet之前的形状是什么样子？有无多尺度
    双模态视频融合模型 - 完整系统（无gt）
 输入:
├── 可见光 ─────────────────┐
├                           ──> VAE 编码 (冻结)
└── 可见光参考图像 ──> CLIP 图像编码器 (冻结)│

训练流程:
┌─────────────────────────────────────────────────────────────────────┐
│                        ControlledV2VUNet  （输入可见光）             │
│  ┌────────────────────┐                                             │
│  │  VideoControlNet   │ ← 🔥 训练                                   │
│  │  (处理 可见光参考图像 hint)    │                                  │
│  │  - input_blocks    │                                             │
│  │  - zero_convs      │                                             │
│  │  - middle_block    │                                             │
│  │  - input_hint_block│                                             │
│  └─────────┬──────────┘                                             │
│            │                                                        │
│            ↓      
ir->rtb ->PixelSpaceFusionModule                                      | (rtb我暂时没想好，其实就是特征处理的模块，期间经过rtb的bchw都不会变化预期和controlnet输出的红外特征维度是一样的，不过如果controlnet有多尺度，我们处理也要有多尺度，这个多尺度也容易，因为rtb不改变大小，所以加个downsample就行)
               │
               │ 控制信号  （由clip给出）
                ↓ 
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Vid2VidSDUNet (主UNet)                          │    │
│  │  - input_blocks ← ❄️ 冻结 (但 local 层 🔥 训练)                │    │
│  │  - middle_block ← ❄️ 冻结 (但 local 层 🔥 训练)  

              │    │
│  │  - output_blocks ← ❄️ 冻结 (但 local 层 🔥 训练)               │    │
│  │                                                              │    │
│  │  控制信号注入方式:                                             │    │
│  │  x = control.pop() + x  (middle)                             │    │
│  │  x = xs.pop() + control.pop() (decoder skip connections)     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                         VAE 解码 (冻结)
                              ↓
                         输出视频
"""
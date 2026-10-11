#!/usr/bin/env python
# coding=utf-8
"""
训练脚本 — VI-conditioned image-to-image latent diffusion

训练和推理使用同一个前向过程：
  x_t = sqrt(ᾱ_t) z_VI + sqrt(1-ᾱ_t) ε
网络直接预测融合 latent z_F。红外是唯一的 ControlNet 条件，残差用光流对齐。
VAE 和 RAFT 冻结。UNet 与红外 ControlNet 全量训练。这一阶段不接 TemporalRestormer。
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 缓解显存碎片（大激活反传时有效）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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
from einops import rearrange
from tqdm.auto import tqdm

import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(this_dir, '..'))
sys.path.insert(0, base_path)
sys.path.insert(0, this_dir)

from dual_modal_fusion_model import VideoControlNetModel
from train_latent_vae48_tem import replace_vae_to_4x

logger = get_logger(__name__)

# 融合辅助损失的内部比例（固定，不做调参入口）
LOSS_MAX_RATIO = 4.0
LOSS_SSIM_RATIO = 1.0
LOSS_SSIM_VIS_RATIO = 1.0
LOSS_SSIM_IR_RATIO = 1.0
LOSS_COLOR_RATIO = 12.0
LOSS_GRAD_RATIO = 10.0
# 使用 Restormer 运动去模糊预训练时，默认先做 x1 清晰化更稳。
SR_UPSCALE = 1

RESTORMER_ROOT = os.path.join(base_path, "Fusion_SGFE", "superresulation", "Restormer")
if RESTORMER_ROOT not in sys.path:
    sys.path.append(RESTORMER_ROOT)
# 直接把 archs 目录加入 sys.path，以顶层模块方式导入，绕开 basicsr.models.__init__
# 的重型依赖链（其中 create_lmdb 需要未必安装的 lmdb 包）。
_RESTORMER_ARCHS = os.path.join(RESTORMER_ROOT, "basicsr", "models", "archs")
if _RESTORMER_ARCHS not in sys.path:
    sys.path.append(_RESTORMER_ARCHS)
try:
    from restormer_temporal_arch import TemporalRestormer
except Exception as _e:
    logger.warning(f"TemporalRestormer 导入失败: {_e}")
    TemporalRestormer = None


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
        self, vi_dir, ir_dir,
        num_frames=8, window_stride=2, vi_per_ir=8,
        video_size=(256, 256), ir_select_mode="start", ir_middle_local_idx=None,
        gt_dir=None,
    ):
        self.vi_dir = Path(vi_dir)
        self.ir_dir = Path(ir_dir)
        self.gt_dir = None if gt_dir is None else Path(gt_dir)
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
            f"video_size={self.video_size}, gt={'on' if self.gt_dir is not None else 'off'}"
        )

    def _window_starts(self, length):
        if length < self.num_frames:
            return []
        if self.window_stride >= self.num_frames:
            starts = list(range(0, length - self.num_frames + 1, self.num_frames))
        else:
            starts = list(range(0, length - self.num_frames + 1, self.window_stride))
            if starts and starts[-1] != length - self.num_frames:
                starts.append(length - self.num_frames)
        return starts

    def _collect_samples(self):
        if self.gt_dir is not None:
            return self._collect_samples_with_gt()
        samples = []
        if not (self.vi_dir.exists() and self.ir_dir.exists()):
            return samples

        vi_subdirs = [d for d in sorted(self.vi_dir.iterdir()) if d.is_dir()]
        if vi_subdirs:
            folder_pairs = []
            for vi_sub in vi_subdirs:
                ir_sub = self.ir_dir / vi_sub.name
                if ir_sub.exists() and ir_sub.is_dir():
                    folder_pairs.append((vi_sub, ir_sub))
        else:
            folder_pairs = [(self.vi_dir, self.ir_dir)]

        for vi_sub, ir_sub in folder_pairs:
            if not (vi_sub.exists() and ir_sub.exists()):
                continue

            vi_files = sorted(list(vi_sub.glob("*.png")) + list(vi_sub.glob("*.jpg")))
            ir_files = sorted(list(ir_sub.glob("*.png")) + list(ir_sub.glob("*.jpg")))
            if len(vi_files) < self.num_frames or len(ir_files) == 0:
                continue

            vi_seq = vi_files
            starts = self._window_starts(len(vi_seq))
            if not starts:
                continue

            samples.append({
                "vi_seq": vi_seq,
                "ir_seq": ir_files,
                "starts": starts,
            })
        return samples

    def _collect_samples_with_gt(self):
        samples = []
        if not (self.vi_dir.exists() and self.ir_dir.exists() and self.gt_dir.exists()):
            return samples

        gt_subdirs = [d for d in sorted(self.gt_dir.iterdir()) if d.is_dir()]
        if gt_subdirs:
            triples = []
            for gt_sub in gt_subdirs:
                vi_sub = self.vi_dir / gt_sub.name
                ir_sub = self.ir_dir / gt_sub.name
                if vi_sub.is_dir() and ir_sub.is_dir():
                    triples.append((vi_sub, ir_sub, gt_sub))
        else:
            triples = [(self.vi_dir, self.ir_dir, self.gt_dir)]

        for vi_sub, ir_sub, gt_sub in triples:
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

            starts = self._window_starts(len(vi_seq))
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
            "ir_anchor_idx": torch.tensor(int(dummy_anchor), dtype=torch.long),
        }
        if self.gt_dir is not None:
            dummy["gt_video"] = torch.randn(3, self.num_frames, *self.video_size)
        if not self.samples:
            return dummy

        sample = self.samples[idx % len(self.samples)]
        start = sample["starts"][np.random.randint(0, len(sample["starts"]))]
        indices = np.arange(start, start + self.num_frames)

        vi_frames = [_load_image_as_gray3(sample["vi_seq"][i], self.video_size) for i in indices]
        vi_video = torch.from_numpy(np.transpose(np.stack(vi_frames), (3, 0, 1, 2))).float()
        gt_video = None
        if "gt_seq" in sample:
            gt_frames = [_load_image_as_gray3(sample["gt_seq"][i], self.video_size) for i in indices]
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

        item = {
            "vi_video": vi_video,
            "ir_video": ir_video,
            "ir_anchor_idx": torch.tensor(int(ir_local_idx), dtype=torch.long),
        }
        if gt_video is not None:
            item["gt_video"] = gt_video
        return item


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


def _load_sr_pretrained(module, path):
    """加载后置 SR 预训练权重（支持 .pth/.pt/.safetensors）。"""
    if not path:
        return
    weight_path = path
    if not os.path.isabs(weight_path):
        candidate = os.path.join(base_path, weight_path)
        if os.path.exists(candidate):
            weight_path = candidate
    if not os.path.exists(weight_path):
        logger.warning(f"SR 预训练权重不存在，跳过: {path}")
        return

    state = None
    try:
        if weight_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(weight_path)
        else:
            raw = torch.load(weight_path, map_location="cpu")
            if isinstance(raw, dict):
                # 兼容常见封装键
                for key in ("params_ema", "params", "state_dict", "model"):
                    if key in raw and isinstance(raw[key], dict):
                        state = raw[key]
                        break
                if state is None:
                    # 若本身就是 state_dict（value 多为 tensor）
                    if all(torch.is_tensor(v) for v in raw.values()):
                        state = raw
            if state is None:
                raise RuntimeError("无法从该权重文件解析 state_dict")

        # 去掉 DataParallel 前缀 + 兼容原版 RFDN/Restormer 键名
        def _remap_rfdn_key(k):
            mapping_prefix = {
                "B1.": "blocks.0.",
                "B2.": "blocks.1.",
                "B3.": "blocks.2.",
                "B4.": "blocks.3.",
                "c.": "fusion_1x1.",
                "LR_conv.": "lr_conv.",
                "upsampler.0.": "up.0.",
            }
            for src, dst in mapping_prefix.items():
                if k.startswith(src):
                    return dst + k[len(src):]
            return k

        clean = {}
        module_state = module.state_dict()
        shape_mismatch = 0
        for k, v in state.items():
            key = k[7:] if k.startswith("module.") else k
            key_rfdn = _remap_rfdn_key(key)
            # Restormer 原权重一般是裸键名，当前适配器里在 restormer 子模块下
            candidates = [key_rfdn, key, f"restormer.{key}", f"restormer.{key_rfdn}"]
            target_key = None
            for cand in candidates:
                if cand in module_state:
                    target_key = cand
                    break
            if target_key is None:
                continue
            if module_state[target_key].shape != v.shape:
                shape_mismatch += 1
                continue
            clean[target_key] = v

        ret = module.load_state_dict(clean, strict=False)
        logger.info(
            f"SR 预训练加载完成({weight_path}): loaded={len(clean)}, missing={len(ret.missing_keys)}, "
            f"unexpected={len(ret.unexpected_keys)}, shape_mismatch_skipped={shape_mismatch}"
        )
    except Exception as e:
        logger.warning(f"SR 预训练加载失败，继续随机初始化训练: {e}")


def _to_01(x):
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def _rgb2gray(x):
    if x.shape[1] == 1:
        return x
    return (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )


def _rgb_to_ycbcr(x):
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], dim=1)


def _gaussian(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window(window_size, channel, device, dtype):
    g = _gaussian(window_size, 1.5, device, dtype).unsqueeze(1)
    window_2d = (g @ g.transpose(0, 1)).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def _ssim_loss(img1, img2, window_size=24):
    (_, channel, height, width) = img1.size()
    real_size = min(window_size, height, width)
    window = _create_window(real_size, channel, img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, groups=channel)
    mu2 = F.conv2d(img2, window, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return 1.0 - ssim_map.mean()


def _spatial_grad_xy(gray):
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device, dtype=gray.dtype
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=gray.device, dtype=gray.dtype
    ).view(1, 1, 3, 3)
    gray_pad = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = torch.abs(F.conv2d(gray_pad, sobel_x))
    gy = torch.abs(F.conv2d(gray_pad, sobel_y))
    return gx, gy


def fusion_unsupervised_loss(
    vi_img, ir_img, pred_img,
    max_ratio=4.0,
    ssim_vis_ratio=1.0,
    ssim_ir_ratio=1.0,
    ssim_ratio=1.0,
    color_ratio=12.0,
    grad_ratio=10.0,
):
    vi_gray = _rgb2gray(vi_img)
    ir_gray = _rgb2gray(ir_img)
    pred_gray = _rgb2gray(pred_img)

    pick_ir_mask = (ir_gray > vi_gray).float()
    target_intensity = pick_ir_mask * ir_img + (1.0 - pick_ir_mask) * vi_img
    loss_max = F.l1_loss(pred_img, target_intensity)

    loss_ssim = (
        ssim_vis_ratio * _ssim_loss(vi_img, pred_img)
        + ssim_ir_ratio * _ssim_loss(ir_gray, pred_gray)
    )

    vi_ycbcr = _rgb_to_ycbcr(vi_img)
    pred_ycbcr = _rgb_to_ycbcr(pred_img)
    loss_color = F.l1_loss(vi_ycbcr[:, 1:], pred_ycbcr[:, 1:])

    vi_gx, vi_gy = _spatial_grad_xy(vi_gray)
    ir_gx, ir_gy = _spatial_grad_xy(ir_gray)
    pred_gx, pred_gy = _spatial_grad_xy(pred_gray)
    loss_grad = F.l1_loss(pred_gx, torch.max(vi_gx, ir_gx)) + F.l1_loss(pred_gy, torch.max(vi_gy, ir_gy))

    total = (
        max_ratio * loss_max
        + ssim_ratio * loss_ssim
        + color_ratio * loss_color
        + grad_ratio * loss_grad
    )
    return total, loss_ssim, loss_max, loss_color, loss_grad


def save_validation_triplet(vi_video, ir_video, pred_video, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)
    h, w = pred_video.shape[-2], pred_video.shape[-1]
    vi_01 = _to_01(vi_video)
    ir_01 = _to_01(ir_video)
    if vi_01.shape[-2] != h or vi_01.shape[-1] != w:
        vi_01 = F.interpolate(
            vi_01[:, :, 0], size=(h, w), mode="bilinear", align_corners=False
        ).unsqueeze(2)
        ir_01 = F.interpolate(
            ir_01[:, :, 0], size=(h, w), mode="bilinear", align_corners=False
        ).unsqueeze(2)
    vi = vi_01[0, :, 0].permute(1, 2, 0).cpu().numpy()
    ir = ir_01[0, :, 0].permute(1, 2, 0).cpu().numpy()
    pred = pred_video[0, :, 0].permute(1, 2, 0).cpu().numpy()
    grid = np.concatenate([vi, ir, pred], axis=1)
    cv2.imwrite(os.path.join(save_dir, f"step_{step:06d}.png"), (np.clip(grid, 0, 1) * 255.0).astype(np.uint8))


def decode_video(model, latent, num_frames):
    """VAE 解码到 [0,1]。保留梯度，让融合损失能回到噪声预测。"""
    batch = latent.shape[0]
    video = model.vae_decode_chunk(latent, chunk_size=num_frames)
    video = rearrange(video, "(b t) c h w -> b c t h w", b=batch, t=num_frames)
    return torch.clamp((video + 1.0) * 0.5, 0.0, 1.0)


def decode_and_enhance(model, sr_module, latent, num_frames):
    """统一的重建尾部：VAE 解码(冻结) + TemporalRestormer 增强。

    - VAE 解码在 no_grad 内（扩散与 VAE 始终冻结，省显存）。
    - 增强在外层梯度上下文中执行：训练时可回传到 sr_module，推理时整体 no_grad。
    latent: (B, C, T, H/4, W/4) -> 返回 (B, C, T, H, W)，范围 [0,1]
    """
    B = latent.shape[0]
    with torch.no_grad():
        video = model.vae_decode_chunk(latent, chunk_size=num_frames)
        video = rearrange(video, '(b t) c h w -> b c t h w', b=B, t=num_frames)
        video = torch.clamp((video + 1.0) * 0.5, 0.0, 1.0)
    return sr_module(video)


def build_temporal_restormer(scale=1):
    """构造内置时序能力的 Restormer（解码尾部增强头）。

    - 复用 Motion Deblurring 预训练权重（2D 子模块同名直接加载）。
    - 时序 3D 卷积内置在 Restormer 内部，恒等初始化，初始不改变其输出。
    """
    if TemporalRestormer is None:
        raise ImportError("无法导入 TemporalRestormer，请检查 superresulation/Restormer 代码路径")
    if int(scale) != 1:
        raise NotImplementedError("当前 TemporalRestormer 仅支持 x1（清晰化/增强，不放大）")
    return TemporalRestormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
        use_temporal_high=True,
        use_grad_checkpoint=True,  # 训练时重算激活省显存（512²×多帧反传必须）
    )


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
        default=os.path.join(base_path, "pretrained_weight", "vae_4x_tvt.ckpt"),
    )
    parser.add_argument(
        "--raft_model_path", type=str,
        default=os.path.join(this_dir, "ProPainter", "weights", "raft-things.pth"),
    )

    # 数据
    parser.add_argument(
        "--vi_data_dir", type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\visible1",
    )
    parser.add_argument(
        "--ir_data_dir", type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\infrared",
    )
    parser.add_argument(
        "--gt_data_dir", type=str,
        default=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD\Fusion",
        help="GT fusion 图像目录，编码后作为被加噪的扩散变量",
    )
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="验证时的 DDIM 步数")
    parser.add_argument("--denoise_strength", type=float, default=1.0,
                        help="推理起始噪声比例。1 从最吵的一步开始，和训练的前向过程一致")
    parser.add_argument("--lambda_int", type=float, default=0.1)
    parser.add_argument("--lambda_ssim", type=float, default=0.1)
    parser.add_argument("--lambda_grad", type=float, default=0.1)

    # 训练
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--video_size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--window_stride", type=int, default=8, help="训练窗口步长；默认8表示非重叠分组，<8为滑动")
    parser.add_argument("--vi_per_ir", type=int, default=8)
    parser.add_argument("--ir_select_mode", type=str, default="start",
                        choices=["middle", "random", "start"])
    parser.add_argument("--ir_middle_local_idx", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--sr_pretrained_path", type=str,
                        default=r"Fusion_SGFE/superresulation/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth",
                        help="后置 SR 适配器预训练权重路径(.pth/.pt/.safetensors)")

    # 输出
    parser.add_argument(
        "--output_dir", type=str,
        default=r"Fusion_SGFE/train_one_step",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=1500)
    parser.add_argument("--validation_steps", type=int, default=100)

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
        vae_path = None

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
    model.controlnet_vi.load_state_dict(model.video_controlnet.state_dict())

    replace_vae_to_4x(model, args.vae_4x_ckpt)
    # 显式再次冻结，避免替换 VAE 后出现 requires_grad 漂移。
    model.vae.eval()
    model.vae.requires_grad_(False)

    # UNet 和红外 ControlNet 全量训练。可见光已经在 x_t 里，不训练第二条 ControlNet。
    # VAE、RAFT 冻结。
    model.freeze_for_training(train_controlnet=True, train_unet=True, train_hfp_only=False)
    for param in model.controlnet_vi.parameters():
        param.requires_grad = False
    model.controlnet_vi.eval()
    model.controlnet_vi.cpu()
    model.train_mode()
    if model.ir_flow_prop is not None:
        model.ir_flow_prop.eval()
        model.ir_flow_prop.requires_grad_(False)

    if accelerator.is_main_process:
        trainable_diffusion = sum(p.numel() for p in model.get_trainable_parameters())
        logger.info(f"可训练参数: diffusion={trainable_diffusion / 1e6:.2f}M")

    dataset = SlidingRatioVideoDataset(
        args.vi_data_dir, args.ir_data_dir,
        num_frames=args.num_frames,
        window_stride=args.window_stride,
        vi_per_ir=args.vi_per_ir,
        video_size=tuple(args.video_size),
        ir_select_mode=args.ir_select_mode,
        ir_middle_local_idx=args.ir_middle_local_idx,
        gt_dir=args.gt_data_dir,
    )
    if len(dataset.samples) == 0:
        raise RuntimeError(f"没有配上 VI/IR/GT 样本: {args.gt_data_dir}")
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

    (
        model.unet, model.video_controlnet,
        optimizer, dataloader, lr_scheduler,
    ) = accelerator.prepare(
        model.unet, model.video_controlnet,
        optimizer, dataloader, lr_scheduler,
    )

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
        print("扩散状态:   带噪 VI latent，训练和推理相同")
        print(f"推理起点:   strength={args.denoise_strength}")
        print("条件:       红外 ControlNet，光流对齐残差")
        print("预测:       融合 latent z_F")
        print("冻结:       VAE / RAFT。UNet 与红外 ControlNet 全量训练")
        print(f"GT fusion:  {args.gt_data_dir}")
        print("后处理:     这一阶段不使用 TemporalRestormer")
        print("Loss配置:")
        print("  L = ||z_hat - z_F||^2")
        print("=" * 60)

    global_step = args.start_step
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step, total=args.max_train_steps,
        desc="Steps", disable=not accelerator.is_local_main_process,
    )

    epoch = 0
    num_train_timesteps = int(model.train_scheduler.config.num_train_timesteps)
    accum_models = (model.unet, model.video_controlnet)
    while global_step < args.max_train_steps:
        model.train_mode()
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(*accum_models):
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
                    fusion_latent = model.vae_encode(gt.permute(0, 2, 1, 3, 4)).float()

                vi_frames = vi.permute(0, 2, 1, 3, 4).contiguous() \
                    if model.ir_flow_prop is not None else None

                timesteps = torch.randint(
                    0, num_train_timesteps, (B,),
                    device=accelerator.device, dtype=torch.long,
                )
                noise = torch.randn_like(vi_latent)
                x_t = model.train_scheduler.add_noise(vi_latent, noise, timesteps)

                with accelerator.autocast():
                    z_pred = model.forward_diffusion(
                        x_t, timesteps, vi_latent, ir_latent,
                        vi_frames=vi_frames,
                        ir_anchor_idx=sample_anchor,
                    ).float()
                loss_z = F.mse_loss(z_pred, fusion_latent)
                loss = loss_z

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
                    "loss_z": loss_z.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    z=f"{loss_z.item():.4f}",
                )

                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    try:
                        model.eval_mode()
                        with torch.no_grad():
                            gen_latent = model.sample_diffusion(
                                vi_latent, ir_latent,
                                num_steps=args.num_inference_steps,
                                vi_frames=vi_frames,
                                ir_anchor_idx=sample_anchor,
                                strength=args.denoise_strength,
                            )
                            vis_video = decode_video(model, gen_latent, args.num_frames)
                        model.train_mode()
                        save_validation_triplet(
                            vi.detach(), ir.detach(), vis_video.detach(),
                            os.path.join(args.output_dir, "validations"), global_step,
                        )
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        model.train_mode()

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

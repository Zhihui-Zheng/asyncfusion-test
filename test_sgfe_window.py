#!/usr/bin/env python
# coding=utf-8
"""
推理脚本 — Fusion_SGFE（前段偏置融合权重版）

与 test_sgfe.py 的唯一区别：
    pos_weights 从中心高→两端低（center-biased）
    改为前段高→后段低（front-biased），STRIDE=2 时按两帧一组避免奇偶闪烁：

        pos:     0     1     2     3     4     5     6     7
        旧权重:  0.125 0.375 0.625 0.875 0.875 0.625 0.375 0.125
        新权重:  1.00  1.00  0.60  0.60  0.30  0.30  0.10  0.10

用法:
  python Fusion_SGFE/test_sgfe_window.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import re
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from einops import rearrange

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from Fusion_SGFE.dual_modal_fusion_model import VideoControlNetModel
from Fusion_SGFE.train_latent_vae48_tem import replace_vae_to_4x
from Fusion_SGFE.train_stage2 import LatentRestormer

DEFAULT_CONTROLNET_CHECKPOINT = r"E:\paper2\codebase\STAR\outputs\33_vae4x\checkpoint-10000\model.safetensors"
DEFAULT_UNET_CHECKPOINT = r"E:\paper2\codebase\STAR\Ablation1\Fusion_SGFE\full_loss\checkpoint-3000\model.safetensors"
DEFAULT_LATENT_NET_CHECKPOINT = r"E:\paper2\codebase\STAR\Fusion_SGFE\train_stage2\checkpoint-1500\model.safetensors"
DEFAULT_RAFT_MODEL_PATH = r"Fusion_SGFE/ProPainter/weights/raft-things.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="推理: Fusion_SGFE (front-biased window)")

    # ── 权重路径 ──
    parser.add_argument("--controlnet_checkpoint", type=str,
                        default=DEFAULT_CONTROLNET_CHECKPOINT)
    parser.add_argument("--unet_checkpoint", type=str,
                        default=DEFAULT_UNET_CHECKPOINT)
    parser.add_argument("--ir_select_mode", type=str, default="first",
                        choices=["middle", "first"],
                        help="IR 选取策略：middle=窗口中间帧，first=窗口第一帧")
    parser.add_argument("--raft_model_path", type=str,
                        default=DEFAULT_RAFT_MODEL_PATH)

    # ── 数据 ──
    parser.add_argument("--vi_root_dir", type=str,
                        default=r"E:\paper2\downstream\Objecttracking\ObjectTrack\visible")
    parser.add_argument("--ir_root_dir", type=str,
                        default=r"E:\paper2\downstream\Objecttracking\ObjectTrack\infrared")

    # ── LatentRestormer（第二阶段增强） ──
    parser.add_argument("--latent_net_checkpoint", type=str,
                        default=DEFAULT_LATENT_NET_CHECKPOINT,
                        help="train_stage2 保存的 LatentRestormer 权重；留空则跳过增强")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--latent_blocks", type=int, default=8)
    parser.add_argument("--latent_heads", type=int, default=4)

    # ── 输出 ──
    parser.add_argument("--output_dir", type=str,
                        default=r"E:\paper2\downstream\ablation_result\full_window")

    # ── 推理设置 ──
    parser.add_argument("--video_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--time_shift", type=float, default=3.0)
    parser.add_argument("--vi_per_ir", type=int, default=8)
    parser.add_argument("--input_interp", type=str, default="area",
                        choices=["area", "linear", "cubic", "lanczos4"])
    parser.add_argument("--output_interp", type=str, default="lanczos4",
                        choices=["lanczos4", "cubic", "linear", "area"])

    return parser.parse_args()


_INTERP_MAP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def load_frames_by_indices(frame_files, indices, video_size, interp=cv2.INTER_AREA):
    first_img = cv2.imread(str(frame_files[indices[0]]))
    original_size = (first_img.shape[0], first_img.shape[1])
    gray_frames = []
    vi_ycrcb_list = []
    for i in indices:
        img = cv2.imread(str(frame_files[i]))
        img = cv2.resize(img, (video_size[1], video_size[0]), interpolation=interp)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        vi_ycrcb_list.append(ycrcb)
        y = ycrcb[:, :, 0]
        gray_3ch = np.stack([y, y, y], axis=-1).astype(np.float32) / 255.0 * 2 - 1
        gray_frames.append(gray_3ch)
    gray_frames = np.stack(gray_frames, axis=0)
    gray_frames = np.transpose(gray_frames, (3, 0, 1, 2))
    return torch.from_numpy(gray_frames).float(), vi_ycrcb_list, original_size


def load_ir_single_frame(frame_files, idx, num_frames, video_size, interp=cv2.INTER_AREA):
    if idx >= len(frame_files):
        idx = len(frame_files) - 1
    img = cv2.imread(str(frame_files[idx]))
    img = cv2.resize(img, (video_size[1], video_size[0]), interpolation=interp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0 * 2 - 1
    gray_3ch = np.transpose(gray_3ch, (2, 0, 1))
    ir_video = np.stack([gray_3ch] * num_frames, axis=1)
    return torch.from_numpy(ir_video).float()


def select_ir_index_for_window(
    vi_files, ir_files, vi_start_idx, num_frames=8, vi_per_ir=8, mode="first"
):
    """数据集 VI/IR 1:1 时序对齐，直接按帧索引取对应 IR 帧。"""
    if len(ir_files) == 0:
        return 0
    vi_start_idx = max(0, min(int(vi_start_idx), len(vi_files) - 1))
    local_offset = (int(num_frames) - 1) // 2 if mode == "middle" else 0
    vi_target_idx = vi_start_idx + local_offset
    return min(vi_target_idx, len(ir_files) - 1)


@torch.no_grad()
def run_inference(model, vi_video, ir_video, num_steps=5, time_shift=3.0, latent_net=None):
    model.eval_mode()
    B, C, T, H, W = vi_video.shape
    device = next(model.unet.parameters()).device
    vi_video_device = vi_video.to(device)
    ir_video_device = ir_video.to(device)
    vi_video_perm = vi_video_device.permute(0, 2, 1, 3, 4)
    ir_video_perm = ir_video_device.permute(0, 2, 1, 3, 4)
    vi_latent = model.vae_encode(vi_video_perm)
    ir_latent = model.vae_encode(ir_video_perm)
    vi_frames = vi_video_perm.float() if model.ir_flow_prop is not None else None
    precomputed_flows = None
    if model.ir_flow_prop is not None and vi_frames is not None:
        precomputed_flows = model.ir_flow_prop.estimate_flow(vi_frames)
    model.noise_scheduler.set_timesteps(num_steps, time_shift=time_shift, device=device)
    x = vi_latent.clone()
    for t_step in model.noise_scheduler.timesteps:
        t_batch = t_step.unsqueeze(0).repeat(B).to(device)
        v_pred = model.forward(
            x_t=x, t=t_batch, ir_latent=ir_latent,
            vi_frames=vi_frames, _precomputed_flows=precomputed_flows
        )
        x = model.noise_scheduler.step(v_pred, t_step, x)
    if latent_net is not None:
        x = latent_net(x.float())
    pred_video = model.vae_decode_chunk(x, chunk_size=T)
    pred_video = rearrange(pred_video, '(b t) c h w -> b c t h w', b=B, t=T)
    pred_video = torch.clamp((pred_video + 1) / 2, 0, 1)
    return pred_video


def save_single_frame_from_y(y_pred_uint8, save_dir, frame_idx, vi_ycrcb=None,
                              interp=cv2.INTER_LANCZOS4, output_name=None):
    os.makedirs(save_dir, exist_ok=True)
    if output_name is None:
        output_name = f"frame_{frame_idx:04d}.png"
    if vi_ycrcb is not None:
        ycrcb = vi_ycrcb.copy()
        ycrcb[:, :, 0] = y_pred_uint8
        color_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        color_bgr = cv2.cvtColor(y_pred_uint8, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(save_dir, output_name), color_bgr)


NUM_FRAMES = 8
STRIDE = 2
MAX_FRAMES = 80

# STRIDE=2：按两帧一组从前向后衰减，避免奇偶帧出现不同融合权重导致闪烁
POS_WEIGHTS = [1.00, 1.00, 0.60, 0.60, 0.30, 0.30, 0.10, 0.10]


def main():
    args = parse_args()
    print("=" * 70)
    print("推理: Fusion_SGFE（前段偏置融合权重，启用 IR 光流 warp）")
    print(f"  pos_weights = {POS_WEIGHTS}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1] 加载模型...")

    class _Opt:
        model_path = None
        vae_path = "stabilityai/stable-video-diffusion-img2vid"

    model = VideoControlNetModel(
        _Opt(), device=device,
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

    replace_vae_to_4x(model, r"pretrained_weight/vae_4x_tvt.ckpt")
    print("    VAE: TVT VAED4 (4x)")

    from safetensors.torch import load_file

    def _load_ckpt(module, path, name):
        if path and os.path.exists(path):
            raw_state = load_file(path)
            module_state = module.state_dict()
            state = {}
            shape_mismatch = 0
            for k, v in raw_state.items():
                key = k.replace('module.', '')
                key = key.replace('.hfp_2d.', '.sgfe_2d.')
                if key not in module_state:
                    continue
                if module_state[key].shape != v.shape:
                    shape_mismatch += 1
                    continue
                state[key] = v
            ret = module.load_state_dict(state, strict=False)
            print(f"    ✓ {name} 加载完成 (loaded={len(state)}, missing={len(ret.missing_keys)}, "
                  f"unexpected={len(ret.unexpected_keys)}, shape_mismatch_skipped={shape_mismatch})")
        elif path:
            print(f"    ⚠ {name} checkpoint 不存在: {path}")

    _load_ckpt(model.video_controlnet, args.controlnet_checkpoint, "ControlNet")
    _load_ckpt(model.unet, args.unet_checkpoint, "UNet")

    model.to_half()
    model.eval_mode()

    latent_net = None
    if args.latent_net_checkpoint and os.path.exists(args.latent_net_checkpoint):
        latent_net = LatentRestormer(
            latent_ch=4, dim=args.latent_dim,
            num_blocks=args.latent_blocks, num_heads=args.latent_heads,
        ).to(device).float().eval()
        from safetensors.torch import load_file as _load_file
        state = _load_file(args.latent_net_checkpoint)
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        missing = latent_net.load_state_dict(clean, strict=False)
        print(f"    ✓ LatentRestormer 加载完成 "
              f"(missing={len(missing.missing_keys)}, unexpected={len(missing.unexpected_keys)})")
    elif args.latent_net_checkpoint:
        print(f"    ⚠ LatentRestormer checkpoint 不存在: {args.latent_net_checkpoint}，跳过增强")

    print(f"    ✓ 模型准备完成")

    vi_root = Path(args.vi_root_dir)
    ir_root = Path(args.ir_root_dir)
    vi_subdirs = sorted([d for d in vi_root.iterdir() if d.is_dir()])
    matched_folders = [(d, ir_root / d.name) for d in vi_subdirs
                       if (ir_root / d.name).is_dir()]
    if not matched_folders:
        vi_imgs = sorted(list(vi_root.glob('*.png')) + list(vi_root.glob('*.jpg')))
        if vi_imgs:
            matched_folders = [(vi_root, ir_root)]

    print(f"\n[2] 找到 {len(matched_folders)} 个匹配文件夹")
    print(f"    输出目录: {args.output_dir}")

    in_interp = _INTERP_MAP[args.input_interp]
    out_interp = _INTERP_MAP[args.output_interp]
    video_size = tuple(args.video_size)
    total_frames_saved = 0

    print(f"\n[3] 开始批量推理...")
    for folder_idx, (vi_sub, ir_sub) in enumerate(matched_folders):
        folder_name = vi_sub.name
        print(f"\n[文件夹 {folder_idx+1}/{len(matched_folders)}] {folder_name}")

        vi_all = sorted(list(vi_sub.glob('*.png')) + list(vi_sub.glob('*.jpg')))
        ir_all = sorted(list(ir_sub.glob('*.png')) + list(ir_sub.glob('*.jpg')))
        vi_frame_files = vi_all[:MAX_FRAMES]
        ir_frame_files = ir_all[:MAX_FRAMES] if ir_all else []

        num_vi = len(vi_frame_files)
        if num_vi < NUM_FRAMES:
            print(f"  ⚠ 帧数不足 ({num_vi} < {NUM_FRAMES})，跳过")
            continue

        save_dir = os.path.join(args.output_dir, folder_name)
        expected_outputs = [os.path.join(save_dir, vi_frame_files[i].name) for i in range(num_vi)]
        if all(os.path.exists(p) for p in expected_outputs):
            print(f"  ⏭ {folder_name} 已全部处理完毕，跳过")
            total_frames_saved += num_vi
            continue

        windows = []
        for s in range(0, num_vi - NUM_FRAMES + 1, STRIDE):
            windows.append(list(range(s, s + NUM_FRAMES)))
        tail_start = num_vi - NUM_FRAMES
        if windows and windows[-1][0] != tail_start:
            windows.append(list(range(tail_start, tail_start + NUM_FRAMES)))

        os.makedirs(save_dir, exist_ok=True)

        frame_sum, frame_wsum, frame_ycrcb, frame_best_w, frame_last_window = {}, {}, {}, {}, {}
        for w_idx, win in enumerate(windows):
            for f_idx in win:
                frame_last_window[f_idx] = w_idx

        for w_idx, win in enumerate(tqdm(windows, desc=f"  {folder_name}", leave=True)):
            vi_video, vi_ycrcb_list, _ = load_frames_by_indices(
                vi_frame_files, win, video_size, interp=in_interp)
            vi_video = vi_video.unsqueeze(0)

            ir_idx = select_ir_index_for_window(
                vi_files=vi_frame_files,
                ir_files=ir_frame_files,
                vi_start_idx=win[0],
                num_frames=NUM_FRAMES,
                vi_per_ir=args.vi_per_ir,
                mode=args.ir_select_mode,
            )
            ir_video = load_ir_single_frame(
                ir_frame_files, ir_idx, NUM_FRAMES, video_size, interp=in_interp)
            ir_video = ir_video.unsqueeze(0)

            pred_video = run_inference(model, vi_video, ir_video,
                                       args.num_inference_steps, args.time_shift,
                                       latent_net=latent_net)

            for pos, f_idx in enumerate(win):
                pred_frame = pred_video[0, :, pos].float().cpu().numpy()
                y_pred = pred_frame.mean(axis=0) * 255.0
                w = float(POS_WEIGHTS[pos])
                if f_idx not in frame_sum:
                    frame_sum[f_idx] = y_pred * w
                    frame_wsum[f_idx] = w
                    frame_ycrcb[f_idx] = vi_ycrcb_list[pos]
                    frame_best_w[f_idx] = w
                else:
                    frame_sum[f_idx] += y_pred * w
                    frame_wsum[f_idx] += w
                    if w > frame_best_w[f_idx]:
                        frame_ycrcb[f_idx] = vi_ycrcb_list[pos]
                        frame_best_w[f_idx] = w

            for f_idx in win:
                if frame_last_window.get(f_idx, -1) != w_idx:
                    continue
                if frame_wsum.get(f_idx, 0) <= 0:
                    continue
                y_fused = np.clip(frame_sum[f_idx] / frame_wsum[f_idx], 0, 255).astype(np.uint8)
                save_single_frame_from_y(
                    y_fused, save_dir, f_idx,
                    vi_ycrcb=frame_ycrcb.get(f_idx),
                    interp=out_interp,
                    output_name=vi_frame_files[f_idx].name)
                total_frames_saved += 1
                for d in (frame_sum, frame_wsum, frame_ycrcb, frame_best_w):
                    d.pop(f_idx, None)

            del vi_video, ir_video, pred_video, vi_ycrcb_list
            torch.cuda.empty_cache()

        for f_idx in sorted(frame_sum.keys()):
            if frame_wsum.get(f_idx, 0) <= 0:
                continue
            y_fused = np.clip(frame_sum[f_idx] / frame_wsum[f_idx], 0, 255).astype(np.uint8)
            save_single_frame_from_y(
                y_fused, save_dir, f_idx,
                vi_ycrcb=frame_ycrcb.get(f_idx),
                interp=out_interp,
                output_name=vi_frame_files[f_idx].name)
            total_frames_saved += 1

        print(f"  ✓ {folder_name} 完成 -> {save_dir}")

    print(f"\n{'='*70}")
    print(f"✓ 全部完成!  共保存 {total_frames_saved} 帧")
    print(f"  结果保存在: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

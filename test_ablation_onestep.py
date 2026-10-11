#!/usr/bin/env python
# coding=utf-8
"""
推理脚本 — Fusion_SGFE 单步扩散（启用 SGFE block3）

输入: 512×512
输出: E:\paper2\ablation_result\none\<子文件夹>\gray|color

用法:
  python Fusion_SGFE/test_ablation_onestep.py
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
from Fusion_SGFE.train_one_step import build_temporal_restormer, decode_and_enhance


def parse_args():
    parser = argparse.ArgumentParser(description="推理: Fusion_SGFE")

    # ── 权重路径 ──
    parser.add_argument("--controlnet_checkpoint", type=str,
                        default=r"E:\paper2\codebase\STAR\outputs\33_vae4x\checkpoint-10000\model.safetensors")
    parser.add_argument("--unet_checkpoint", type=str,
                        default=r"E:\paper2\codebase\STAR\Ablation1\Fusion_SGFE\full_loss\checkpoint-3000\model.safetensors")
    parser.add_argument("--ir_select_mode", type=str, default="first",
                        choices=["middle", "first"],
                        help="IR 选取策略：middle=窗口中间帧，first=窗口第一帧")
    parser.add_argument("--raft_model_path", type=str,
                        default=r"Fusion_SGFE/ProPainter/weights/raft-things.pth")
    parser.add_argument("--sr_checkpoint", type=str,
                        default=r"Fusion_SGFE/train_one_step/checkpoint-3000/model.safetensors",
                        help="训练好的 TemporalRestormer 权重(accelerate 保存的 model.safetensors)；不存在则跳过增强")

    # ── 数据 ──
    parser.add_argument("--vi_root_dir", type=str,
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\test\LLVIP\visible2")
    parser.add_argument("--ir_root_dir", type=str,
                        default=r"E:\paper2\codebase\STAR\Fusion\dataset\test\LLVIP\infrared")

    # ── 输出 ──
    parser.add_argument("--output_dir", type=str,
                        default=r"E:\paper2\result\Our4")

    # ── 推理设置 ──
    parser.add_argument("--video_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--num_inference_steps", type=int, default=1,
                        help="扩散步数（单步脚本默认=1，实际仅执行一步）")
    parser.add_argument("--time_shift", type=float, default=3.0)
    parser.add_argument("--vi_per_ir", type=int, default=8,
                        help="仅作同名缺失时的回退比例索引；默认优先按窗口第一帧同名配对")
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


def _parse_trailing_int(stem):
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def select_ir_index_for_window(
    vi_files, ir_files, vi_start_idx, num_frames=8, vi_per_ir=8, mode="first"
):
    """窗口 IR 选取：同名优先，其次编号就近，最后比例回退。"""
    if len(ir_files) == 0:
        return 0
    vi_start_idx = max(0, min(int(vi_start_idx), len(vi_files) - 1))
    local_offset = (int(num_frames) - 1) // 2 if mode == "middle" else 0
    vi_target_idx = max(0, min(vi_start_idx + local_offset, len(vi_files) - 1))
    vi_stem = vi_files[vi_target_idx].stem
    ir_stem_to_idx = {p.stem: i for i, p in enumerate(ir_files)}
    if vi_stem in ir_stem_to_idx:
        return ir_stem_to_idx[vi_stem]
    vi_num = _parse_trailing_int(vi_stem)
    if vi_num is not None:
        pairs = sorted(
            [(_parse_trailing_int(p.stem) or 0, i) for i, p in enumerate(ir_files)],
            key=lambda x: x[0]
        )
        floor_cands = [(n, i) for n, i in pairs if n <= vi_num]
        if floor_cands:
            return floor_cands[-1][1]
        if pairs:
            return pairs[0][1]
    return min(vi_target_idx // max(1, int(vi_per_ir)), len(ir_files) - 1)


@torch.no_grad()
def run_inference(model, vi_video, ir_video, num_steps=1, time_shift=13.0, sr_module=None):
    model.eval_mode()
    B, C, T, H, W = vi_video.shape
    device = next(model.unet.parameters()).device
    vi_video_device = vi_video.to(device)
    ir_video_device = ir_video.to(device)
    vi_video_perm = vi_video_device.permute(0, 2, 1, 3, 4)
    ir_video_perm = ir_video_device.permute(0, 2, 1, 3, 4)
    vi_latent = model.vae_encode(vi_video_perm)
    ir_latent = model.vae_encode(ir_video_perm)
    # 开启 IR 光流 warp：传入 VI 像素帧并预计算光流供采样循环复用
    vi_frames = vi_video_perm.float() if model.ir_flow_prop is not None else None
    precomputed_flows = None
    if model.ir_flow_prop is not None and vi_frames is not None:
        precomputed_flows = model.ir_flow_prop.estimate_flow(vi_frames)
    # 单步扩散：只做一次速度预测 + 一次 scheduler 更新。
    steps = max(1, int(num_steps))
    model.noise_scheduler.set_timesteps(steps, time_shift=time_shift, device=device)
    x_t = vi_latent.clone()
    t_step = model.noise_scheduler.timesteps[0]
    t_batch = t_step.unsqueeze(0).repeat(B).to(device)
    v_pred = model.forward(
        x_t=x_t, t=t_batch, ir_latent=ir_latent,
        vi_frames=vi_frames, _precomputed_flows=precomputed_flows
    )
    # 显式单步积分：x_{t+1} = x_t + v * Δt
    # 与 scheduler.step 等价，但这里直接写出，便于与“单次速度预测”假设对齐。
    dt = model.noise_scheduler.sigmas[1] - model.noise_scheduler.sigmas[0]
    x_next = x_t.to(torch.float32) + v_pred.to(torch.float32) * dt
    if sr_module is not None:
        # 与训练一致的重建尾部：VAE 解码 + TemporalRestormer 增强
        pred_video = decode_and_enhance(model, sr_module, x_next, T)
    else:
        pred_video = model.vae_decode_chunk(x_next, chunk_size=T)
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
TAIL_FRAMES = 60


def main():
    args = parse_args()
    print("=" * 70)
    print("推理: Fusion_SGFE 单步扩散（启用 SGFE，启用 IR 光流 warp）")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1] 加载模型...")

    class _Opt:
        model_path = None
        vae_path = "stabilityai/stable-video-diffusion-img2vid"

    # 与 Fusion_SGFE 训练对齐：开启 SGFE(block3) + spa/temp local + 光流。
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
                # 兼容旧命名；同时跳过结构变化导致的 shape mismatch。
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
    print(f"    ✓ 模型准备完成")

    # ── 加载训练好的 TemporalRestormer（解码尾部增强头）──
    sr_module = None
    sr_path = args.sr_checkpoint
    if sr_path and not os.path.isabs(sr_path):
        cand = os.path.join(base_path, sr_path)
        if os.path.exists(cand):
            sr_path = cand
    if sr_path and os.path.exists(sr_path):
        sr_module = build_temporal_restormer(scale=1).to(device)
        sr_state = load_file(sr_path)
        sr_state = {k.replace('module.', ''): v for k, v in sr_state.items()}
        ret = sr_module.load_state_dict(sr_state, strict=False)
        sr_module.eval()
        print(f"    ✓ TemporalRestormer 加载完成 (loaded={len(sr_state)}, "
              f"missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)})")
    else:
        print(f"    ⚠ 未找到 TemporalRestormer 权重，跳过增强（仅 VAE 解码）: {args.sr_checkpoint}")

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
        if len(vi_all) > TAIL_FRAMES:
            vi_frame_files = vi_all[-TAIL_FRAMES:]
            ir_frame_files = ir_all[-min(TAIL_FRAMES, len(ir_all)):]
        else:
            vi_frame_files = vi_all
            ir_frame_files = ir_all

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

        center_pos = (NUM_FRAMES - 1) / 2.0
        pos_weights = [1.0 - abs(p - center_pos) / (center_pos + 0.5) for p in range(NUM_FRAMES)]

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
                                       sr_module=sr_module)

            for pos, f_idx in enumerate(win):
                pred_frame = pred_video[0, :, pos].float().cpu().numpy()
                y_pred = pred_frame.mean(axis=0) * 255.0
                w = float(pos_weights[pos])
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

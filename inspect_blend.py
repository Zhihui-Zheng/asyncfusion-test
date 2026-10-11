#!/usr/bin/env python
# coding=utf-8
"""
数据集预览脚本：每隔 4 帧生成一张 (VI + IR) / 2 的粗融合图
用途：人工检查各子文件夹的配准质量 & 动态内容丰富度
      根据结果决定哪些子文件夹值得纳入正式训练集

输入:
  visible2  : E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\visible2
  infrared  : E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\infrared

输出:
  LLVIP331  : E:\paper2\codebase\STAR\Fusion\dataset\LLVIP331
              ├── <子文件夹名>/
              │   ├── frame_0000_blend.png   ← 第 0 帧
              │   ├── frame_0004_blend.png   ← 第 4 帧
              │   ├── frame_0008_blend.png   ← 第 8 帧
              │   └── ...
              └── summary.txt               ← 各文件夹帧数 / 采样数汇总
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============ 路径配置 ============
VI_ROOT  = r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\visible2"
IR_ROOT  = r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\infrared"
OUT_ROOT = r"E:\paper2\codebase\STAR\Fusion\dataset\LLVIP331"
STEP     = 4        # 每隔多少帧采样一张
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def get_sorted_images(folder: Path):
    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in IMG_EXTS]
    return files


def blend_frame(vi_path: Path, ir_path: Path) -> np.ndarray:
    """
    加载 VI 和 IR，统一转灰度后 (VI + IR) / 2，返回 BGR 3 通道 uint8。
    VI 保留彩色通道用于肉眼对比颜色信息，最终拼成左右对照图:
      | VI原图 | IR原图 | 融合图 |
    这样方便你一眼判断配准 & 动态情况。
    """
    vi_bgr = cv2.imread(str(vi_path))
    ir_bgr = cv2.imread(str(ir_path))

    if vi_bgr is None or ir_bgr is None:
        return None

    # 统一尺寸（以 VI 为基准）
    h, w = vi_bgr.shape[:2]
    ir_bgr = cv2.resize(ir_bgr, (w, h))

    # 灰度融合：(VI_gray + IR_gray) / 2
    vi_gray = cv2.cvtColor(vi_bgr, cv2.COLOR_BGR2GRAY)
    ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)
    blend_gray = ((vi_gray.astype(np.float32) + ir_gray.astype(np.float32)) / 2).clip(0, 255).astype(np.uint8)
    blend_bgr  = cv2.cvtColor(blend_gray, cv2.COLOR_GRAY2BGR)

    # 拼成三联图：左=VI彩色  中=IR（伪彩）  右=融合灰度
    ir_show = cv2.applyColorMap(ir_gray, cv2.COLORMAP_INFERNO)  # IR 伪彩更直观
    panel   = np.concatenate([vi_bgr, ir_show, blend_bgr], axis=1)
    return panel


def main():
    vi_root  = Path(VI_ROOT)
    ir_root  = Path(IR_ROOT)
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # 扫描匹配的子文件夹
    vi_subdirs = sorted([d for d in vi_root.iterdir() if d.is_dir()])
    matched = []
    for vi_sub in vi_subdirs:
        ir_sub = ir_root / vi_sub.name
        if ir_sub.exists() and ir_sub.is_dir():
            matched.append((vi_sub, ir_sub))

    if not matched:
        print("未找到任何匹配的子文件夹，请检查路径。")
        return

    print(f"找到 {len(matched)} 个匹配子文件夹，每隔 {STEP} 帧采样一张融合预览图\n")

    summary_lines = [f"{'文件夹':<30} {'VI帧数':>8} {'IR帧数':>8} {'采样数':>8}"]
    summary_lines.append("-" * 60)

    for vi_sub, ir_sub in tqdm(matched, desc="处理子文件夹"):
        vi_files = get_sorted_images(vi_sub)
        ir_files = get_sorted_images(ir_sub)

        n_vi = len(vi_files)
        n_ir = len(ir_files)

        if n_vi == 0 or n_ir == 0:
            summary_lines.append(f"{vi_sub.name:<30} {n_vi:>8} {n_ir:>8} {'跳过(空)':>8}")
            continue

        out_sub = out_root / vi_sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        sample_indices = list(range(0, n_vi, STEP))
        n_saved = 0

        for idx in sample_indices:
            vi_path = vi_files[idx]
            # IR 可能帧数与 VI 不完全一致，做安全映射
            ir_idx  = min(idx, n_ir - 1)
            ir_path = ir_files[ir_idx]

            panel = blend_frame(vi_path, ir_path)
            if panel is None:
                continue

            # 文件名保留原始帧序号，方便回溯
            save_name = f"frame_{idx:04d}_blend.png"
            cv2.imwrite(str(out_sub / save_name), panel)
            n_saved += 1

        summary_lines.append(f"{vi_sub.name:<30} {n_vi:>8} {n_ir:>8} {n_saved:>8}")

    # 写汇总文件
    summary_path = out_root / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\n完成！预览图保存在: {out_root}")
    print(f"汇总文件: {summary_path}")
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()

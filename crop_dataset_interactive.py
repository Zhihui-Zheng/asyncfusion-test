#!/usr/bin/env python
# coding=utf-8
"""
交互式裁剪数据集脚本

功能:
1. 以 Fusion_bad 为基准，遍历每个子文件夹
2. 对每个子文件夹，显示第一个 group 的最后一帧，让用户手动画框
3. 同一子文件夹的所有 group 共用同一个裁剪框
4. 裁剪后 resize 到 256x256
5. 同时对 4 个数据集做相同裁剪：visible1, infrared, Fusion_good, Fusion_bad
6. 保存到新的大文件夹，保持原有目录结构

使用方法:
    python Fusion/crop_dataset_interactive.py

操作说明:
    - 鼠标左键按住拖动画框
    - 按 'r' 重新画框
    - 按 'c' 或 Enter 确认裁剪
    - 按 's' 跳过当前子文件夹
    - 按 'q' 退出
"""

import os
import cv2
import numpy as np
from pathlib import Path
import re
import shutil

# ============ 配置 ============
BASE_DIR = r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP"
OUTPUT_DIR = r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD_cropped"
TARGET_SIZE = (256, 256)  # 裁剪后 resize 到这个尺寸

# 需要同时裁剪的 4 个数据集
DATASETS = {
    'visible1': 'visible1',
    'infrared': 'infrared', 
    'Fusion_good': 'Fusion_good',
    'Fusion_bad': 'Fusion_bad',
}

# 用于画框的全局变量
drawing = False
ix, iy = -1, -1
rect_start = None
rect_end = None
current_rect = None


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于画框"""
    global drawing, ix, iy, rect_start, rect_end, current_rect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect_start = (x, y)
        rect_end = None
        current_rect = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect_end = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end = (x, y)
        # 确保坐标正确（左上角到右下角）
        x1 = min(ix, x)
        y1 = min(iy, y)
        x2 = max(ix, x)
        y2 = max(iy, y)
        if x2 - x1 > 10 and y2 - y1 > 10:  # 最小框大小
            current_rect = (x1, y1, x2, y2)


def get_user_crop_rect(image, window_name="Select ROI - Press 'c' to confirm, 'r' to reset, 's' to skip"):
    """
    让用户在图片上画框选择裁剪区域
    
    Returns:
        (x1, y1, x2, y2) 或 None (如果跳过)
    """
    global drawing, rect_start, rect_end, current_rect
    
    # 重置状态
    drawing = False
    rect_start = None
    rect_end = None
    current_rect = None
    
    # 创建窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    display_img = image.copy()
    
    while True:
        # 绘制当前帧
        show_img = display_img.copy()
        
        # 绘制正在画的框
        if drawing and rect_start and rect_end:
            cv2.rectangle(show_img, rect_start, rect_end, (0, 255, 0), 2)
        
        # 绘制已确定的框
        if current_rect:
            x1, y1, x2, y2 = current_rect
            cv2.rectangle(show_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 显示尺寸信息
            cv2.putText(show_img, f"Size: {x2-x1}x{y2-y1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示提示
        cv2.putText(show_img, "Draw box with mouse. 'c'=confirm, 'r'=reset, 's'=skip, 'q'=quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow(window_name, show_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') or key == 13:  # 'c' 或 Enter 确认
            if current_rect:
                cv2.destroyWindow(window_name)
                return current_rect
            else:
                print("  请先画一个框！")
        
        elif key == ord('r'):  # 重置
            current_rect = None
            rect_start = None
            rect_end = None
            print("  已重置，请重新画框")
        
        elif key == ord('s'):  # 跳过
            cv2.destroyWindow(window_name)
            return None
        
        elif key == ord('q'):  # 退出
            cv2.destroyWindow(window_name)
            return 'quit'
    
    cv2.destroyWindow(window_name)
    return None


def extract_group_name(filename):
    """从文件名提取 group 名称"""
    match = re.search(r'(group\d+)', filename)
    if match:
        return match.group(1)
    return None


def get_groups_in_sample(sample_dir):
    """
    获取某个子文件夹中所有 group 的信息
    
    Returns:
        dict: {group_name: [file_paths], ...}
    """
    all_files = sorted(list(sample_dir.glob('*.png')) + list(sample_dir.glob('*.jpg')))
    
    if not all_files:
        return {}
    
    # 按 group 分组
    groups = {}
    for f in all_files:
        group_name = extract_group_name(f.name)
        if group_name:
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(f)
    
    # 按文件名排序每个 group 内的文件
    for group_name in groups:
        groups[group_name] = sorted(groups[group_name])
    
    return groups


def get_last_frame_of_group(groups, group_name):
    """获取指定 group 的最后一帧"""
    if group_name not in groups:
        return None
    return groups[group_name][-1]


def crop_and_save_image(src_path, dst_path, crop_rect, target_size):
    """
    裁剪图片并保存
    
    Args:
        src_path: 源图片路径
        dst_path: 目标图片路径
        crop_rect: (x1, y1, x2, y2) 裁剪框
        target_size: (width, height) 目标尺寸
    """
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"    警告: 无法读取 {src_path}")
        return False
    
    x1, y1, x2, y2 = crop_rect
    
    # 确保裁剪框在图片范围内
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h))
    
    # 裁剪
    cropped = img[y1:y2, x1:x2]
    
    if cropped.size == 0:
        print(f"    警告: 裁剪后图片为空 {src_path}")
        return False
    
    # Resize
    resized = cv2.resize(cropped, target_size)
    
    # 确保目标目录存在
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    cv2.imwrite(str(dst_path), resized)
    return True


def extract_frame_number(filename):
    """从文件名提取帧号"""
    # 匹配 groupXX_frame_XXXX 或 frame_XXXX 格式
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return match.group(1)
    return None


def process_group(sample_name, group_name, crop_rect, base_dir, output_dir, datasets, target_size):
    """
    处理一个样本的一个 group
    
    对 4 个数据集中同名子文件夹的同一个 group 应用相同的裁剪
    注意: Fusion_good/Fusion_bad 使用 groupXX_frame_XXXX 格式
          visible1/infrared 使用 frame_XXXX 格式
    """
    total_files = 0
    success_files = 0
    
    # 首先从 Fusion_bad 获取该 group 的帧号列表
    fusion_bad_dir = Path(base_dir) / 'Fusion_bad' / sample_name
    groups = get_groups_in_sample(fusion_bad_dir)
    if group_name not in groups:
        return 0, 0
    
    # 获取该 group 所有文件的帧号
    frame_numbers = []
    group_files = groups[group_name]
    for f in group_files:
        frame_num = extract_frame_number(f.name)
        if frame_num:
            frame_numbers.append(frame_num)
    
    for dataset_name, dataset_folder in datasets.items():
        src_sample_dir = Path(base_dir) / dataset_folder / sample_name
        dst_sample_dir = Path(output_dir) / dataset_folder / sample_name
        
        if not src_sample_dir.exists():
            continue
        
        # 根据数据集类型选择不同的文件名格式
        if dataset_folder in ['Fusion_good', 'Fusion_bad']:
            # 使用 groupXX_frame_XXXX 格式
            local_groups = get_groups_in_sample(src_sample_dir)
            if group_name not in local_groups:
                continue
            
            for src_file in local_groups[group_name]:
                dst_file = dst_sample_dir / src_file.name
                total_files += 1
                
                if crop_and_save_image(src_file, dst_file, crop_rect, target_size):
                    success_files += 1
        else:
            # visible1/infrared 使用 frame_XXXX 格式
            for frame_num in frame_numbers:
                # 尝试不同的扩展名
                for ext in ['.png', '.jpg', '.jpeg']:
                    src_file = src_sample_dir / f"frame_{frame_num}{ext}"
                    if src_file.exists():
                        # 保存时使用 groupXX_frame_XXXX 格式以保持一致性
                        dst_file = dst_sample_dir / f"{group_name}_frame_{frame_num}{ext}"
                        total_files += 1
                        
                        if crop_and_save_image(src_file, dst_file, crop_rect, target_size):
                            success_files += 1
                        break
    
    return success_files, total_files


def main():
    print("=" * 70)
    print("交互式数据集裁剪工具 (按 Group 单独画框)")
    print("=" * 70)
    print(f"基准目录: {BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"目标尺寸: {TARGET_SIZE}")
    print(f"数据集: {list(DATASETS.keys())}")
    print("=" * 70)
    print("\n操作说明:")
    print("  - 鼠标左键拖动画框")
    print("  - 'c' 或 Enter: 确认裁剪")
    print("  - 'r': 重新画框")
    print("  - 's': 跳过当前 group")
    print("  - 'q': 退出程序")
    print("\n注意: 每个 group 需要单独画框！")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 以 Fusion_bad 为基准获取所有子文件夹
    base_dataset = Path(BASE_DIR) / 'Fusion_bad'
    
    if not base_dataset.exists():
        print(f"错误: 基准目录不存在 {base_dataset}")
        return
    
    sample_dirs = sorted([d for d in base_dataset.iterdir() if d.is_dir()])
    print(f"\n找到 {len(sample_dirs)} 个子文件夹待处理\n")
    
    total_groups_processed = 0
    total_groups_skipped = 0
    quit_flag = False
    
    for idx, sample_dir in enumerate(sample_dirs):
        if quit_flag:
            break
            
        sample_name = sample_dir.name
        print(f"\n{'='*50}")
        print(f"[{idx+1}/{len(sample_dirs)}] 子文件夹: {sample_name}")
        print(f"{'='*50}")
        
        # 获取该子文件夹的所有 group
        groups = get_groups_in_sample(sample_dir)
        
        if not groups:
            print(f"  跳过: 找不到有效的 group")
            continue
        
        group_names = sorted(groups.keys())
        print(f"  找到 {len(group_names)} 个 group: {group_names}")
        
        # 对每个 group 单独画框
        for g_idx, group_name in enumerate(group_names):
            if quit_flag:
                break
                
            # 获取该 group 的最后一帧
            last_frame_path = get_last_frame_of_group(groups, group_name)
            
            if last_frame_path is None:
                print(f"    [{group_name}] 跳过: 找不到图片")
                total_groups_skipped += 1
                continue
            
            print(f"\n    [{g_idx+1}/{len(group_names)}] {group_name}: {last_frame_path.name}")
            
            # 读取图片
            img = cv2.imread(str(last_frame_path))
            if img is None:
                print(f"    跳过: 无法读取图片")
                total_groups_skipped += 1
                continue
            
            # 让用户画框
            window_title = f"{sample_name} - {group_name} ({g_idx+1}/{len(group_names)}) - Draw crop box"
            crop_rect = get_user_crop_rect(img, window_title)
            
            if crop_rect == 'quit':
                print("\n用户退出")
                quit_flag = True
                break
            
            if crop_rect is None:
                print(f"    跳过 {group_name}")
                total_groups_skipped += 1
                continue
            
            print(f"    裁剪框: {crop_rect}")
            
            # 对该 group 应用裁剪（4个数据集）
            success, total = process_group(
                sample_name, group_name, crop_rect, BASE_DIR, OUTPUT_DIR, DATASETS, TARGET_SIZE
            )
            
            print(f"    完成: {success}/{total} 张图片")
            total_groups_processed += 1
    
    print("\n" + "=" * 70)
    print(f"处理完成!")
    print(f"  已处理: {total_groups_processed} 个 group")
    print(f"  已跳过: {total_groups_skipped} 个 group")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()


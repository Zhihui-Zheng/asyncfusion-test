"""
SEA-RAFT 光流可视化 Demo
用法: python SEA_RAFT/demo.py --input_dir <输入图片文件夹> --output_dir <输出文件夹>
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from SEA_RAFT.core.raft import RAFT
from SEA_RAFT.core.utils.utils import load_ckpt
from SEA_RAFT.core.utils.flow_viz import flow_to_image
from SEA_RAFT.config.parser import json_to_args


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_image(imfile, resize=None):
    """加载图片并转换为tensor"""
    img = Image.open(imfile).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), size=resize, mode='bilinear', align_corners=False).squeeze(0)
    return img[None].to(DEVICE)


def forward_flow(args, model, image1, image2):
    """前向计算光流"""
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    return flow_final


def calc_flow(args, model, image1, image2):
    """计算光流，支持多尺度"""
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    flow = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    return flow_down


def save_flow_visualization(flow, output_path):
    """保存光流可视化结果"""
    # flow: [1, 2, H, W] -> [H, W, 2]
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
    # 转换为RGB图像
    flow_vis = flow_to_image(flow_np, convert_to_bgr=False)
    # 保存图片
    Image.fromarray(flow_vis).save(output_path)


def get_image_files(input_dir):
    """获取文件夹中的所有图片文件"""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, ext)))
    images = sorted(images)
    return images


@torch.no_grad()
def demo(args):
    """主函数：计算相邻帧光流并保存可视化结果"""
    # 加载配置
    model_args = json_to_args(args.cfg)
    model_args.scale = args.scale
    model_args.iters = args.iters if args.iters else model_args.iters
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = RAFT(model_args)
    load_ckpt(model, args.model)
    model = model.to(DEVICE)
    model.eval()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取图片列表
    images = get_image_files(args.input_dir)
    if len(images) < 2:
        print(f"错误: 输入文件夹 {args.input_dir} 中图片数量少于2张")
        return
    
    print(f"找到 {len(images)} 张图片")
    print(f"将计算 {len(images) - 1} 对相邻帧的光流")
    
    # 计算相邻帧光流
    for i in tqdm(range(len(images) - 1), desc="计算光流"):
        imfile1 = images[i]
        imfile2 = images[i + 1]
        
        # 加载图片
        if args.resize:
            resize = tuple(args.resize)
            image1 = load_image(imfile1, resize=resize)
            image2 = load_image(imfile2, resize=resize)
        else:
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
        
        # 计算光流
        flow = calc_flow(model_args, model, image1, image2)
        
        # 生成输出文件名
        basename1 = os.path.splitext(os.path.basename(imfile1))[0]
        basename2 = os.path.splitext(os.path.basename(imfile2))[0]
        output_filename = f"flow_{basename1}_{basename2}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # 保存可视化结果
        save_flow_visualization(flow, output_path)
    
    print(f"完成! 光流可视化结果已保存到: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SEA-RAFT 光流可视化 Demo')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入图片文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出光流可视化结果的文件夹路径')
    parser.add_argument('--model', type=str, 
                        default='models/Tartan-C-T-TSKH-kitti432x960-M.pth',
                        help='模型权重路径')
    parser.add_argument('--cfg', type=str,
                        default='SEA_RAFT/config/eval/kitti-M.json',
                        help='配置文件路径')
    parser.add_argument('--iters', type=int, default=None,
                        help='迭代次数 (默认使用配置文件中的值)')
    parser.add_argument('--scale', type=int, default=0,
                        help='图像缩放尺度 (默认: 0, 即不缩放)')
    parser.add_argument('--resize', type=int, nargs=2, default=None,
                        help='调整图像大小 (高度 宽度), 例如: --resize 432 960')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    # 检查配置文件是否存在
    if not os.path.exists(args.cfg):
        print(f"错误: 配置文件不存在: {args.cfg}")
        return
    
    demo(args)


if __name__ == '__main__':
    main()


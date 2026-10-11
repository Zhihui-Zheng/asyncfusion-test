# Copyright (c) Image-conditioned Video Generation Dataset
# 图片条件视频生成数据集 - 支持EXR/PNG序列和MP4视频

import os
import random
import glob
import torchvision
from einops import rearrange
from torch.utils import data as data
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import OpenEXR
import Imath


def read_exr_image(exr_path):
    """
    读取EXR图片并转换为numpy数组
    
    Args:
        exr_path: EXR文件路径
    
    Returns:
        numpy array: [H, W, C] 格式的图片数据
    """
    exr_file = OpenEXR.InputFile(exr_path)
    dw = exr_file.header()['dataWindow']
    height = dw.max.y - dw.min.y + 1
    width = dw.max.x - dw.min.x + 1
    
    # 获取通道
    channels = exr_file.header()['channels'].keys()
    
    # 读取RGB通道
    if {'R', 'G', 'B'}.issubset(channels):
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        (R, G, B) = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in ('R', 'G', 'B')]
        
        # Reshape
        R = R.reshape((height, width))
        G = G.reshape((height, width))
        B = B.reshape((height, width))
        
        # 组合成[H, W, C]
        img = np.stack([R, G, B], axis=-1)
    else:
        # 单通道
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        R = np.frombuffer(exr_file.channel('R', FLOAT), dtype=np.float32)
        R = R.reshape((height, width))
        img = np.stack([R, R, R], axis=-1)  # 复制到3通道
    
    return img


def load_image_sequence(sequence_dir, num_frames=None, file_pattern='*.exr', random_start=True):
    """
    从目录加载图片序列（支持EXR、PNG、JPG等多种格式）
    
    Args:
        sequence_dir: 图片序列目录
        num_frames: 要加载的帧数（None表示加载所有）
        file_pattern: 文件匹配模式（不再使用，自动检测所有支持的格式）
        random_start: 是否随机选择起始帧（用于数据增强）
    
    Returns:
        torch.Tensor: [C, T, H, W] 格式的视频张量
    """
    # 自动检测并加载多种格式的图片
    image_files = []
    supported_extensions = ['*.exr', '*.EXR', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    
    for pattern in supported_extensions:
        image_files.extend(glob.glob(os.path.join(sequence_dir, pattern)))
    
    # 按文件名排序
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {sequence_dir}. Supported formats: {supported_extensions}")
    
    # 随机连续帧采样（数据增强）
    if num_frames is not None and len(image_files) > num_frames:
        if random_start:
            max_start = len(image_files) - num_frames
            start_idx = random.randint(0, max_start)
            image_files = image_files[start_idx:start_idx + num_frames]
        else:
            image_files = image_files[:num_frames]
    elif num_frames is not None:
        image_files = image_files[:num_frames]
    
    frames = []
    for img_path in image_files:
        ext = os.path.splitext(img_path)[1].lower()
        
        if ext == '.exr':
            # 读取EXR
            img_np = read_exr_image(img_path)  # [H, W, C]
        elif ext in ['.png', '.jpg', '.jpeg']:
            # 读取PNG/JPG
            img_pil = Image.open(img_path).convert('RGB')
            img_np = np.array(img_pil) / 255.0  # 归一化到[0, 1]
        else:
            raise ValueError(f"Unsupported image format: {ext}")
        
        frames.append(img_np)
    
    # 转换为tensor [T, H, W, C]
    frames_np = np.stack(frames, axis=0)
    
    # 转换为torch tensor并调整维度顺序: [T, H, W, C] -> [C, T, H, W]
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
    
    # 归一化到[-1, 1]（如果还没有归一化）
    if frames_tensor.max() > 1.0:  # EXR可能有HDR值
        # 使用tone mapping将HDR映射到[0, 1]
        frames_tensor = torch.clamp(frames_tensor, 0, None)  # 裁剪负值
        # 简单的Reinhard tone mapping
        frames_tensor = frames_tensor / (1.0 + frames_tensor)
    
    # 缩放到[-1, 1]
    frames_tensor = frames_tensor * 2.0 - 1.0
    
    return frames_tensor


class PairedImageConditionedVideoDataset(data.Dataset):
    """
    数据集结构（EXR/PNG序列格式）：
    root_folder/
        ├── lq/              # 低质量图片序列
        │   ├── video1/      # 每个视频是一个文件夹
        │   │   ├── 0001.exr
        │   │   ├── 0002.exr
        │   │   └── ...
        │   ├── video2/
        │   └── ...
        └── gt/              # 高质量图片序列（真值）
            ├── video1/
            │   ├── 0001.exr
            │   ├── 0002.exr
            │   └── ...
            └── ...
    
    也支持MP4视频格式（自动检测）
    
    注意：条件图片会从gt序列中随机选择一帧，不再需要单独的condition文件夹
    """
    
    def __init__(
            self,
            root_folders=None,
            null_condition_ratio=0.1,
            num_frames=16,
            image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.exr'),
            sequence_pattern='*.exr',  # 序列文件匹配模式
            use_tone_mapping=True,  # 是否对HDR使用tone mapping
            video_size=(224, 224),  # 视频帧的空间分辨率 (H, W)
    ):
        super(PairedImageConditionedVideoDataset, self).__init__()

        self.null_condition_ratio = null_condition_ratio
        self.lr_list = []
        self.gt_list = []
        self.num_frames = num_frames
        self.image_extensions = image_extensions
        self.sequence_pattern = sequence_pattern
        self.use_tone_mapping = use_tone_mapping
        self.video_size = video_size

        # 条件图片预处理（用于CLIP）
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP标准尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])  # CLIP标准归一化
        ])
        
        # 视频帧预处理（空间resize）
        self.video_transform = transforms.Resize(video_size, antialias=True)

        for root_folder in root_folders:
            lr_path = os.path.join(root_folder, 'lq')
            gt_path = os.path.join(root_folder, 'gt')

            # 检测是MP4视频还是图片序列
            # 优先检查是否有序列文件夹
            lr_sequences = sorted([d for d in glob.glob(os.path.join(lr_path, '*')) if os.path.isdir(d)])
            
            if len(lr_sequences) > 0:
                # 图片序列模式
                gt_sequences = sorted([d for d in glob.glob(os.path.join(gt_path, '*')) if os.path.isdir(d)])
                
                for lr_seq, gt_seq in zip(lr_sequences, gt_sequences):
                    self.lr_list.append(lr_seq)
                    self.gt_list.append(gt_seq)
            else:
                # MP4视频模式（兼容原有格式）
                lr_videos = sorted(glob.glob(os.path.join(lr_path, '*.mp4')))
                gt_videos = sorted(glob.glob(os.path.join(gt_path, '*.mp4')))
                
                for lr_video, gt_video in zip(lr_videos, gt_videos):
                    self.lr_list.append(lr_video)
                    self.gt_list.append(gt_video)

        assert len(self.lr_list) == len(self.gt_list), \
            f"Mismatch in dataset sizes: lq={len(self.lr_list)}, gt={len(self.gt_list)}"
        
        print(f"Dataset initialized with {len(self.lr_list)} video/sequence pairs")

    def __getitem__(self, index):
        gt_path = self.gt_list[index]
        lq_path = self.lr_list[index]
        
        # 判断是视频文件还是图片序列
        if os.path.isdir(gt_path):
            # 图片序列模式
            vframes_gt = load_image_sequence(gt_path, self.num_frames, self.sequence_pattern)
            vframes_lq = load_image_sequence(lq_path, self.num_frames, self.sequence_pattern)
            fps = 30  # 默认fps
            
            # 确保 gt 和 lq 帧数一致（取最小值）
            min_frames = min(vframes_gt.shape[1], vframes_lq.shape[1], self.num_frames)
            vframes_gt = vframes_gt[:, :min_frames, :, :]
            vframes_lq = vframes_lq[:, :min_frames, :, :]
        else:
            # MP4视频模式
            vframes_gt, _, info = torchvision.io.read_video(filename=gt_path, pts_unit="sec", output_format="TCHW")
            fps = info['video_fps']
            vframes_gt = (rearrange(vframes_gt, "T C H W -> C T H W") / 255) * 2 - 1
            
            vframes_lq, _, _ = torchvision.io.read_video(filename=lq_path, pts_unit="sec", output_format="TCHW")
            vframes_lq = (rearrange(vframes_lq, "T C H W -> C T H W") / 255) * 2 - 1
            
            # 限制帧数
            min_frames = min(vframes_gt.shape[1], vframes_lq.shape[1], self.num_frames)
            vframes_gt = vframes_gt[:, :min_frames, :, :]
            vframes_lq = vframes_lq[:, :min_frames, :, :]
        
        # Resize 视频帧到指定尺寸 [C, T, H, W] -> [C, T, H_new, W_new]
        if self.video_size is not None:
            C, T, H, W = vframes_gt.shape
            # 转换为 [T, C, H, W] 以便使用 transforms
            vframes_gt = rearrange(vframes_gt, 'c t h w -> t c h w')
            vframes_lq = rearrange(vframes_lq, 'c t h w -> t c h w')
            # 应用 resize
            vframes_gt = torch.stack([self.video_transform(frame) for frame in vframes_gt], dim=0)
            vframes_lq = torch.stack([self.video_transform(frame) for frame in vframes_lq], dim=0)
            # 转回 [C, T, H, W]
            vframes_gt = rearrange(vframes_gt, 't c h w -> c t h w')
            vframes_lq = rearrange(vframes_lq, 't c h w -> c t h w')

        # ========== 修改：从GT序列中随机选择一帧作为条件图片 ==========
        # 使用null condition（用于classifier-free guidance）
        if random.random() < self.null_condition_ratio:
            # 创建黑色图片作为null condition
            condition_image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            # 从GT视频中随机选择一帧作为参考帧
            # vframes_gt shape: [C, T, H, W]
            T = vframes_gt.shape[1]
            random_frame_idx = random.randint(0, T - 1)
            
            # 提取该帧 [C, H, W]
            reference_frame = vframes_gt[:, random_frame_idx, :, :]  # [C, H, W]
            
            # 反归一化：从[-1, 1]到[0, 1]
            reference_frame = (reference_frame + 1.0) / 2.0
            reference_frame = torch.clamp(reference_frame, 0, 1)
            
            # 转换为PIL Image
            # [C, H, W] -> [H, W, C]
            reference_frame_np = reference_frame.permute(1, 2, 0).numpy()
            reference_frame_np = (reference_frame_np * 255).astype(np.uint8)
            condition_image = Image.fromarray(reference_frame_np, mode='RGB')
        
        # 预处理图片（resize到224x224并应用CLIP归一化）
        condition_image = self.image_transform(condition_image)

        return {
            "gt": vframes_gt, 
            "lq": vframes_lq, 
            "condition_image": condition_image,
            'fps': fps
        }

    def __len__(self):
        return len(self.gt_list)


class PairedImageConditionedImageDataset(data.Dataset):
    """
    图片超分辨率数据集（用于图片条件）
    支持EXR和PNG格式
    
    数据集结构：
    root_folder/
        ├── lq/              # 低质量图片
        │   ├── img1.exr / img1.png
        │   └── ...
        ├── gt/              # 高质量图片
        │   ├── img1.exr / img1.png
        │   └── ...
        └── condition/       # 条件图片
            ├── img1.jpg
            └── ...
    """
    
    def __init__(
            self,
            root_folder=None,
            null_condition_ratio=0.1,
            image_extensions=('.jpg', '.jpeg', '.png', '.bmp'),
            data_format='exr',  # 'exr' or 'png'
    ):
        super(PairedImageConditionedImageDataset, self).__init__()

        self.null_condition_ratio = null_condition_ratio
        self.lr_list = []
        self.gt_list = []
        self.condition_image_list = []
        self.image_extensions = image_extensions
        self.data_format = data_format

        lr_path = os.path.join(root_folder, 'lq')
        gt_path = os.path.join(root_folder, 'gt')
        condition_path = os.path.join(root_folder, 'condition')

        # 获取所有图片
        if data_format == 'exr':
            lr_images = sorted(glob.glob(os.path.join(lr_path, '*.exr')))
            gt_images = sorted(glob.glob(os.path.join(gt_path, '*.exr')))
        else:
            lr_images = sorted(glob.glob(os.path.join(lr_path, '*.png')))
            gt_images = sorted(glob.glob(os.path.join(gt_path, '*.png')))

        for lr_img, gt_img in zip(lr_images, gt_images):
            base_name = os.path.splitext(os.path.basename(lr_img))[0]
            
            condition_image = None
            for ext in self.image_extensions:
                candidate = os.path.join(condition_path, base_name + ext)
                if os.path.exists(candidate):
                    condition_image = candidate
                    break
            
            if condition_image is None:
                print(f"Warning: No condition image found for {lr_img}, skipping...")
                continue
            
            self.lr_list.append(lr_img)
            self.gt_list.append(gt_img)
            self.condition_image_list.append(condition_image)

        assert len(self.lr_list) == len(self.gt_list) == len(self.condition_image_list)

        # 图片预处理
        self.condition_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])

        crop_size = (720, 1280)
        self.center_crop = transforms.CenterCrop(crop_size)

    def __getitem__(self, index):
        gt_path = self.gt_list[index]
        lq_path = self.lr_list[index]
        
        # 读取GT和LQ图片
        if self.data_format == 'exr':
            gt_img = read_exr_image(gt_path)  # [H, W, C]
            lq_img = read_exr_image(lq_path)
            
            # 转换为tensor
            gt_img = torch.from_numpy(gt_img).float().permute(2, 0, 1)  # [C, H, W]
            lq_img = torch.from_numpy(lq_img).float().permute(2, 0, 1)
            
            # Tone mapping
            gt_img = torch.clamp(gt_img, 0, None)
            lq_img = torch.clamp(lq_img, 0, None)
            gt_img = gt_img / (1.0 + gt_img)
            lq_img = lq_img / (1.0 + lq_img)
        else:
            gt_img = Image.open(gt_path).convert('RGB')
            lq_img = Image.open(lq_path).convert('RGB')
            gt_img = transforms.ToTensor()(gt_img)
            lq_img = transforms.ToTensor()(lq_img)
        
        # Center crop
        gt_img = self.center_crop(gt_img)
        lq_img = self.center_crop(lq_img)

        # 加载条件图片
        condition_image_path = self.condition_image_list[index]
        
        if random.random() < self.null_condition_ratio:
            condition_image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            condition_image = Image.open(condition_image_path).convert('RGB')
        
        condition_image = self.condition_transform(condition_image)

        example = dict()
        example["lq"] = (lq_img * 2.0 - 1.0).unsqueeze(1)
        example["gt"] = (gt_img * 2.0 - 1.0).unsqueeze(1)
        example["condition_image"] = condition_image

        return example

    def __len__(self):
        return len(self.gt_list)

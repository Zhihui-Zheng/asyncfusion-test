"""
TemCoCo 解码器独立实现
从 archs/Blocks.py 和 archs/net_basics.py 提取
"""

import torch
import torch.nn as nn


class Conv3DModule(nn.Module):
    """3D卷积模块，包含卷积和激活函数"""
    def __init__(self, in_channels=3, out_channels=64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)):
        super(Conv3DModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.act(x)
        return x


class Separable3DConv(nn.Module):
    """
    可分离3D卷积
    先进行空间卷积(H×W)，再进行时间维度卷积(T)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(Separable3DConv, self).__init__()
        # 2D 空间卷积 (Height × Width)，kernel_size=(1,3,3)表示只在H和W维度卷积
        self.spatial_conv = Conv3DModule(in_channels, out_channels, 
                                        kernel_size=(1,3,3), 
                                        stride=(1,stride,stride), 
                                        padding=(0,1,1))
        # 1D 时间维度卷积 (Depth/Time)，kernel_size=(3,1,1)表示只在T维度卷积
        self.depth_conv = Conv3DModule(out_channels, out_channels, 
                                       kernel_size=(3,1,1), 
                                       stride=1, 
                                       padding=(1,0,0))

    def forward(self, x):
        """
        输入: (B, C, T, H, W)
        输出: (B, C, T, H, W) - 通道数从in_channels变为out_channels
        """
        x = self.spatial_conv(x)  # 空间卷积
        x = self.depth_conv(x)     # 时间维度卷积
        return x


class Decoder_v3(nn.Module):
    """
    TemCoCo使用的解码器 - 这是TemCoCo官方代码中的实现
    
    输入格式: (B, C, T, H, W)
    - B: batch size
    - C: 输入通道数 (in_dims，实际使用中是64，来自DCN_2的输出)
    - T: 时间帧数
    - H: 高度
    - W: 宽度
    
    输出格式: (B, C_out, T, H, W)
    - C_out: 输出通道数 (nf，默认3，即RGB)
    - 其他维度保持不变
    
    ⚠️ 重要：这个解码器只有单层结构！
    - 没有上采样（因为输入已经是原始分辨率）
    - 没有多尺度融合
    - 没有残差块或其他复杂结构
    - 只有一个Separable3DConv层，将64维特征映射到3通道RGB
    
    从代码注释可以看到，原本可能有更多层，但都被注释掉了：
    # self.conv1 = conv_resblock_two(nf, nf)
    # self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
    # self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
    
    这种设计是因为DCN_2已经将特征上采样到原始分辨率，解码器只需要简单的通道映射。
    """
    def __init__(self, in_dims=32, nf=3):
        """
        Args:
            in_dims: 输入特征维度，默认32
            nf: 输出通道数，默认3（RGB）
        """
        super(Decoder_v3, self).__init__()
        self.conv0 = Separable3DConv(in_dims, nf)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 (B, in_dims, T, H, W)
        
        Returns:
            output: 输出图像，形状为 (B, nf, T, H, W)
        """
        x_ = self.conv0(x)
        return x_


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 创建解码器
    # 输入32维特征，输出3通道RGB
    decoder = Decoder_v3(in_dims=32, nf=3)
    
    # 模拟输入: (B=2, C=32, T=5, H=256, W=256)
    batch_size = 2
    num_frames = 5
    height, width = 256, 256
    input_features = torch.randn(batch_size, 32, num_frames, height, width)
    
    print(f"输入形状: {input_features.shape}")
    
    # 前向传播
    output = decoder(input_features)
    
    print(f"输出形状: {output.shape}")
    print(f"输入通道数: {input_features.shape[1]}")
    print(f"输出通道数: {output.shape[1]}")
    print(f"时间维度: {input_features.shape[2]} -> {output.shape[2]} (保持不变)")
    print(f"空间维度: {input_features.shape[3:]} -> {output.shape[3:]} (保持不变)")
    # 统计参数量
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder_v3 参数量: {num_params:,d}")
    # 验证输出
    assert output.shape == (batch_size, 3, num_frames, height, width), \
        f"输出形状不匹配！期望 {(batch_size, 3, num_frames, height, width)}, 得到 {output.shape}"
    print("\n✓ 解码器测试通过！")


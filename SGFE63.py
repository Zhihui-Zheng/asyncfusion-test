import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 频率工具（通道路径：正确的复数 FFT 带通滤波） ─────────────────────────────

def _frequency_mask(h, w, ratio, device):
    """
    生成高通 mask。FFT 低频分量分布在四个角落（Hermitian 对称），
    需对四角对称清零才能正确去除低频。
    """
    h0 = max(1, int(h * ratio[0]))
    w0 = max(1, int(w * ratio[1]))
    mask = torch.ones((h, w), device=device, dtype=torch.float32, requires_grad=False)
    mask[:h0,  :w0]  = 0
    mask[-h0:, :w0]  = 0
    mask[:h0,  -w0:] = 0
    mask[-h0:, -w0:] = 0
    return mask


def _fft_highpass(x: torch.Tensor, ratio) -> torch.Tensor:
    """
    正确的 FFT 高通滤波：保留完整复数频谱 → 乘高通 mask → irfft2。
    相位信息完整保留，无重建误差。
    """
    h, w = x.shape[-2], x.shape[-1]
    X      = torch.fft.rfft2(x, norm='ortho')
    w_r    = X.shape[-1]
    mask_r = _frequency_mask(h, w, ratio, device=x.device)[:, :w_r]
    return torch.fft.irfft2(X * mask_r.view(1, 1, h, w_r), s=(h, w), norm='ortho')


def _fft_lowpass(x: torch.Tensor, ratio) -> torch.Tensor:
    """
    FFT 低通滤波：高通 mask 取反即为低通，与高通互补。
    保留低频结构成分（大面积均匀区域、整体形状轮廓），
    用作 structure_gate 的输入，与高频路径形成真正的频域互补。
    """
    h, w = x.shape[-2], x.shape[-1]
    X      = torch.fft.rfft2(x, norm='ortho')
    w_r    = X.shape[-1]
    # 低通 mask = 1 - 高通 mask（精确互补，无重叠无遗漏）
    low_mask_r = 1.0 - _frequency_mask(h, w, ratio, device=x.device)[:, :w_r]
    return torch.fft.irfft2(X * low_mask_r.view(1, 1, h, w_r), s=(h, w), norm='ortho')


# ── GRN：全局响应归一化（ConvNeXt V2 风格） ──────────────────────────────────

class GRN(nn.Module):
    """
    用每个通道的全局 L2 范数做自归一化，防止强势通道垄断增强输出。
    零初始化 → 训练初期等效恒等映射，不干扰预训练特征。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta


# ── 空间路径：梯度算子替代 DCT 往返 ──────────────────────────────────────────

class GradSpatialInteraction(nn.Module):
    """
    空间维度的高频注意力。

    用固定 Sobel（一阶梯度）+ Laplacian（二阶导数）作为互补的局部空间变化描述符：
    - Sobel grad_mag：检测特征图中激活值变化的幅度（边界/结构响应）
    - Laplacian mag ：检测梯度的变化速率（角点/细纹理振荡）
    两者正交互补，拼接后由唯一可学习的 1×1 Conv 融合为空间注意力图。

    为何不用 DCT 空间路径：DCT 往返在 latent 特征图上存在频谱泄漏和相位损失，
    梯度算子对局部空间变化的测量更直接、无重建误差，且计算量更低。
    所有卷积核固定（register_buffer），仅 1×1 Conv 参与训练。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.register_buffer("sobel_x",
            torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3),
            persistent=False)
        self.register_buffer("sobel_y",
            torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3),
            persistent=False)
        self.register_buffer("laplace",
            torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3),
            persistent=False)

        self.gate = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _dw(self, x, k):
        return F.conv2d(x, k.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_mag = torch.sqrt(
            self._dw(x, self.sobel_x).pow(2) + self._dw(x, self.sobel_y).pow(2) + 1e-6
        ).mean(dim=1, keepdim=True)                         # (B,1,H,W)
        lap_mag  = self._dw(x, self.laplace).abs().mean(dim=1, keepdim=True)

        return x * torch.sigmoid(self.gate(torch.cat([grad_mag, lap_mag], dim=1)))


# ── 通道路径：FFT 高通聚合 + GRN ─────────────────────────────────────────────

class FreqChannelInteraction(nn.Module):
    """
    通道维度的高频注意力。

    全局池化 FFT 高通特征以生成通道注意力权重：
    不同通道对不同频率成分的响应强度不同，通过池化高通特征的 max/avg
    聚合各通道的全局高频能量，再由两个 1×1 GroupConv 生成通道门控。
    GRN 防止强势通道垄断（类 ConvNeXt V2 设计）。

    空间路径 + 通道路径共同作用，等效于 frequency-guided CBAM：
    空间路径解决"哪里有高频"，通道路径解决"哪些通道含高频"，两者正交互补。
    """
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super().__init__()
        self.ph    = patch[0]
        self.pw    = patch[1]
        self.ratio = ratio
        self.isdct = isdct

        self.channel1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)
        self.relu = nn.ReLU()
        self.grn  = GRN(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()

        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, (1, 1))
            aavgp = F.adaptive_avg_pool2d(x, (1, 1))
            ch = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return self.grn(x * torch.sigmoid(self.channel2x1(ch)))

        high  = _fft_highpass(x, self.ratio)
        amaxp = torch.sum(self.relu(F.adaptive_max_pool2d(high, (self.ph, self.pw))),
                          dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(F.adaptive_avg_pool2d(high, (self.ph, self.pw))),
                          dim=[2, 3]).view(n, c, 1, 1)
        ch    = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return self.grn(x * torch.sigmoid(self.channel2x1(ch)))


# ── Gate：FFT 低通结构门控（与高频路径精确互补） ─────────────────────────────

class StructureGate(nn.Module):
    """
    结构边界门控：Sobel( FFT_lowpass(x) )

    核心操作分两步：
      1. FFT 低通滤波：去除纹理、噪声、高频伪影，只保留平滑语义块
      2. Sobel 梯度：检测这些平滑语义块之间的过渡边界 = 结构边界

    与 GradSpatialInteraction 的本质区别：
      GradSpatialInteraction: Sobel(x)         → 所有空间变化（含纹理、噪声）
      StructureGate:          Sobel(lowpass(x)) → 仅语义区域间的结构边界
                                                  噪声/纹理不进入低通，不会被响应

    gate 在结构边界处高 → 高频增强在语义边界处发生，有明确物理意义
    gate 在平坦区域低  → 平坦背景不产生高频伪纹理

    1×1 Conv 作用：对通道级结构边界做可学习校准（非空间感知），
    空间感知由 Sobel 算子提供，不依赖 Conv 的感受野。
    零初始化：训练初期 gate ≈ 0.5，mod ≈ 0，近似恒等映射。
    """
    def __init__(self, in_channels: int, ratio=(0.25, 0.25)):
        super().__init__()
        self.ratio = ratio
        self.register_buffer("sobel_x",
            torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3),
            persistent=False)
        self.register_buffer("sobel_y",
            torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3),
            persistent=False)
        # 保持与旧版接口一致：按通道输入再聚合到 1 通道 gate。
        self.gate = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _dw(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, k.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = _fft_lowpass(x, self.ratio)                        # 平滑语义块 (B,C,H,W)
        gx  = self._dw(low, self.sobel_x)
        gy  = self._dw(low, self.sobel_y)
        struct_boundary = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)  # (B,C,H,W)
        return torch.sigmoid(self.gate(struct_boundary))         # (B,1,H,W) ∈ (0,1)


# ── 主模块 ────────────────────────────────────────────────────────────────────

class Structure_Guided_Frequency_Enhancement_Module(nn.Module):
    """
    SGFE v2：Structure-Guided Frequency Enhancement Module

    严格频域互补设计：
      高频路径（空间 + 通道）→ 提取"哪里/哪些通道有高频响应"（high）
      低频门控（FFT 低通）   → 提取"哪里有低频结构支撑"（gate）
    两路互补，无冗余。

    guided_high = high × (1 + α · gate)      gate ∈ (0,1)，仅增强不压制
    输出 = Conv3×3 + GRN(guided_high)
    """

    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True):
        super().__init__()
        self.spatial        = GradSpatialInteraction(in_channels)
        self.channel        = FreqChannelInteraction(in_channels, patch=patch,
                                                      ratio=ratio, isdct=isdct)
        self.structure_gate = StructureGate(in_channels, ratio=ratio)
        self.alpha          = nn.Parameter(torch.tensor(0.1))
        self.out            = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),   # 归一化到零均值，防止残差叠加引入色偏
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high        = self.spatial(x) + self.channel(x)
        gate        = self.structure_gate(x)          # (B,1,H,W) ∈ (0,1)，纯增强，不压制
        guided_high = high * (1 + self.alpha * gate)
        return self.out(guided_high)


if __name__ == '__main__':
    x = torch.randn(2, 320, 32, 32)
    m = Structure_Guided_Frequency_Enhancement_Module(320)
    y = m(x)
    print("in:", x.shape, "out:", y.shape)
    print(f"params: {sum(p.numel() for p in m.parameters())/1e3:.1f}K")

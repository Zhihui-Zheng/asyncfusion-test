import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_2d(x, norm='ortho'):
    return torch.fft.fft2(x, norm=norm).real


def idct_2d(x, norm='ortho'):
    return torch.fft.ifft2(x, norm=norm).real


def _frequency_mask(h, w, ratio, band, device):
    h0 = max(1, int(h * ratio[0]))
    w0 = max(1, int(w * ratio[1]))
    mask = torch.zeros((h, w), device=device, requires_grad=False)

    if band == "low":
        mask[:h0, :w0] = 1
    elif band == "high":
        mask[:] = 1
        mask[:h0, :w0] = 0
    else:
        raise ValueError(f"Unknown frequency band: {band}")

    return mask


class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio, isdct=True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct

        if not self.isdct:
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()

        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))

        x_freq = dct_2d(x, norm='ortho')
        mask = _frequency_mask(h, w, self.ratio, band="high", device=x.device)
        mask = mask.view(1, 1, h, w)
        high = idct_2d(x_freq * mask, norm='ortho')

        return x * high


class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super().__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct

        self.channel1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()

        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        x_freq = dct_2d(x, norm='ortho')
        mask = _frequency_mask(h, w, self.ratio, band="high", device=x.device)
        mask = mask.view(1, 1, h, w)
        high = idct_2d(x_freq * mask, norm='ortho')

        amaxp = F.adaptive_max_pool2d(high, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(high, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))


class LowFrequencyStructureGate(nn.Module):
    def __init__(self, in_channels, ratio, isdct=True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct
        self.gate = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        _, _, h, w = x.size()

        if self.isdct:
            x_freq = dct_2d(x, norm='ortho')
            mask = _frequency_mask(h, w, self.ratio, band="low", device=x.device)
            mask = mask.view(1, 1, h, w)
            structure = idct_2d(x_freq * mask, norm='ortho')
        else:
            structure = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        return torch.sigmoid(self.gate(structure))


class Structure_Guided_Frequency_Enhancement_Module(nn.Module):
    """
    SGFE: Structure-Guided Frequency Enhancement.

    高频分支负责提取边缘/纹理响应，低频结构分支生成空间 gate，
    用于选择性增强结构相关的高频细节并抑制无关高频噪声。
    """

    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True):
        super().__init__()

        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.structure_gate = LowFrequencyStructureGate(in_channels, ratio=ratio, isdct=isdct)

        # 双向调制强度。gate 初始为 0.5，因此 mod=2*gate-1 初始为 0，
        # guided_high 初始等于 high；alpha 非零保证 gate 分支可直接获得梯度。
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
        )

    def forward(self, x):
        high = self.spatial(x) + self.channel(x)
        gate = self.structure_gate(x)
        mod = 2 * gate - 1
        guided_high = high * (1 + self.alpha * mod)
        return self.out(guided_high)


if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)
    model = Structure_Guided_Frequency_Enhancement_Module(32)
    y = model(x)
    print("输入:", x.shape)
    print("输出:", y.shape)

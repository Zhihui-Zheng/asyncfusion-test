import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
FFT 负责拆高频；
Conv_k 负责学习不同频带的细节补偿；
Cat + 1×1 Conv 负责跨频带融合；
CA/SA 负责对融合后的高频残差做可靠性校准；
Residual Add 负责不破坏主干低频结构。
"""
def _valid_groups(channels: int, max_groups: int = 32) -> int:
    """
    找到一个可以整除 channels 的分组数。
    用于 grouped 1x1 conv，避免固定 groups=32 在某些通道数下报错。
    """
    for g in [32, 16, 8, 4, 2, 1]:
        if g <= max_groups and channels % g == 0:
            return g
    return 1


class ChannelRefine(nn.Module):
    """
    高频残差的通道校准。

    作用：
        频带融合之后，不同通道可能包含不同类型的高频信息：
        - 有些通道偏边缘；
        - 有些通道偏纹理；
        - 有些通道可能是噪声或伪影。

    CA 用来判断：
        哪些高频通道更可靠，哪些通道需要被削弱。

    注意：
        这里不是核心创新，只是对融合后的高频残差做轻量校准。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 16)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        # 零初始化最后一层，使初始时 tanh 输出接近 0，
        # 即 scale ≈ 1，不会一开始扰动主干特征。
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(torch.cat([avg, mx], dim=1))

        # tanh 输出范围 [-1, 1]，后面用 1 + gamma * attn 做残差式校准。
        return torch.tanh(attn)


class SpatialRefine(nn.Module):
    """
    高频残差的空间校准。

    作用：
        高频不一定都是有用细节，也可能是噪声或运动错位伪影。
        SA 用来判断：
        哪些空间位置的高频残差更可靠，哪些位置需要降低补偿强度。

    注意：
        它不是用红外低频去 gate 可见光高频，
        而是只根据融合后的高频残差自身做空间可靠性校准。
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # 初始时输出接近 0，scale ≈ 1。
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg, mx], dim=1))
        return torch.tanh(attn)


class HFRefinement(nn.Module):
    """
    高频残差细化模块。

    输入：
        已经经过分频带卷积补偿和 1x1 band fusion 的高频残差特征。

    作用：
        CA：校准哪些通道的高频残差可靠；
        SA：校准哪些空间位置的高频残差可靠。

    这里使用残差式缩放：
        x_refined = x * (1 + gamma_c * CA) * (1 + gamma_s * SA)

    这样比直接 sigmoid 相乘更稳定，因为初始状态接近恒等映射。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.ca = ChannelRefine(channels, reduction=reduction)
        self.sa = SpatialRefine(kernel_size=7)

        # 控制 CA/SA 的影响强度。
        # 初始化较小，避免训练初期过度改变高频残差。
        self.gamma_c = nn.Parameter(torch.tensor(0.1))
        self.gamma_s = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.ca(x)  # (B,C,1,1)
        sa = self.sa(x)  # (B,1,H,W)

        x = x * (1.0 + self.gamma_c * ca)
        x = x * (1.0 + self.gamma_s * sa)
        return x


class FourierBandResidualConvolutionRefine(nn.Module):
    """
    FBRC-Refine:
    Fourier Band Residual Convolution with High-frequency Refinement

    使用位置：
        用于红外-可见光已经混合后的特征 F_mix。

    总体思想：
        主网络负责低频主体结构和整体融合；
        本模块只补偿混合过程中被削弱的高频细节。

    处理流程：
        1. FFT 将 mixed feature 变换到频域；
        2. 只处理 cutoff 以上的高频区域；
        3. 将高频区域划分为多个 Fourier bands；
        4. 每个 band IFFT 回空间特征图 H_k；
        5. 每个 H_k 用专属 Conv_k 学习该尺度的高频补偿；
        6. Concat + 1x1 Conv 融合不同频带补偿结果；
        7. CA/SA 对融合后的高频残差进行可靠性校准；
        8. Projection 得到最终高频残差 Delta F_h；
        9. 残差加回 F_mix，不直接重建低频主体结构。

    输入：
        x: (B, C, H, W) 或 (B, T, C, H, W)

    输出：
        与 x 同形状。
    """

    def __init__(
        self,
        channels: int,
        num_bands: int = 3,
        cutoff: float = 0.22,
        tau: float = 0.06,
        kernel_size: int = 3,
        use_band_reprojection: bool = True,
        use_hf_refine: bool = True,
        reduction: int = 8,
        res_scale_init: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.num_bands = num_bands
        self.cutoff = cutoff
        self.tau = tau
        self.use_band_reprojection = use_band_reprojection
        self.use_hf_refine = use_hf_refine

        padding = kernel_size // 2
        groups = _valid_groups(channels)

        # ------------------------------------------------------------
        # Band-specific Conv_k
        # ------------------------------------------------------------
        # 每个频带一个专属卷积核。
        # Conv_k 的作用不是普通增强，而是学习该频率尺度下的细节补偿方式。
        #
        # 例如：
        #   Band 1：偏粗边缘和局部轮廓；
        #   Band 2：偏纹理和局部细节；
        #   Band 3：偏细边缘和锐度变化。
        #
        # Depthwise Conv 学局部高频模式；
        # Grouped 1x1 Conv 做轻量通道混合。
        # ------------------------------------------------------------
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=channels,
                    bias=False,
                ),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=1,
                    groups=groups,
                    bias=False,
                ),
            )
            for _ in range(num_bands)
        ])

        # ------------------------------------------------------------
        # Band Fusion
        # ------------------------------------------------------------
        # Conv_k 只负责各自频带内部的补偿。
        # 但是真实边缘/纹理通常跨多个频带共同表达，
        # 所以需要 Cat + 1x1 Conv 学习不同频带补偿结果如何组合。
        #
        # 输入通道：C * K
        # 输出通道：C
        # ------------------------------------------------------------
        self.band_fusion = nn.Conv2d(
            channels * num_bands,
            channels,
            kernel_size=1,
            groups=groups,
            bias=False,
        )

        # 高频残差细化：CA/SA 只作用于融合后的高频残差。
        if use_hf_refine:
            self.hf_refine = HFRefinement(channels, reduction=reduction)
        else:
            self.hf_refine = nn.Identity()

        # ------------------------------------------------------------
        # Projection
        # ------------------------------------------------------------
        # 将 refined high-frequency feature 投影成最终残差 Delta F_h。
        # 这里使用 1x1 Conv，保持轻量。
        #
        # 零初始化：
        #   训练初期 Delta F_h ≈ 0，
        #   因此输出 out ≈ x，
        #   不会破坏主干已经建模好的低频结构。
        # ------------------------------------------------------------
        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.proj.weight)

        # 残差强度。
        # 默认 0.0 更稳定，适合插入已有网络；
        # 如果从头训练，也可以设置为 0.1。
        self.res_scale = nn.Parameter(torch.tensor(res_scale_init))

    def _flatten_video(self, x: torch.Tensor):
        """
        支持两种输入：
            4D: (B, C, H, W)
            5D: (B, T, C, H, W)

        对视频特征，逐帧做频域分解和高频补偿。
        这样不会把单帧红外结构强行作为时序 gate。
        """
        if x.dim() == 4:
            return x, None

        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.reshape(b * t, c, h, w), (b, t, c, h, w)

        raise ValueError(f"Expected x to be 4D or 5D, but got {tuple(x.shape)}")

    def _restore_video(self, x: torch.Tensor, shape_info):
        if shape_info is None:
            return x

        b, t, c, h, w = shape_info
        return x.reshape(b, t, c, h, w)

    def _make_band_masks(self, h: int, w: int, device):
        """
        生成高频分频带 soft masks。

        重要解释：
            分频带不是裁剪空间位置，而是裁剪频率范围。

        cutoff 以下：
            视为低频主体结构，由主干网络负责，本模块不处理。

        cutoff 以上：
            划分成 num_bands 个高频频带。
            使用 soft mask，而不是 hard mask，减少频域硬截断造成的 ringing。
        """
        fy = torch.fft.fftfreq(h, device=device).view(h, 1)
        fx = torch.fft.rfftfreq(w, device=device).view(1, w // 2 + 1)

        radius = torch.sqrt(fy.pow(2) + fx.pow(2))
        radius = radius / math.sqrt(0.5 ** 2 + 0.5 ** 2)

        # 高频总开关：cutoff 以下接近 0，cutoff 以上接近 1。
        high_gate = torch.sigmoid((radius - self.cutoff) / self.tau)

        edges = torch.linspace(
            self.cutoff,
            1.0,
            self.num_bands + 1,
            device=device,
        )

        masks = []
        for i in range(self.num_bands):
            left = edges[i]
            right = edges[i + 1]

            # 第 i 个频带：
            # 频率大于 left 且小于 right 的部分被保留。
            left_gate = torch.sigmoid((radius - left) / self.tau)
            right_gate = torch.sigmoid((right - radius) / self.tau)

            band = left_gate * right_gate
            masks.append(band)

        masks = torch.stack(masks, dim=0)  # (K, H, W_r)

        # 归一化多个频带，避免相邻 soft band 重叠导致能量过大。
        masks = masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-6)

        # cutoff 以下低频不进入补偿。
        masks = masks * high_gate.unsqueeze(0)

        return masks  # (K, H, W_r)

    def _band_project(self, x: torch.Tensor, mask: torch.Tensor, h: int, w: int):
        """
        可选：频带回投影。

        为什么需要：
            H_k 是第 k 个频带；
            但 Conv_k 之后可能产生跨频成分。
            如果想让每个分支的输出仍主要属于自己的频带，
            就再做一次 FFT-mask-IFFT。

        这一步让模块逻辑更严谨：
            频带输入约束 + 频带输出约束。
        """
        dtype = x.dtype
        X = torch.fft.rfft2(x.float(), norm="ortho")
        X = X * mask.view(1, 1, h, w // 2 + 1)
        out = torch.fft.irfft2(X, s=(h, w), norm="ortho")
        return out.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, shape_info = self._flatten_video(x)

        dtype = x_flat.dtype
        b, c, h, w = x_flat.shape

        # FFT 使用 fp32 更稳定，尤其是混合精度训练时。
        X = torch.fft.rfft2(x_flat.float(), norm="ortho")  # (B,C,H,W_r), complex

        band_masks = self._make_band_masks(h, w, x_flat.device)  # (K,H,W_r)

        band_outputs = []

        for k in range(self.num_bands):
            mask = band_masks[k]  # (H,W_r)

            # 1. 取第 k 个高频频带。
            # 频带外的频率被置 0 或强烈抑制。
            X_k = X * mask.view(1, 1, h, w // 2 + 1)

            # 2. IFFT 回空间/特征图域。
            # H_k 仍然是完整 H×W 特征图，
            # 但内容主要来自第 k 个频率范围。
            H_k = torch.fft.irfft2(
                X_k,
                s=(h, w),
                norm="ortho",
            ).to(dtype)

            # 3. 频带专属卷积学习该尺度的细节补偿。
            D_k = self.band_convs[k](H_k)

            # 4. 可选频带回投影。
            # 防止 Conv_k 输出产生过多跨频成分。
            if self.use_band_reprojection:
                D_k = self._band_project(D_k, mask, h, w)

            band_outputs.append(D_k)

        # 5. Concat 所有频带补偿结果。
        #    shape: (B, C*K, H, W)
        detail_cat = torch.cat(band_outputs, dim=1)

        # 6. 跨频带融合。
        #    学习不同频带补偿结果之间如何组合。
        detail = self.band_fusion(detail_cat)  # (B,C,H,W)

        # 7. 高频残差细化。
        #    CA：哪些通道的高频可靠；
        #    SA：哪些空间位置的高频可靠。
        detail = self.hf_refine(detail)

        # 8. 投影成最终高频残差 Delta F_h。
        delta_h = self.proj(detail)

        # 9. 残差加回。
        #    主干低频结构保留在 x 中，本模块只补偿高频细节。
        out = x_flat + self.res_scale * delta_h

        return self._restore_video(out, shape_info)


if __name__ == "__main__":
    # 4D mixed feature
    x = torch.randn(2, 320, 32, 32)
    m = FourierBandResidualConvolutionRefine(
        channels=320,
        num_bands=3,
        cutoff=0.22,
        tau=0.06,
        use_band_reprojection=True,
        use_hf_refine=True,
        res_scale_init=0.0,
    )
    y = m(x)
    print("4D:", x.shape, y.shape)

    # 5D video mixed feature: (B,T,C,H,W)
    xv = torch.randn(2, 8, 320, 32, 32)
    yv = m(xv)
    print("5D:", xv.shape, yv.shape)

    params = sum(p.numel() for p in m.parameters())
    print(f"params: {params / 1e3:.2f}K")




## TemporalRestormer: Restormer with built-in temporal (3D) capability.
##
## 在原版 Restormer 的特征层内部插入轻量 Conv3d(kernel=(3,1,1)) 做时间混合，
## 使其具备时序一致能力；时序层采用恒等(Dirac)初始化，初始输出与原 2D Restormer
## 完全一致，因此可以直接加载 motion_deblurring 等 3 通道预训练权重而不损伤性能。
##
## - 所有 2D 子模块名称与原版 Restormer 保持一致，便于加载预训练。
## - forward 接收视频张量 (B, C, T, H, W)，输出同形状，数值范围 [0, 1]。

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from einops import rearrange

try:
    # 作为 basicsr 包的一部分导入时
    from .restormer_arch import Restormer
except ImportError:
    # 当 archs 目录被直接加入 sys.path、以顶层模块导入时
    # （用于绕开 basicsr.models.__init__ 的重型依赖，如未安装的 lmdb）
    from restormer_arch import Restormer


def _init_identity_conv3d(conv3d: nn.Conv3d) -> None:
    """把时间 Conv3d 初始化为恒等映射：中间时间抽头=单位阵，其余=0，bias=0。

    对 kernel=(3,1,1)、in==out 的 Conv3d：
        out[t] = sum_{c'} W[c, c', 1, 0, 0] * x[t, c'] = x[t, c]
    即初始时逐帧恒等，完全不改变 Restormer 的输出。
    """
    with torch.no_grad():
        nn.init.zeros_(conv3d.weight)
        out_c = conv3d.weight.shape[0]
        center = conv3d.weight.shape[2] // 2
        eye = torch.eye(out_c, dtype=conv3d.weight.dtype, device=conv3d.weight.device)
        conv3d.weight.data[:, :, center, 0, 0] = eye
        if conv3d.bias is not None:
            nn.init.zeros_(conv3d.bias)


class TemporalConv(nn.Module):
    """时间维 3D 卷积，恒等初始化，作用在 (B, C, T, H, W)。"""

    def __init__(self, channels: int, kernel_t: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel_t, 1, 1),
            padding=(kernel_t // 2, 0, 0),
            bias=True,
        )
        _init_identity_conv3d(self.conv)

    def forward(self, x_2d: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """x_2d: ((b t), c, h, w) -> 时间混合后再展平回 ((b t), c, h, w)。"""
        x = rearrange(x_2d, "(b t) c h w -> b c t h w", b=b, t=t)
        x = self.conv(x)
        return rearrange(x, "b c t h w -> (b t) c h w")


class TemporalRestormer(Restormer):
    """内置时序能力的 Restormer。

    在瓶颈(latent)与解码末端(decoder_level1 之后)各插入一个恒等初始化的时间 Conv3d，
    既能在低分辨率上做强语义时间融合，又能在高分辨率上对齐细节。
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
        use_temporal_high=True,
        use_grad_checkpoint=False,
    ):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=list(num_blocks),
            num_refinement_blocks=num_refinement_blocks,
            heads=list(heads),
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
        )

        self.use_temporal_high = bool(use_temporal_high)
        # 梯度检查点：训练时对各 TransformerBlock 序列重算激活以省显存
        self.use_grad_checkpoint = bool(use_grad_checkpoint)

        # 瓶颈时序层 (dim * 2**3 通道, 最低分辨率, 必选)
        self.temporal_latent = TemporalConv(int(dim * 2 ** 3))
        # 解码末端时序层 (decoder_level1 输出通道 = dim * 2**1, 高分辨率, 可选)
        if self.use_temporal_high:
            self.temporal_dec1 = TemporalConv(int(dim * 2 ** 1))

    def _run_seq(self, seq, x):
        """对一个 TransformerBlock 序列前向；训练时可启用梯度检查点省显存。"""
        if self.use_grad_checkpoint and self.training and x.requires_grad:
            return cp.checkpoint(seq, x, use_reentrant=False)
        return seq(x)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, C, T, H, W), 数值范围 [0, 1]
        b, c, t, h, w = video.shape
        inp_img = rearrange(video, "b c t h w -> (b t) c h w")

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self._run_seq(self.encoder_level1, inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self._run_seq(self.encoder_level2, inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self._run_seq(self.encoder_level3, inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self._run_seq(self.latent, inp_enc_level4)
        # ── 瓶颈时序融合 (恒等初始化) ──
        latent = self.temporal_latent(latent, b, t)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self._run_seq(self.decoder_level3, inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self._run_seq(self.decoder_level2, inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self._run_seq(self.decoder_level1, inp_dec_level1)

        # ── 高分辨率时序融合 (恒等初始化, 可选) ──
        if self.use_temporal_high:
            out_dec_level1 = self.temporal_dec1(out_dec_level1, b, t)

        out_dec_level1 = self._run_seq(self.refinement, out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        out = rearrange(out_dec_level1, "(b t) c h w -> b c t h w", b=b, t=t)
        return torch.clamp(out, 0.0, 1.0)

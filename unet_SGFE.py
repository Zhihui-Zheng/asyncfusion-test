"""
SGFEVid2VidSDUNet
================
Vid2VidSDUNet wrapper with TemporalSGFE blocks.

SGFE means Structure-Guided Frequency Enhancement:
  - per-frame 2D frequency enhancement guided by low-frequency structure
  - lightweight temporal Conv3d mixing for video features
"""

import os
import sys
import torch.nn as nn
from einops import rearrange

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from video_to_video.modules.unet_v2v import Vid2VidSDUNet

try:
    from Fusion_SGFE.block3 import FourierBandResidualConvolutionRefine
except ImportError:
    from block3 import FourierBandResidualConvolutionRefine


_BLOCK_OUT_CH = {
    0: 320,
    1: 320,
    2: 320,
    3: 320,
    4: 640,
    5: 640,
    6: 640,
    7: 1280,
    8: 1280,
    9: 1280,
    10: 1280,
    11: 1280,
}


class TemporalSGFE(nn.Module):
    """2+1D SGFE: per-frame block3 refinement + lightweight temporal mixing."""

    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(4, 4), isdct=True):
        super().__init__()
        # 使用 block3 的 FBRC-Refine 替换旧 SGFE63。
        # 统一沿用 sgfe_2d 属性名，保证外部脚本/旧 checkpoint key 前缀兼容。
        self.sgfe_2d = FourierBandResidualConvolutionRefine(
            channels=in_channels,
            res_scale_init=0.0,
        )
        self.temporal_mix = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, in_channels, (3, 1, 1), padding=(1, 0, 0)),
        )
        nn.init.zeros_(self.temporal_mix[-1].weight)
        nn.init.zeros_(self.temporal_mix[-1].bias)

    def forward(self, x, batch_size):
        # block3 返回 x + residual；外层注入逻辑是 x + hfp_out，
        # 因此这里改成只返回增量分支，避免重复叠加恒等项。
        sgfe_delta = self.sgfe_2d(x) - x
        num_frames = x.shape[0] // batch_size
        sgfe_5d = rearrange(
            sgfe_delta,
            '(b f) c h w -> b c f h w',
            b=batch_size,
            f=num_frames,
        )
        sgfe_5d = sgfe_5d + self.temporal_mix(sgfe_5d)
        return rearrange(sgfe_5d, 'b c f h w -> (b f) c h w')


class SGFEVid2VidSDUNet(Vid2VidSDUNet):
    """
    Insert TemporalSGFE blocks into selected Vid2VidSDUNet encoder blocks.

    The public argument names still use hfp_* to avoid breaking existing
    training scripts, but the module itself is SGFE.
    """

    def __init__(self,
                 hfp_block_indices=(1, 4, 7),
                 hfp_ratio=(0.25, 0.25),
                 hfp_patch=(4, 4),
                 hfp_use_dct=True,
                 hfp_serial=True,
                 hfp_init='ones',
                 **unet_kwargs):
        super().__init__(**unet_kwargs)

        self.hfp_block_indices = set(hfp_block_indices)
        self.hfp_ratio = hfp_ratio
        self.hfp_patch = hfp_patch
        self.hfp_use_dct = hfp_use_dct
        self.hfp_serial = hfp_serial
        self.hfp_init = hfp_init

        # Keep the old attribute name because dual_modal_fusion_model and older
        # checkpoints expect hfp_modules.
        self.hfp_modules = nn.ModuleDict()
        for idx in hfp_block_indices:
            ch = _BLOCK_OUT_CH[idx]
            self.hfp_modules[str(idx)] = TemporalSGFE(
                in_channels=ch,
                ratio=hfp_ratio,
                patch=hfp_patch,
                isdct=hfp_use_dct,
            )

        if hfp_init == 'ones':
            self._ones_init_hfp()
        elif hfp_init == 'zeros':
            self._zeros_init_hfp()

    def _ones_init_hfp(self):
        for module in self.hfp_modules.values():
            sgfe_2d = module.sgfe_2d
            if hasattr(sgfe_2d, "alpha") and hasattr(sgfe_2d, "out"):
                for m in sgfe_2d.modules():
                    if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                        nn.init.zeros_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.GroupNorm):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

                nn.init.constant_(sgfe_2d.alpha, 0.1)
                out_conv = sgfe_2d.out[0]
                nn.init.dirac_(out_conv.weight)
                if out_conv.bias is not None:
                    nn.init.zeros_(out_conv.bias)
            else:
                # block3 分支：保持近似恒等，再把残差强度设为较小正值。
                for p in sgfe_2d.parameters():
                    nn.init.zeros_(p)
                if hasattr(sgfe_2d, "res_scale"):
                    nn.init.constant_(sgfe_2d.res_scale, 0.1)

    def _zeros_init_hfp(self):
        for module in self.hfp_modules.values():
            sgfe_2d = module.sgfe_2d
            if hasattr(sgfe_2d, "alpha") and hasattr(sgfe_2d, "out"):
                nn.init.constant_(sgfe_2d.alpha, 0.1)
                out_conv = sgfe_2d.out[0]
                nn.init.zeros_(out_conv.weight)
                if out_conv.bias is not None:
                    nn.init.zeros_(out_conv.bias)
            else:
                # 你要求的“全0初始化”：block3 参数全部归零，残差增量起始为 0。
                for p in sgfe_2d.parameters():
                    nn.init.zeros_(p)

    def get_hfp_parameters(self):
        """Return SGFE parameters. Method name is kept for script compatibility."""
        return list(self.hfp_modules.parameters())

    def freeze_base_unfreeze_hfp(self):
        """Freeze base UNet and train SGFE modules only."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.hfp_modules.parameters():
            param.requires_grad = True

    def extra_repr(self):
        return (
            f"sgfe_blocks={sorted(self.hfp_block_indices)}, "
            f"mode=parallel_3d, init={self.hfp_init}, "
            f"ratio={self.hfp_ratio}, patch={self.hfp_patch}, "
            f"use_dct={self.hfp_use_dct}"
        )


# Compatibility alias for older imports.
HFPVid2VidSDUNet = SGFEVid2VidSDUNet

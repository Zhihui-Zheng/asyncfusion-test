"""
IR 光流传播模块 (IR Flow Propagation)

利用 VI (可见光) 帧间光流将 ControlNet 输出的 skip connection 做多尺度 warp，
使静态 IR 的控制信号在空间上对齐到每帧 VI 的运动位置。

策略 B 数据流:
  静态 IR → ControlNet (frozen, 输入分布不变) → 多尺度 skip connections
  VI pixel frames → RAFT (frozen) → 帧间光流 → 累积光流
  每尺度 skip connection × 缩放后的累积光流 → flow_warp → 时序对齐的控制信号 → UNet

同时保留原始 forward() 接口 (warp IR latent) 以便兼容旧流程。
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros',
              align_corners=True):
    """
    用光流对图像/特征做 warp (来自 ProPainter)。

    Args:
        x:    (N, C, H, W)
        flow: (N, H, W, 2)  像素级位移 (dx, dy)，不是归一化坐标
    Returns:
        (N, C, H, W) warped 结果
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(
            f'Spatial size mismatch: x {x.size()[-2:]} vs flow {flow.size()[1:3]}')
    _, _, h, w = x.size()
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device),
        torch.arange(0, w, device=device),
        indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(x, grid_flow, mode=interpolation,
                         padding_mode=padding_mode,
                         align_corners=align_corners)


def _init_raft(model_path, device='cpu'):
    """
    加载 ProPainter 的 RAFT 模型 (冻结)。

    权重下载:
      https://huggingface.co/zhuang/ProPainter/resolve/main/raft-things.pth
      → 放到 Fusion5/ProPainter/weights/raft-things.pth
    """
    propainter_root = os.path.join(
        os.path.dirname(__file__), 'ProPainter')
    if propainter_root not in sys.path:
        sys.path.insert(0, propainter_root)

    from RAFT import RAFT

    args = argparse.Namespace(
        raft_model=model_path,
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )
    model = nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.module
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class IRFlowPropagation(nn.Module):
    """
    用 VI 像素帧的光流将静态 IR latent 传播为时序对齐的多帧 IR latent。

    forward(ir_latent, vi_frames):
        ir_latent : (B, C, T, H_lat, W_lat)  — 同一帧 IR 复制 T 次
        vi_frames : (B, T, 3, H_pix, W_pix)  — VI 像素帧，范围 [-1, 1]
        返回      : (B, C, T, H_lat, W_lat)  — 时序对齐的 IR latent
    """

    def __init__(self, raft_model_path, latent_scale=4, raft_iters=12):
        """
        Args:
            raft_model_path: raft-things.pth 权重路径
            latent_scale:    VAE 下采样倍数 (4x TVT / 8x SVD)
            raft_iters:      RAFT 迭代次数 (越多越精确，越慢)
        """
        super().__init__()
        self.latent_scale = latent_scale
        self.raft_iters = raft_iters
        self.raft = _init_raft(raft_model_path)

    @torch.no_grad()
    def estimate_flow(self, vi_frames):
        """
        VI 像素帧 → 相邻帧间的前向光流。

        Args:
            vi_frames: (B, T, 3, H, W), 范围 [-1, 1]
        Returns:
            flows: list of (B, 2, H, W), 长度 T-1
        """
        B, T, C, H, W = vi_frames.shape
        vi_255 = ((vi_frames.float() + 1.0) * 0.5 * 255.0).clamp(0, 255)

        flows = []
        for t in range(T - 1):
            img1 = vi_255[:, t].contiguous()   # (B, 3, H, W)
            img2 = vi_255[:, t + 1].contiguous()
            _, flow_up = self.raft(img1, img2, iters=self.raft_iters,
                                  test_mode=True)
            flows.append(flow_up)  # (B, 2, H, W)
        return flows

    def propagate_ir(self, ir_latent, flows_latent):
        """
        用 latent 分辨率的光流链式传播 IR latent。

        Args:
            ir_latent:    (B, C, T, H_lat, W_lat)
            flows_latent: list of (B, 2, H_lat, W_lat), 长度 T-1
        Returns:
            (B, C, T, H_lat, W_lat)  时序对齐的 IR latent
        """
        ir_frames = [ir_latent[:, :, 0]]  # 第 0 帧作为锚点，不动

        for t, flow in enumerate(flows_latent):
            # flow: (B, 2, H, W) → (B, H, W, 2) for flow_warp
            prev = ir_frames[-1]
            warped = flow_warp(prev, flow.permute(0, 2, 3, 1))
            ir_frames.append(warped)

        return torch.stack(ir_frames, dim=2)  # (B, C, T, H, W)

    @torch.no_grad()
    def estimate_bidirectional_flows(self, vi_frames):
        """
        估计 VI 帧间的前向 + 后向光流, 用于 anchor 在窗口中部时做双向 warp。

        Args:
            vi_frames: (B, T, 3, H, W), [-1, 1]
        Returns:
            flows_fwd: list of (B, 2, H, W), 长度 T-1, RAFT(VI[t], VI[t+1])
            flows_bwd: list of (B, 2, H, W), 长度 T-1, RAFT(VI[t+1], VI[t])
        """
        B, T, C, H, W = vi_frames.shape
        vi_255 = ((vi_frames.float() + 1.0) * 0.5 * 255.0).clamp(0, 255)

        flows_fwd, flows_bwd = [], []
        for t in range(T - 1):
            img1 = vi_255[:, t].contiguous()
            img2 = vi_255[:, t + 1].contiguous()
            _, fwd = self.raft(img1, img2, iters=self.raft_iters, test_mode=True)
            _, bwd = self.raft(img2, img1, iters=self.raft_iters, test_mode=True)
            flows_fwd.append(fwd)
            flows_bwd.append(bwd)
        return flows_fwd, flows_bwd

    def warp_ctrl_signals(self, ctrl_signals, flows_fwd, pixel_h, pixel_w,
                          num_frames, anchor_idx=0, flows_bwd=None):
        """
        用累积光流 warp ControlNet 输出的多尺度 skip connections (策略 B 核心, 支持双向)。

        ControlNet 处理静态 IR 后, 8 帧 skip connection 中 IR 信息的空间位置相同 ──
        都在数据加载时取的那一张 IR 的位置上 (即与 VI[anchor_idx] 同名的那帧 IR).
        本方法以 anchor_idx 为锚点, 用 VI 帧间累积光流将其它帧的 skip 做空间对齐:
          - t == anchor_idx : 不动
          - t >  anchor_idx : 用 flows_fwd 累加 VI[anchor]→VI[t]
          - t <  anchor_idx : 用 flows_bwd 累加 VI[anchor]→VI[t]

        Args:
            ctrl_signals : list of (B*F, C_i, H_i, W_i)
            flows_fwd    : list of (B, 2, H_pix, W_pix), 长度 T-1
                           RAFT(VI[t], VI[t+1]) 前向光流
            pixel_h      : int, VI 像素帧高度
            pixel_w      : int, VI 像素帧宽度
            num_frames   : int, 帧数 T
            anchor_idx   : int, 锚点 0-indexed 位置 (默认 0 兼容旧行为)
            flows_bwd    : list of (B, 2, H_pix, W_pix), 长度 T-1, 可选
                           RAFT(VI[t+1], VI[t]) 后向光流
                           anchor_idx > 0 时若不传, 会用前向光流取负作为兜底近似
                           (大位移/遮挡处不准, 推荐传双向光流)
        Returns:
            list of (B*F, C_i, H_i, W_i) — warp 后的控制信号
        """
        if num_frames <= 1 or len(flows_fwd) == 0:
            return ctrl_signals

        anchor_idx = int(max(0, min(anchor_idx, num_frames - 1)))

        # 后向光流兜底: 如果没传且需要, 用前向光流取负 (近似, 不推荐)
        if anchor_idx > 0 and flows_bwd is None:
            flows_bwd = [-f for f in flows_fwd]

        # 计算从 anchor_idx 出发到每个目标帧的累积位移 (像素级)
        cum_flows = [None] * num_frames
        cum_flows[anchor_idx] = torch.zeros_like(flows_fwd[0])

        cum = torch.zeros_like(flows_fwd[0])
        for t in range(anchor_idx, num_frames - 1):
            cum = cum + flows_fwd[t]              # VI[anchor]→VI[t+1]
            cum_flows[t + 1] = cum.clone()

        cum = torch.zeros_like(flows_fwd[0])
        for t in range(anchor_idx, 0, -1):
            cum = cum + flows_bwd[t - 1]          # VI[anchor]→VI[t-1]
            cum_flows[t - 1] = cum.clone()

        warped_signals = []
        for ctrl in ctrl_signals:
            BF, C, H_f, W_f = ctrl.shape
            B = BF // num_frames
            orig_dtype = ctrl.dtype

            ctrl_5d = ctrl.float().view(B, num_frames, C, H_f, W_f)
            frames_out = [None] * num_frames

            for t in range(num_frames):
                if t == anchor_idx:
                    frames_out[t] = ctrl_5d[:, t]
                    continue

                cum_flow = cum_flows[t]
                flow_feat = F.interpolate(
                    cum_flow.float(), size=(H_f, W_f),
                    mode='bilinear', align_corners=True)
                flow_feat[:, 0] *= (W_f / pixel_w)
                flow_feat[:, 1] *= (H_f / pixel_h)

                warped = flow_warp(
                    ctrl_5d[:, t],
                    flow_feat.permute(0, 2, 3, 1))
                frames_out[t] = warped

            result = torch.stack(frames_out, dim=1).view(BF, C, H_f, W_f)
            warped_signals.append(result.to(orig_dtype))

        return warped_signals

    def forward(self, ir_latent, vi_frames):
        """
        原始接口 (兼容旧流程): warp IR latent 本身。
        策略 B 下不再由 forward() 调用，改用 warp_ctrl_signals()。

        Args:
            ir_latent : (B, C, T, H_lat, W_lat)  同一帧 IR 复制 T 次
            vi_frames : (B, T, 3, H_pix, W_pix)  VI 像素帧, [-1, 1]
        Returns:
            (B, C, T, H_lat, W_lat)  时序对齐的 IR latent
        """
        B, C, T, H_lat, W_lat = ir_latent.shape

        # 1. 估计 VI 帧间光流 (pixel-level, frozen)
        flows_pixel = self.estimate_flow(vi_frames)

        # 2. 下采样光流到 latent 分辨率
        flows_latent = []
        for flow in flows_pixel:
            flow_down = F.interpolate(
                flow.float(), size=(H_lat, W_lat),
                mode='bilinear', align_corners=True)
            flow_down = flow_down / self.latent_scale
            flows_latent.append(flow_down)

        # 3. 链式传播 IR latent
        return self.propagate_ir(ir_latent, flows_latent)

"""
频域解耦时序一致性损失 (Latent Space)

核心思想: 只对低频分量施加时序约束，保护高频细节不被时序平滑抹掉。

与原论文 (像素空间) 的区别:
  - 我们有 GT latent，直接用 GT 帧间差做参考，不需要偏移估计
  - 逐通道独立做低通滤波，不做通道均值 (保留 latent 4 通道各自的信息)
  - 高斯核适配 latent 分辨率 (5×5 σ=1.0 ≈ 像素空间 20×20 for 4x VAE)

L_tc = L_diff + λ_grad * L_grad

  - L_diff : 低频帧间差一致性 — pred 的帧间变化 ≈ GT 的帧间变化 (Charbonnier)
  - L_grad : 低频帧间差的空间梯度一致性 — 局部边缘也对齐 (Charbonnier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentTemporalConsistencyLoss(nn.Module):

    def __init__(self, kernel_size=5, sigma=1.0, eps=1e-3, grad_weight=0.3):
        """
        Args:
            kernel_size: 高斯低通滤波核大小
            sigma:       高斯核标准差
            eps:         Charbonnier ε (数值稳定 + 抗异常值)
            grad_weight: 空间梯度项权重 (论文附录 0.3)
        """
        super().__init__()
        self.eps = eps
        self.grad_weight = grad_weight

        kernel = self._make_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('gaussian_kernel', kernel)

    @staticmethod
    def _make_gaussian_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = g.outer(g)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)          # (1, 1, kH, kW)

    def _low_pass(self, x):
        """
        逐通道高斯低通滤波.
        x: (B, C, H, W) → (B, C, H, W)
        """
        B, C, H, W = x.shape
        pad = self.gaussian_kernel.shape[-1] // 2
        k = self.gaussian_kernel.to(dtype=x.dtype)
        k = k.expand(C, -1, -1, -1)                     # (C, 1, kH, kW)
        return F.conv2d(x, k, padding=pad, groups=C)

    def _charbonnier(self, x):
        """Charbonnier penalty: sqrt(x² + ε²)，对异常值比 MSE 更鲁棒"""
        return torch.sqrt(x ** 2 + self.eps ** 2).mean()

    @staticmethod
    def _spatial_gradient(x):
        """空间梯度 (前向差分)"""
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    def forward(self, pred_fused, gt_latent):
        """
        Args:
            pred_fused: 预测融合 latent (B, C, T, H, W) — 有梯度
                        = vi_latent + pred_velocity
            gt_latent:  GT 融合 latent  (B, C, T, H, W) — 无梯度
        Returns:
            L_tc: 标量
        """
        B, C, T, H, W = pred_fused.shape

        if T < 2:
            return pred_fused.sum() * 0.0

        # ── Step 1: 低通滤波 (逐通道，保留所有 latent 通道信息) ──────────
        # reshape → (B*T, C, H, W) 统一做卷积
        pred_flat = pred_fused.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        gt_flat   = gt_latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        pred_lf = self._low_pass(pred_flat).reshape(B, T, C, H, W)
        gt_lf   = self._low_pass(gt_flat).reshape(B, T, C, H, W)

        # ── Step 2: 帧间差 (temporal difference) ────────────────────────
        # (B, T-1, C, H, W)
        pred_diff = pred_lf[:, 1:] - pred_lf[:, :-1]
        gt_diff   = gt_lf[:, 1:]   - gt_lf[:, :-1]

        # ── Step 3: L_diff — 低频帧间差一致性 (Charbonnier) ─────────────
        L_diff = self._charbonnier(pred_diff - gt_diff.detach())

        # ── Step 4: L_grad — 帧间差的空间梯度也要一致 ───────────────────
        # 把 (B, T-1) 合并到 batch 维
        pred_diff_flat = pred_diff.reshape(B * (T - 1), C, H, W)
        gt_diff_flat   = gt_diff.reshape(B * (T - 1), C, H, W)

        pred_gx, pred_gy = self._spatial_gradient(pred_diff_flat)
        gt_gx,   gt_gy   = self._spatial_gradient(gt_diff_flat)

        L_grad = 0.5 * (
            self._charbonnier(pred_gx - gt_gx.detach())
          + self._charbonnier(pred_gy - gt_gy.detach())
        )

        return L_diff + self.grad_weight * L_grad

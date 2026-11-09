"""Standalone degradation-aware wavelet feature enhancement."""

import torch
from torch import nn
from torch.nn import functional as F

from .degradation import (
    Conv2dAct,
    ConvBNAct,
    DWConvBNAct,
    DegradationCueEstimator,
    _conv2d_fp32,
)
from .wavelet import HaarDWT


class DWM(nn.Module):
    """Enhance BCHW features with a wavelet residual while preserving shape and dtype.
    Keep module parameters and buffers in FP32."""

    def __init__(
        self,
        channels: int,
        wave_dim: int | None = None,
        gain_floor: float = 0.25,
    ) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer.")
        if wave_dim is None:
            wave_dim = max(channels // 2, 64)
        if isinstance(wave_dim, bool) or not isinstance(wave_dim, int) or wave_dim <= 0:
            raise ValueError("wave_dim must be a positive integer or None.")
        if not 0.0 <= gain_floor <= 1.0:
            raise ValueError("gain_floor must be between 0 and 1.")
        self.channels = channels
        self.wave_dim = wave_dim
        self.gain_floor = float(gain_floor)

        self.input_proj = (
            nn.Conv2d(channels, wave_dim, kernel_size=1, bias=False)
            if channels != wave_dim
            else nn.Identity()
        )
        self.dwt = HaarDWT()
        self.dce = DegradationCueEstimator(wave_dim)
        self.dir_hf_gate = nn.Sequential(
            ConvBNAct(wave_dim * 5, wave_dim, k=1),
            Conv2dAct(wave_dim, wave_dim * 3, k=1, bias=True),
            nn.Sigmoid(),
        )
        self.hf_reliability_gate = nn.Sequential(
            ConvBNAct(wave_dim * 4, wave_dim, k=1),
            Conv2dAct(wave_dim, wave_dim, k=1, bias=True),
            nn.Sigmoid(),
        )
        self.lf_gate = nn.Sequential(
            ConvBNAct(wave_dim * 2, wave_dim, k=1),
            Conv2dAct(wave_dim, wave_dim, k=1, bias=True),
            nn.Sigmoid(),
        )
        self.lf_residual = DWConvBNAct(wave_dim, k=3)
        self.fusion_gate = nn.Sequential(
            ConvBNAct(wave_dim * 3, wave_dim, k=1),
            Conv2dAct(wave_dim, wave_dim, k=1, bias=True),
            nn.Sigmoid(),
        )
        self.wave_out_proj = ConvBNAct(wave_dim, wave_dim, k=1)
        self.wave_to_c = nn.Conv2d(wave_dim, channels, kernel_size=1, bias=False)
        self.log_lambda0 = nn.Parameter(torch.tensor(-0.2))
        self.log_lambda_z = nn.Parameter(torch.tensor(0.4))
        nn.init.zeros_(self.wave_to_c.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input plus a reliability-weighted wavelet residual."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x must be a 4D tensor with shape (B, C, H, W).")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype.")
        if any(size == 0 for size in x.shape):
            raise ValueError("All dimensions of x must be positive.")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, got {x.shape[1]}.")
        out_dtype = x.dtype
        with torch.amp.autocast(x.device.type, enabled=False):
            y = self._forward_impl(x.float())
        return y.to(dtype=out_dtype)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        feat = x if isinstance(self.input_proj, nn.Identity) else _conv2d_fp32(x, self.input_proj)

        subbands = self.dwt(feat)
        f_ll, f_lh, f_hl, f_hh = subbands
        d, e_h, e_l, _, _ = self.dce(subbands)

        gates = self.dir_hf_gate(torch.cat([f_ll, f_lh, f_hl, f_hh, d], dim=1))
        g_lh, g_hl, g_hh = torch.chunk(gates, chunks=3, dim=1)
        f_hf = g_lh * f_lh + g_hl * f_hl + g_hh * f_hh
        r_h = self.hf_reliability_gate(torch.cat([f_hf, d, e_h, e_l], dim=1))
        f_hf_d = r_h * f_hf
        r_l = self.lf_gate(torch.cat([f_ll, d], dim=1))
        f_ll_d = f_ll + r_l * self.lf_residual(f_ll)
        alpha = self.fusion_gate(torch.cat([f_ll_d, f_hf_d, d], dim=1))
        f_wave = self.wave_out_proj(alpha * f_hf_d + (1.0 - alpha) * f_ll_d)
        f_wave = _conv2d_fp32(f_wave, self.wave_to_c)
        f_wave = F.interpolate(f_wave, size=(height, width), mode="bilinear", align_corners=False)

        rel = r_h.mean(dim=1, keepdim=True)
        rel = F.interpolate(rel, size=(height, width), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        lambda0 = F.softplus(self.log_lambda0.float()) + 1e-4
        lambda_z = (F.softplus(self.log_lambda_z.float()) + 1e-4) * (
            self.gain_floor + (1.0 - self.gain_floor) * rel
        )
        gain = lambda_z / (lambda0 + lambda_z)
        return x + gain * f_wave

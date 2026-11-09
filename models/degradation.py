"""Wavelet statistics and learned degradation cues for the feature branch."""

import torch
from torch import nn
from torch.nn import functional as F


def _conv2d_fp32(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """Evaluate convolution in FP32, including inside an autocast context."""
    with torch.amp.autocast(x.device.type, enabled=False):
        return F.conv2d(
            x.float(),
            conv.weight.float(),
            conv.bias.float() if conv.bias is not None else None,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )


def _bn_fp32(x: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
    """Evaluate the feature branch's functional BatchNorm operation in FP32."""
    with torch.amp.autocast(x.device.type, enabled=False):
        return F.batch_norm(
            x.float(),
            bn.running_mean.float() if bn.running_mean is not None else None,
            bn.running_var.float() if bn.running_var is not None else None,
            bn.weight.float() if bn.weight is not None else None,
            bn.bias.float() if bn.bias is not None else None,
            bn.training,
            bn.momentum,
            bn.eps,
        )


class ConvBNAct(nn.Module):
    """Convolution, BatchNorm and optional SiLU with FP32 computation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(_bn_fp32(_conv2d_fp32(x, self.conv), self.bn))


class DWConvBNAct(nn.Module):
    """Depthwise convolution, BatchNorm and SiLU for spatial mixing."""

    def __init__(self, channels: int, k: int = 3) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, k, 1, k // 2, groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(_bn_fp32(_conv2d_fp32(x, self.dw), self.bn))


class Conv2dAct(nn.Module):
    """Convolution and optional activation with FP32 computation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int = 0,
        bias: bool = True,
        act: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=bias)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(_conv2d_fp32(x, self.conv))


class DegradationCueEstimator(nn.Module):
    """Estimate local degradation cues from wavelet statistics in FP32."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer.")
        self.channels = channels
        self.net = nn.Sequential(
            ConvBNAct(channels * 4, channels, k=1),
            Conv2dAct(channels, channels, k=1, bias=True),
        )

    def forward(
        self,
        subbands: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return FP32 (D, E_h, E_l, C_l, B) from (LL, LH, HL, HH)."""
        if not isinstance(subbands, (tuple, list)) or len(subbands) != 4:
            raise ValueError("subbands must contain four tensors: (LL, LH, HL, HH).")
        for band in subbands:
            if not isinstance(band, torch.Tensor) or band.ndim != 4:
                raise ValueError("Each subband must be a 4D tensor (B, C, H, W).")
            if not band.is_floating_point():
                raise TypeError("Subbands must have floating-point dtypes.")
            if any(size == 0 for size in band.shape):
                raise ValueError("All subband dimensions must be positive.")
            if band.shape != subbands[0].shape or band.device != subbands[0].device:
                raise ValueError("All subbands must have the same shape and device.")
        if subbands[0].shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels per subband.")

        with torch.amp.autocast(subbands[0].device.type, enabled=False):
            f_ll, f_lh, f_hl, f_hh = (band.float() for band in subbands)
            e_h = f_lh.abs() + f_hl.abs() + f_hh.abs()
            e_l = f_ll.abs()
            local_mean = F.avg_pool2d(f_ll, kernel_size=3, stride=1, padding=1)
            c_l = (f_ll - local_mean).abs()
            blur_cue = torch.log1p(e_l / (e_h + 1e-6))
            d = torch.tanh(self.net(torch.cat([e_h, e_l, c_l, blur_cue], dim=1)))
            return d, e_h, e_l, c_l, blur_cue

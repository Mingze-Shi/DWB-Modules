"""Fixed Haar decomposition for feature tensors, using PyTorch only."""

import torch
from torch import nn
from torch.nn import functional as F


class HaarDWT(nn.Module):
    """Compute FP32 Haar subbands from BCHW features."""

    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ) * 0.5
        self.register_buffer("filters", kernels.unsqueeze(1))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return subbands in ``(LL, LH, HL, HH)`` order."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x must be a 4D tensor with shape (B, C, H, W).")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype.")
        if any(size == 0 for size in x.shape):
            raise ValueError("All dimensions of x must be positive.")

        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float()
            batch, channels, height, width = x.shape
            if height % 2 or width % 2:
                x = F.pad(x, (0, width % 2, 0, height % 2), mode="replicate")
            weights = self.filters.float().repeat(channels, 1, 1, 1)
            y = F.conv2d(x, weights, stride=2, padding=0, groups=channels)
            y = y.view(batch, channels, 4, y.shape[-2], y.shape[-1])
            return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]

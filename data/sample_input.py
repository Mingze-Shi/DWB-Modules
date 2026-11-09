"""Generate random CPU features with user-supplied dimensions."""

import torch


def make_sample_features(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Return FP32 BCHW features using a local random generator."""
    dimensions = {
        "batch_size": batch_size,
        "channels": channels,
        "height": height,
        "width": width,
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    generator = torch.Generator(device="cpu")
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return torch.randn(
        batch_size,
        channels,
        height,
        width,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )

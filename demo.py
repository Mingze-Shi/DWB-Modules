"""Run the feature modules on CPU with user-sized random input."""

from __future__ import annotations

import argparse

import torch

from data.sample_input import make_sample_features
from models.degradation import DegradationCueEstimator
from models.dwm import DWM
from models.wavelet import HaarDWT


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, required=True)
    parser.add_argument("--channels", type=positive_int, required=True)
    parser.add_argument("--height", type=positive_int, required=True)
    parser.add_argument("--width", type=positive_int, required=True)
    parser.add_argument("--wave-dim", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(2)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    x = make_sample_features(args.batch_size, args.channels, args.height, args.width, args.seed)
    wavelet = HaarDWT().cpu().eval()
    degradation = DegradationCueEstimator(args.channels).cpu().eval()
    module = DWM(args.channels, wave_dim=args.wave_dim).cpu().eval()

    with torch.inference_mode():
        subbands = wavelet(x)
        descriptor, high_energy, low_energy, contrast, blur_cue = degradation(subbands)
        output = module(x)
        max_difference = (output - x).abs().max().item()

    print("DWB feature modules | CPU | synthetic input | untrained weights")
    print(f"Input:          {tuple(x.shape)} ({x.dtype})")
    for name, band in zip(("LL", "LH", "HL", "HH"), subbands):
        print(f"Haar {name}:        {tuple(band.shape)} ({band.dtype})")
    print(f"Descriptor D:   {tuple(descriptor.shape)}")
    print(f"High energy:    {tuple(high_energy.shape)}")
    print(f"Low energy:     {tuple(low_energy.shape)}")
    print(f"Local contrast: {tuple(contrast.shape)}")
    print(f"Blur cue:       {tuple(blur_cue.shape)}")
    print(f"DWM output:     {tuple(output.shape)} ({output.dtype})")
    print(f"Initial max |output - input|: {max_difference:.8g}")


if __name__ == "__main__":
    main()

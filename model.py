"""Satellite image super-resolution mapping network."""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    """Residual feature block for stable high-frequency feature extraction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SatelliteSRNet(nn.Module):
    """Residual sub-pixel network for 2x multispectral image super-resolution."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int | None = None,
        features: int = 64,
        num_res_blocks: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or features <= 0 or num_res_blocks <= 0:
            raise ValueError("Channel, feature, and residual-block counts must be positive")

        out_channels = in_channels if out_channels is None else out_channels
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        self.head = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(
            *(ResBlock(features) for _ in range(num_res_blocks)),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample a normalized multispectral tensor by a factor of two."""
        shallow_features = self.head(x)
        features = shallow_features + self.trunk(shallow_features)
        return self.output(self.upsample(features))


if __name__ == "__main__":
    dummy_input = torch.rand(2, 4, 64, 64)
    model = SatelliteSRNet(in_channels=4, out_channels=4)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    expected_shape = (2, 4, 128, 128)
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(dummy_output.shape)}")
    if tuple(dummy_output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: expected {expected_shape}, "
            f"got {tuple(dummy_output.shape)}"
        )
    if not torch.all((dummy_output >= 0) & (dummy_output <= 1)):
        raise RuntimeError("Output values are outside the expected [0, 1] range")
    print("SatelliteSRNet forward-pass verification succeeded.")

"""Train SatelliteSRNet on paired multispectral raster patches."""

from __future__ import annotations

# Google Colab persistence:
# 1. Mount Drive once per session:
#       from google.colab import drive
#       drive.mount("/content/drive")
# 2. Copy the generated archive from Drive into the Colab workspace, for example:
#       !cp /content/drive/MyDrive/srm/dataset_patches.npz /content/dataset_patches.npz
# 3. Save this script's best_model.pth back to Drive after training:
#       !cp best_model.pth /content/drive/MyDrive/srm/best_model.pth
# Keeping both the input archive and checkpoint in Drive protects them from
# Google Colab runtime/session resets.

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import SatelliteSRNet


class SatellitePatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of normalized, paired LR and HR multispectral patches."""

    def __init__(self, npz_path: str | Path) -> None:
        archive_path = Path(npz_path)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Patch archive not found: {archive_path}")

        with np.load(archive_path) as archive:
            missing = {"lr_patches", "hr_patches"} - set(archive.files)
            if missing:
                raise KeyError(f"Archive is missing required arrays: {sorted(missing)}")
            lr_patches = np.asarray(archive["lr_patches"], dtype=np.float32)
            hr_patches = np.asarray(archive["hr_patches"], dtype=np.float32)

        if lr_patches.ndim != 4 or hr_patches.ndim != 4:
            raise ValueError("Both patch arrays must have shape (N, C, H, W)")
        if lr_patches.shape[0] != hr_patches.shape[0]:
            raise ValueError("LR and HR arrays must contain the same number of patches")
        if lr_patches.shape[1] != hr_patches.shape[1]:
            raise ValueError("LR and HR arrays must contain the same number of bands")
        if lr_patches.shape[0] == 0:
            raise ValueError("The patch archive contains no training pairs")

        self.lr_patches = torch.from_numpy(lr_patches)
        self.hr_patches = torch.from_numpy(hr_patches)

    def __len__(self) -> int:
        return self.lr_patches.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lr_patches[index], self.hr_patches[index]


def calculate_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return PSNR in decibels for tensors normalized to [0, 1]."""
    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def train(
    dataset_path: str | Path = "dataset_patches.npz",
    epochs: int = 25,
    checkpoint_path: str | Path = "best_model.pth",
) -> None:
    """Train the model and persist the weights with the lowest epoch loss."""
    dataset = SatellitePatchDataset(dataset_path)
    if len(dataset) < 16:
        raise ValueError("At least 16 patch pairs are required with drop_last=True")

    # drop_last keeps every optimization batch at the requested batch size.
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatelliteSRNet(
        in_channels=dataset.lr_patches.shape[1],
        out_channels=dataset.hr_patches.shape[1],
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = math.inf
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loaded {len(dataset)} patch pairs; training for {epochs} epoch(s).")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        for lr_patches, hr_patches in loader:
            lr_patches = lr_patches.to(device, non_blocking=True)
            hr_patches = hr_patches.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(lr_patches)
            loss = criterion(prediction, hr_patches)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_psnr += calculate_psnr(prediction.detach(), hr_patches)

        batch_count = len(loader)
        epoch_loss = running_loss / batch_count
        epoch_psnr = running_psnr / batch_count
        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
            f"Loss: {epoch_loss:.6f} | PSNR: {epoch_psnr:.2f} dB"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint)
            print(f"  Saved improved weights to {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SatelliteSRNet on an LR/HR .npz patch archive."
    )
    parser.add_argument("--data", default="dataset_patches.npz", help="Input .npz archive")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument(
        "--output", default="best_model.pth", help="Checkpoint path for best weights"
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    train(args.data, args.epochs, args.output)


if __name__ == "__main__":
    main()

"""Prepare paired raster patches for a super-resolution mapping pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


def _normalize_patch(patch: np.ma.MaskedArray) -> np.ndarray:
    """Normalize each band to [0, 1], replacing nodata and non-finite values."""
    values = np.asarray(patch.filled(np.nan), dtype=np.float32)
    invalid = np.ma.getmaskarray(patch) | ~np.isfinite(values)
    normalized = np.zeros_like(values, dtype=np.float32)

    for band_index in range(values.shape[0]):
        valid_values = values[band_index][~invalid[band_index]]
        if valid_values.size == 0:
            continue

        minimum = np.min(valid_values)
        maximum = np.max(valid_values)
        value_range = maximum - minimum
        if not np.isfinite(value_range) or value_range == 0:
            normalized[band_index][~invalid[band_index]] = 0.0
            continue

        band_values = (values[band_index] - minimum) / value_range
        normalized[band_index] = np.clip(band_values, 0.0, 1.0)
        normalized[band_index][invalid[band_index]] = 0.0

    return normalized


def _validate_inputs(
    lr_dataset: rasterio.DatasetReader,
    hr_dataset: rasterio.DatasetReader,
    patch_size: int,
    scale_factor: int,
) -> None:
    if patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be a positive integer")
    if lr_dataset.count != hr_dataset.count:
        raise ValueError(
            f"LR and HR must have the same band count; got "
            f"{lr_dataset.count} and {hr_dataset.count}"
        )
    if lr_dataset.width * scale_factor > hr_dataset.width or (
        lr_dataset.height * scale_factor > hr_dataset.height
    ):
        raise ValueError("The HR raster is too small for the requested scale factor")


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for command-line arguments."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than zero, got {parsed}")
    return parsed


def extract_patches(
    lr_path: str | Path,
    hr_path: str | Path,
    patch_size: int = 64,
    scale_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract aligned, normalized LR/HR patch tensors.

    Patches are sampled on the LR raster grid. An LR window at (row, column)
    maps to the HR window at (row * scale_factor, column * scale_factor).
    Partial windows at the right and bottom edges are skipped.
    """
    with rasterio.open(lr_path) as lr_dataset, rasterio.open(hr_path) as hr_dataset:
        _validate_inputs(lr_dataset, hr_dataset, patch_size, scale_factor)

        hr_patch_size = patch_size * scale_factor
        band_count = lr_dataset.count
        lr_patches = []
        hr_patches = []

        row_offsets = range(0, lr_dataset.height - patch_size + 1, patch_size)
        column_offsets = range(0, lr_dataset.width - patch_size + 1, patch_size)
        for row_offset in row_offsets:
            for column_offset in column_offsets:
                lr_window = Window(column_offset, row_offset, patch_size, patch_size)
                hr_window = Window(
                    column_offset * scale_factor,
                    row_offset * scale_factor,
                    hr_patch_size,
                    hr_patch_size,
                )

                lr_patch = lr_dataset.read(window=lr_window, masked=True)
                hr_patch = hr_dataset.read(window=hr_window, masked=True)
                if lr_patch.shape[1:] != (patch_size, patch_size):
                    continue
                if hr_patch.shape[1:] != (hr_patch_size, hr_patch_size):
                    continue

                lr_patches.append(_normalize_patch(lr_patch))
                hr_patches.append(_normalize_patch(hr_patch))

    lr_tensor = np.asarray(lr_patches, dtype=np.float32)
    hr_tensor = np.asarray(hr_patches, dtype=np.float32)
    if not lr_patches:
        lr_tensor = np.empty((0, band_count, patch_size, patch_size), dtype=np.float32)
        hr_tensor = np.empty(
            (0, band_count, hr_patch_size, hr_patch_size), dtype=np.float32
        )

    print(f"Successfully extracted {len(lr_tensor)} matching patch pair(s).")
    print(f"LR tensor shape: {lr_tensor.shape}")
    print(f"HR tensor shape: {hr_tensor.shape}")
    return lr_tensor, hr_tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract normalized LR/HR raster patches for super-resolution mapping."
    )
    parser.add_argument("--lr", required=True, help="Path to the low-resolution GeoTIFF")
    parser.add_argument("--hr", required=True, help="Path to the high-resolution GeoTIFF")
    parser.add_argument(
        "--patch_size",
        type=_positive_int,
        default=64,
        help="LR patch width and height (default: 64)",
    )
    parser.add_argument(
        "--scale",
        type=_positive_int,
        default=2,
        help="HR-to-LR scale factor (default: 2)",
    )
    parser.add_argument("--output", required=True, help="Output compressed .npz archive path")
    args = parser.parse_args()

    lr_patches, hr_patches = extract_patches(
        args.lr,
        args.hr,
        patch_size=args.patch_size,
        scale_factor=args.scale,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, lr_patches=lr_patches, hr_patches=hr_patches)
    print(f"Saved compressed patch archive to: {output_path}")


if __name__ == "__main__":
    main()

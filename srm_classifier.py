"""Estimate sub-pixel land-cover fractions from super-resolved imagery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch


LAND_COVER_CLASSES = ("Water", "Urban", "Vegetation", "Barren")


def predict_land_cover_fractions(
    sr_tensor: torch.Tensor, num_classes: int = 4
) -> torch.Tensor:
    """Return per-pixel class fractions for a ``(C,H,W)`` or ``(B,C,H,W)`` tensor.

    The fixed spectral logits provide a useful zero-shot baseline for normalized
    multispectral data (RGB + NIR when four bands are supplied). Softmax outputs
    are continuous abundances: each pixel's class fractions sum to one rather
    than being reduced to a hard class label.
    """
    if not isinstance(sr_tensor, torch.Tensor):
        raise TypeError("sr_tensor must be a torch.Tensor")
    if sr_tensor.ndim not in (3, 4):
        raise ValueError("sr_tensor must have shape (C,H,W) or (B,C,H,W)")
    if sr_tensor.shape[-3] <= 0 or num_classes <= 0:
        raise ValueError("The input must have channels and num_classes must be positive")
    if num_classes != 4:
        raise ValueError("The baseline classifier defines exactly four land-cover classes")

    batched = sr_tensor.ndim == 4
    image = sr_tensor.unsqueeze(0) if not batched else sr_tensor
    image = image.to(dtype=torch.float32)
    channels = image.shape[1]

    # Use RGB/NIR semantics when available; otherwise use stable channel groups.
    red = image[:, 0]
    green = image[:, min(1, channels - 1)]
    blue = image[:, min(2, channels - 1)]
    nir = image[:, min(3, channels - 1)]
    brightness = (red + green + blue) / 3.0
    logits = torch.stack(
        (
            blue - 0.5 * (green + nir),  # Water
            brightness - 0.5 * nir,  # Urban
            nir - red,  # Vegetation
            brightness - 0.5 * nir,  # Barren
        ),
        dim=1,
    )

    fractions = torch.softmax(logits, dim=1)
    fractions = fractions / fractions.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(fractions.dtype).eps
    )
    return fractions if batched else fractions.squeeze(0)


def save_classified_fractions(
    fraction_array: np.ndarray | torch.Tensor,
    reference_raster_path: str | Path,
    output_path: str | Path = "srm_classified_output.tif",
) -> None:
    """Save ``(classes,H,W)`` fractions using georeferencing from a reference raster."""
    if isinstance(fraction_array, torch.Tensor):
        fraction_array = fraction_array.detach().cpu().numpy()
    fractions = np.asarray(fraction_array, dtype=np.float32)
    if fractions.ndim != 3:
        raise ValueError("fraction_array must have shape (classes, height, width)")
    if not np.isfinite(fractions).all():
        raise ValueError("fraction_array contains NaN or infinite values")
    if np.any(fractions < 0):
        raise ValueError("fraction_array cannot contain negative fractions")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_raster_path) as reference:
        profile = reference.profile.copy()
        profile.update(
            count=fractions.shape[0],
            dtype="float32",
            height=fractions.shape[1],
            width=fractions.shape[2],
            compress="deflate",
            nodata=None,
        )
        with rasterio.open(output, "w", **profile) as destination:
            destination.write(fractions)
            destination.set_band_description(
                1, LAND_COVER_CLASSES[0] if fractions.shape[0] == 4 else "Class 1"
            )
            if fractions.shape[0] == 4:
                for band, name in enumerate(LAND_COVER_CLASSES[1:], start=2):
                    destination.set_band_description(band, name)
    print(f"Saved {fractions.shape[0]} fraction bands to {output}")


if __name__ == "__main__":
    sample_input = torch.rand(4, 128, 128)
    sample_fractions = predict_land_cover_fractions(sample_input)
    reference_path = Path("sample_hr.tif")
    if not reference_path.exists():
        raise FileNotFoundError(
            "Standalone example requires sample_hr.tif; run dummydata.py first."
        )
    save_classified_fractions(sample_fractions, reference_path)
    print(f"Fraction tensor shape: {tuple(sample_fractions.shape)}")
    print(
        "Per-pixel fraction sum range: "
        f"{sample_fractions.sum(dim=0).min().item():.4f} - "
        f"{sample_fractions.sum(dim=0).max().item():.4f}"
    )

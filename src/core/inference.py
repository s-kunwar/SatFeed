"""FastAPI service for satellite super-resolution mapping inference."""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from rasterio.enums import Resampling
from rasterio.transform import Affine
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from shapely.geometry import Polygon

from model import SatelliteSRNet


class _LegacyScale4Net(nn.Module):
    """Compatibility wrapper for earlier scale-4 checkpoints."""

    def __init__(self, in_channels: int, out_channels: int, features: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU())
        self.body = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, 3, padding=1),
        )
        self.output = nn.Sequential(
            nn.Conv2d(features, out_channels * 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        return self.output(features + self.body(features))


CHECKPOINT_PATH = Path(os.getenv("SRM_MODEL_PATH", "best_model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_READY = False
MODEL_OUTPUT_CHANNELS = 4
MODEL = SatelliteSRNet(in_channels=4, out_channels=4).to(DEVICE)
if CHECKPOINT_PATH.is_file():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint must contain a state dict: {CHECKPOINT_PATH}")
    if isinstance(checkpoint.get("model_state_dict"), dict):
        checkpoint = checkpoint["model_state_dict"]

    head_weight = checkpoint.get("head.weight", checkpoint.get("head.0.weight"))
    output_weight = checkpoint.get("output.0.weight")
    if not isinstance(head_weight, torch.Tensor) or not isinstance(
        output_weight, torch.Tensor
    ):
        raise RuntimeError(
            "Checkpoint is missing head.weight or output.0.weight; "
            f"cannot infer channel dimensions from {CHECKPOINT_PATH}"
        )
    checkpoint_in_channels = head_weight.shape[1]
    checkpoint_out_channels = output_weight.shape[0]
    if checkpoint_in_channels <= 0 or checkpoint_out_channels <= 0:
        raise RuntimeError("Checkpoint contains invalid channel dimensions")

    # Size mismatches still fail with strict=False, so build the model to match
    # the checkpoint before loading its complete state dict.
    if "head.weight" in checkpoint:
        MODEL = SatelliteSRNet(
            in_channels=checkpoint_in_channels,
            out_channels=checkpoint_out_channels,
        ).to(DEVICE)
    elif "head.0.weight" in checkpoint and output_weight.shape[0] % 16 == 0:
        MODEL = _LegacyScale4Net(
            in_channels=checkpoint_in_channels,
            out_channels=checkpoint_out_channels // 16,
        ).to(DEVICE)
    else:
        raise RuntimeError("Unsupported checkpoint architecture")
    MODEL.load_state_dict(checkpoint, strict=True)
    MODEL_OUTPUT_CHANNELS = (
        checkpoint_out_channels // 16
        if "head.0.weight" in checkpoint and "head.weight" not in checkpoint
        else checkpoint_out_channels
    )
    MODEL_READY = True
MODEL.eval()
MODEL_INPUT_CHANNELS = (
    MODEL.head[0].in_channels
    if isinstance(MODEL.head, nn.Sequential)
    else MODEL.head.in_channels
)

app = FastAPI(title="Satellite Super-Resolution Mapping API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _write_upload(upload: UploadFile, destination: BinaryIO) -> None:
    while True:
        chunk = upload.file.read(1024 * 1024)
        if not chunk:
            break
        destination.write(chunk)


def _clear_torch_memory() -> None:
    """Release temporary inference allocations without assuming CUDA is present."""
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def _infer_in_tiles(
    input_tensor: torch.Tensor,
    scale_factor: int = 4,
    tile_size: int = 512,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference in bounded tiles and return baseline/model arrays in HWC order."""
    height, width = input_tensor.shape[-2:]
    output_height, output_width = height * scale_factor, width * scale_factor
    output_channels = MODEL_OUTPUT_CHANNELS
    baseline_array = np.empty((output_height, output_width, input_tensor.shape[1]), dtype=np.float32)
    model_array = np.empty((output_height, output_width, output_channels), dtype=np.float32)

    total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
    completed_tiles = 0
    with torch.no_grad():
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                tile = input_tensor[:, :, top:bottom, left:right]
                tile_height, tile_width = bottom - top, right - left
                padded_height = ((tile_height + tile_size - 1) // tile_size) * tile_size
                padded_width = ((tile_width + tile_size - 1) // tile_size) * tile_size
                tile = F.pad(tile, (0, padded_width - tile_width, 0, padded_height - tile_height))
                model_output = MODEL(tile)
                tile_output_size = (tile_height * scale_factor, tile_width * scale_factor)
                baseline_tile = F.interpolate(
                    tile,
                    size=(padded_height * scale_factor, padded_width * scale_factor),
                    mode="bicubic",
                    align_corners=False,
                )
                super_resolved_tile = F.interpolate(
                    model_output,
                    size=(padded_height * scale_factor, padded_width * scale_factor),
                    mode="bilinear",
                    align_corners=False,
                )
                baseline_tile = baseline_tile[..., : tile_output_size[0], : tile_output_size[1]]
                super_resolved_tile = super_resolved_tile[
                    ..., : tile_output_size[0], : tile_output_size[1]
                ]
                baseline_tile_array = np.moveaxis(baseline_tile.squeeze(0).cpu().numpy(), 0, -1)
                model_tile_array = np.moveaxis(super_resolved_tile.squeeze(0).cpu().numpy(), 0, -1)
                y, x = top * scale_factor, left * scale_factor
                out_h, out_w = tile_output_size
                try:
                    canvas_h, canvas_w = baseline_array.shape[:2]
                    clipped_h = min(out_h, canvas_h - y)
                    clipped_w = min(out_w, canvas_w - x)
                    if clipped_h <= 0 or clipped_w <= 0:
                        raise ValueError(f"Tile origin ({y}, {x}) is outside output canvas")
                    baseline_array[y : y + clipped_h, x : x + clipped_w] = baseline_tile_array[
                        :clipped_h, :clipped_w
                    ]
                    model_array[y : y + clipped_h, x : x + clipped_w] = model_tile_array[
                        :clipped_h, :clipped_w
                    ]
                except (IndexError, ValueError) as exc:
                    raise RuntimeError(f"Unable to stitch tile at ({top}, {left}): {exc}") from exc
                del tile, model_output, baseline_tile, super_resolved_tile
                del baseline_tile_array, model_tile_array
                completed_tiles += 1
                if progress_callback is not None:
                    progress_callback(completed_tiles, total_tiles)
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
    return baseline_array, model_array


def _percentile_normalize_raster(
    raster: np.ma.MaskedArray | np.ndarray,
) -> np.ndarray:
    """Normalize each raster band to [0, 1] using its 2nd and 98th percentiles."""
    if isinstance(raster, np.ma.MaskedArray):
        raster_float = raster.astype(np.float32)
        values = raster_float.filled(np.nan)
        invalid = np.ma.getmaskarray(raster)
    else:
        values = np.asarray(raster, dtype=np.float32)
        invalid = np.zeros_like(values, dtype=bool)
    invalid |= ~np.isfinite(values)
    np.nan_to_num(values, copy=False, nan=0.0)
    normalized = np.zeros_like(values, dtype=np.float32)
    for band_index in range(values.shape[0]):
        valid = values[band_index][~invalid[band_index]]
        if valid.size == 0:
            continue
        low, high = np.percentile(valid, (2.0, 98.0))
        if high <= low:
            normalized[band_index][~invalid[band_index]] = 0.0
        else:
            normalized[band_index] = np.clip(
                (values[band_index] - low) / (high - low), 0.0, 1.0
            )
            normalized[band_index][invalid[band_index]] = 0.0
    return normalized


def _postprocess_super_resolved(
    super_resolved: torch.Tensor,
) -> np.ndarray:
    """Convert normalized ``(C,H,W)`` output to an HWC uint8 RGB-order array."""
    # Training uses normalized raster values, not ImageNet normalization.
    sr_array = super_resolved.detach().cpu().numpy()
    sr_array = np.transpose(sr_array, (1, 2, 0))
    return np.clip(sr_array * 255.0, 0, 255).astype(np.uint8)


def _match_color_statistics(
    output_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> np.ndarray:
    """Match output channel mean/std to the upsampled input reference."""
    if output_hwc.shape != reference_hwc.shape:
        raise ValueError("Output and reference images must have the same shape")
    output = np.asarray(output_hwc, dtype=np.float32)
    reference = np.asarray(reference_hwc, dtype=np.float32)
    output_mean = output.mean(axis=(0, 1), keepdims=True)
    output_std = output.std(axis=(0, 1), keepdims=True)
    reference_mean = reference.mean(axis=(0, 1), keepdims=True)
    reference_std = reference.std(axis=(0, 1), keepdims=True)
    matched = (
        (output - output_mean)
        / (output_std + 1e-5)
        * reference_std
        + reference_mean
    )
    return np.clip(matched, 0.0, 1.0)


def _extract_vector_mapping(
    sr_uint8: np.ndarray,
    transform: Affine,
    crs: object,
    output_path: str,
    minimum_area_pixels: float = 9.0,
) -> int:
    """Extract bright structures and save their georeferenced polygons as GeoJSON."""
    if sr_uint8.ndim != 3 or sr_uint8.shape[2] == 0:
        raise ValueError("sr_uint8 must have shape (height, width, channels)")

    display_bands = sr_uint8[:, :, :3]
    if display_bands.shape[2] == 1:
        grayscale = display_bands[:, :, 0]
    elif display_bands.shape[2] == 2:
        grayscale = cv2.cvtColor(
            np.concatenate((display_bands, display_bands[:, :, :1]), axis=2),
            cv2.COLOR_RGB2GRAY,
        )
    else:
        grayscale = cv2.cvtColor(display_bands, cv2.COLOR_RGB2GRAY)
    thresholded = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    edges = cv2.Canny(grayscale, threshold1=50, threshold2=150)
    mask = cv2.bitwise_or(thresholded, edges)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    geometries = []
    areas = []
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area_pixels:
            continue
        pixel_coordinates = contour.reshape(-1, 2)
        map_coordinates = [transform * (float(x), float(y)) for x, y in pixel_coordinates]
        if len(map_coordinates) < 3:
            continue
        polygon = Polygon(map_coordinates)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.geom_type in {"Polygon", "MultiPolygon"}:
            geometries.append(polygon)
            areas.append(float(cv2.contourArea(contour)))

    vectors = gpd.GeoDataFrame(
        {"feature_type": ["detected_boundary"] * len(geometries), "pixel_area": areas},
        geometry=geometries,
        crs=crs,
    )
    vectors.to_file(output_path, driver="GeoJSON")
    return len(geometries)


@app.get("/")
def root() -> dict[str, str]:
    """Report service status."""
    return {"status": "SRM API is active", "device": str(DEVICE)}


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
) -> dict[str, object]:
    """Super-resolve a GeoTIFF, preserve georeferencing, and report quality metrics."""
    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found: {CHECKPOINT_PATH}",
        )
    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Upload must be a GeoTIFF file")

    input_path = ""
    output_path = ""
    geojson_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as uploaded:
            input_path = uploaded.name
            _write_upload(file, uploaded)

        with rasterio.open(input_path) as source:
            if source.count < 3:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Input raster must contain at least 3 bands for RGB inference; "
                        f"received {source.count}"
                    ),
                )
            # Bound inference memory for large rasters while retaining the source
            # extent and CRS for the output georeferencing.
            inference_width = source.width
            inference_height = source.height
            if max(inference_width, inference_height) > 2048:
                downsample_factor = 2048 / max(inference_width, inference_height)
                inference_width = max(1, round(inference_width * downsample_factor))
                inference_height = max(1, round(inference_height * downsample_factor))
                rgb_data = source.read(
                    [1, 2, 3],
                    out_shape=(3, inference_height, inference_width),
                    resampling=Resampling.bilinear,
                    masked=True,
                )
            else:
                rgb_data = source.read([1, 2, 3], masked=True)
            source_data = _percentile_normalize_raster(rgb_data)
            source_profile = source.profile.copy()
            source_transform = source.transform
            source_bounds = source.bounds
            input_tensor = torch.from_numpy(source_data).unsqueeze(0).to(DEVICE)

        _clear_torch_memory()
        try:
            baseline_array, model_array = _infer_in_tiles(input_tensor)
            baseline_array = np.clip(baseline_array, 0.0, 1.0)
            model_array = np.clip(model_array, 0.0, 1.0)
        finally:
            del input_tensor
            _clear_torch_memory()

        matched_array = _match_color_statistics(model_array, baseline_array)
        baseline_uint8 = np.clip(baseline_array * 255.0, 0, 255).astype(np.uint8)
        model_uint8 = np.clip(matched_array * 255.0, 0, 255).astype(np.uint8)
        reference_hwc = baseline_uint8
        prediction_hwc = model_uint8
        psnr = float(
            peak_signal_noise_ratio(reference_hwc, prediction_hwc, data_range=255)
        )
        ssim = float(
            structural_similarity(
                reference_hwc,
                prediction_hwc,
                channel_axis=-1,
                data_range=255,
            )
        )
        super_resolved_uint8 = model_uint8

        with tempfile.NamedTemporaryFile(suffix="_srm_super_resolved.tif", delete=False) as classified:
            output_path = classified.name
        output_profile = source_profile.copy()
        output_profile.update(
            driver="GTiff",
            width=super_resolved_uint8.shape[1],
            height=super_resolved_uint8.shape[0],
            count=super_resolved_uint8.shape[2],
            dtype="uint8",
            transform=source_transform
            * Affine.scale(
                source_profile["width"] / super_resolved_uint8.shape[1],
                source_profile["height"] / super_resolved_uint8.shape[0],
            ),
            compress="deflate",
            nodata=None,
        )
        with rasterio.open(output_path, "w", **output_profile) as destination:
            destination.write(np.transpose(super_resolved_uint8, (2, 0, 1)))
        with tempfile.NamedTemporaryFile(
            suffix="_srm_vector_mapping.geojson", delete=False
        ) as vector_output:
            geojson_path = vector_output.name
        feature_count = _extract_vector_mapping(
            super_resolved_uint8,
            output_profile["transform"],
            source_profile.get("crs"),
            geojson_path,
        )
    except rasterio.errors.RasterioIOError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid GeoTIFF: {exc}") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"SRM inference failed: {exc}") from exc
    finally:
        file.file.close()

    Path(input_path).unlink(missing_ok=True)
    return {
        "output_path": output_path,
        "output_format": "GeoTIFF",
        "geojson_path": geojson_path,
        "geojson_feature_count": feature_count,
        "scale_factor": 4,
        "width": super_resolved_uint8.shape[1],
        "height": super_resolved_uint8.shape[0],
        "bounds": {
            "left": source_bounds.left,
            "bottom": source_bounds.bottom,
            "right": source_bounds.right,
            "top": source_bounds.top,
        },
        "crs": source_profile.get("crs").to_string()
        if source_profile.get("crs") is not None
        else None,
        "psnr_db": psnr,
        "ssim": ssim,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# SatFeed — High-Resolution Geospatial Super-Resolution & GIS Intelligence Engine

SatFeed is a high-performance Python application for upscaling medium-resolution multispectral GeoTIFF imagery to sub-meter spatial resolution. It preserves source spatial transforms and coordinate reference systems while producing GIS-ready vector outputs for detected buildings, roads, and image boundaries.

## Key Capabilities

- **Sub-Pixel Spatial Super-Resolution**: 4x spatial resolution enhancement for multispectral satellite rasters using a PyTorch sub-pixel convolution model.
- **Tile-Based Memory Optimization**: Grid-based 512 × 512 window processing for large (including >100 MB) GeoTIFF rasters without loading the complete inference graph into GPU memory at once.
- **Automated Vector Extraction**: Adaptive thresholding, edge detection, contour conversion, and GeoJSON export for building footprints and road or boundary features.
- **Coordinate System Retention**: Preservation of the source EPSG projection, affine transformation, raster dimensions, and spatial bounds in generated GeoTIFF and GeoJSON products.
- **Radiometric Normalization**: Per-band 2nd–98th percentile normalization with masked-data, NaN, and nodata handling.
- **Operational Quality Reporting**: Bicubic baseline comparison using PSNR and SSIM, with downloadable georeferenced outputs from the Streamlit dashboard.

## Project Architecture

```text
GeoTIFF Upload
      |
      v
RGB Band Selection + Percentile Normalization
      |
      v
Tile Windowing (512 x 512)
      |
      v
PyTorch Inference (4x Super-Resolution)
      |
      v
Color Alignment + Bicubic Baseline Metrics
      |
      +---------------------> Georeferenced GeoTIFF
      |
      v
GeoJSON Extraction (buildings / roads / boundaries)
      |
      +---------------------> GIS Vector Output
      |
      v
Streamlit UI (previews, metrics, downloads, map overlay)
```

### Repository Layout

```text
.
├── app.py                         # FastAPI compatibility entry point
├── ui.py                          # Streamlit compatibility entry point
├── model.py                       # SuperResolutionModel definition
├── train.py                       # Patch dataset and training workflow
├── data_preprocessing.py          # LR/HR GeoTIFF patch generation
└── src/
    ├── core/
    │   ├── inference.py           # Model loading, tiled inference, API route
    │   └── vectorizer.py          # Raster-to-GeoJSON feature extraction
    ├── utils/
    │   ├── raster.py              # Raster normalization helpers
    │   └── metrics.py             # Color-statistics alignment
    └── ui/
        ├── main.py                # SatFeed Streamlit application
        ├── components.py          # Reusable UI helpers
        └── styles.py              # Shared UI styling primitives
```

The root launchers are intentionally retained so existing commands and integrations continue to work. Internal implementation code lives under `src/` and uses package-qualified imports.

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/s-kunwar/CPP.git
cd CPP
```

If the project is already available locally, change into its root directory instead.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install runtime dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy rasterio scikit-image streamlit fastapi "uvicorn[standard]" python-multipart requests pydeck opencv-python shapely geopandas
```

For CUDA-enabled PyTorch, install the wheel matching the target NVIDIA driver and CUDA runtime from the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the remaining packages.

### 4. Prepare a model checkpoint

Place the trained checkpoint at `best_model.pth`, or set an explicit path:

Windows PowerShell:

```powershell
$env:SRM_MODEL_PATH = "D:\models\best_model.pth"
```

Linux or macOS:

```bash
export SRM_MODEL_PATH=/models/best_model.pth
```

The inference service accepts GeoTIFFs containing at least three bands and uses bands 1–3 as RGB. Four-band RGB + NIR training data is supported by the model and training pipeline.

### 5. Launch the application

Start the FastAPI inference service in one terminal:

```bash
python app.py
```

Start the Streamlit dashboard in a second terminal:

```bash
streamlit run ui.py
```

Open the URL printed by Streamlit, upload a `.tif` or `.tiff`, and select **Run SatFeed Super-Resolution**. The dashboard provides:

- side-by-side original and super-resolved previews;
- PSNR and SSIM metrics;
- a georeferenced GeoTIFF download;
- a GeoJSON vector download; and
- an interactive map overlay for extracted features.

## Free online deployment

The current architecture uses two services: a Streamlit frontend and a FastAPI
inference backend. A practical free setup is:

1. Deploy the repository's `app.py` as a Render **Free Web Service**.
   Use build command `pip install -r requirements.txt` and start command
   `python app.py`. Render provides the `PORT` environment variable, which the
   launcher uses automatically. Copy the resulting HTTPS URL.
2. Deploy the repository on [Streamlit Community Cloud](https://share.streamlit.io/)
   with main file `ui.py`.
3. In the Streamlit app settings, add this secret:
   `SRM_API_URL = "https://YOUR-RENDER-SERVICE.onrender.com/predict"`.
4. Confirm the Render service's `/` URL returns its JSON status response, then
   upload a GeoTIFF in Streamlit.

Free services sleep when idle, so the first request can take a minute or more.
They also use ephemeral disk storage; generated GeoTIFF and GeoJSON files are
temporary download artifacts, not permanent storage. Large GeoTIFF inference
may exceed free CPU/RAM/time limits.

### Optional: Generate training patches

```bash
python data_preprocessing.py --lr sample_lr.tif --hr sample_hr.tif --output dataset_patches.npz
python train.py --data dataset_patches.npz --epochs 25 --output best_model.pth
```

Training expects paired LR and HR GeoTIFFs. The generated archive contains normalized `lr_patches` and `hr_patches` tensors in `(N, C, H, W)` layout.

## Quality Metrics & Performance Benchmarks

SatFeed reports metrics after converting both the bicubic baseline and model output to the same clipped 8-bit range `[0, 255]`. This avoids comparing incompatible radiometric scales.

- **PSNR (dB)** measures pixel-level reconstruction error. Higher values indicate lower mean squared error. As a practical interpretation, values below 20 dB usually indicate substantial error, 20–30 dB indicates usable reconstruction, and values above 30 dB indicate stronger pixel fidelity. The useful range depends heavily on sensor noise, scene content, and reference quality.
- **SSIM** measures structural similarity on a normalized 0–1 scale. Values closer to 1.0 indicate stronger structural agreement. Values below roughly 0.70 commonly indicate visible structural degradation; 0.70–0.90 is scene-dependent; values above 0.90 generally indicate strong structural preservation. SSIM is not a substitute for geospatial accuracy assessment.

The following table defines the comparison reported by the service. Values are measured per input and should not be treated as fixed model guarantees.

| Evaluation dimension | Bicubic baseline | SatFeed output |
|---|---|---|
| Spatial scaling | 4x interpolation | 4x learned super-resolution |
| Input data | Normalized RGB raster | Same normalized RGB raster |
| Radiometric range for metrics | Clipped `uint8`, `[0, 255]` | Color-aligned, clipped `uint8`, `[0, 255]` |
| PSNR | Baseline `psnr_db` | SatFeed `psnr_db` |
| SSIM | Baseline structural score | SatFeed structural score |
| Geospatial metadata | Can be retained by a separate raster workflow | CRS, bounds, and scaled affine transform retained |
| GIS vectors | Not generated | GeoJSON contours extracted from the output |
| Large-raster memory behavior | Depends on caller implementation | 512 × 512 tiled inference with cache cleanup |

For reproducible benchmarking, evaluate both outputs against the same independent HR reference, record the sensor and scene characteristics, and report image dimensions, band selection, checkpoint version, device, PSNR, SSIM, and processing time.

## Output Products

The API returns JSON metadata containing output paths, output dimensions, CRS, bounds, scale factor, PSNR, SSIM, and GeoJSON feature count. Generated products are:

- `*_srm_super_resolved.tif`: compressed, uint8, georeferenced high-resolution raster.
- `*_srm_vector_mapping.geojson`: vector features in the source CRS.

## Operational Notes

- Inputs larger than 2048 pixels on their longest side are downsampled with bilinear raster resampling before tiled inference to cap memory usage.
- Partial edge tiles are padded for model execution, cropped back to their exact spatial extent, and bounds-checked during stitching.
- Inference runs under `torch.no_grad()` and releases temporary CPU and CUDA allocations between tiles.
- The API reads only the first three bands for RGB inference when a raster has more than three bands; source CRS and transform metadata remain available for output generation.
- GeoJSON contour extraction is image-based and should be reviewed before use in authoritative mapping or cadastral workflows.

## License

Distributed under the MIT License. See LICENSE for more information.

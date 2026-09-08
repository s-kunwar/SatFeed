# SatFeed Technical Deep Dive

## 1. Project scope and objective

SatFeed is an end-to-end satellite super-resolution mapping pipeline. Its primary objective is to upsample medium-resolution multispectral raster data into higher-resolution imagery while preserving georeferencing and making the result useful for downstream Spatial Intelligence / GIS tasks.

The implementation blends:
- deep learning for image enhancement
- geospatial raster logic for preserving CRS and transform
- computer vision for boundary extraction
- GIS vector export for map-ready outputs

The project is implemented as a combination of:
- training scripts for dataset construction and model training
- a FastAPI inference backend
- a Streamlit dashboard
- supporting utility functions and geospatial preprocessing

## 2. Repository structure and responsibilities

### Root scripts

- [app.py](../app.py)  
  Starts the FastAPI service.

- [ui.py](../ui.py)  
  Runs the Streamlit dashboard through module execution.

- [model.py](../model.py)  
  Defines the SatelliteSRNet architecture.

- [train.py](../train.py)  
  Trains the model from a .npz patch archive.

- [data_preprocessing.py](../data_preprocessing.py)  
  Extracts and normalizes LR/HR patch pairs from GeoTIFFs.

### Source package

- [src/core/inference.py](../src/core/inference.py)  
  Contains the API logic, tiled inference engine, metrics, and GeoJSON export.

- [src/ui/dashboard.py](../src/ui/dashboard.py)  
  Implements the interactive interface and result display.

## 3. Data model and input assumptions

The system expects input rasters with at least three bands, typically representing RGB. The code reads bands 1, 2, and 3 as the visible-image input. The README also notes support for 4-band RGB+NIR data in the training pipeline.

Important input assumptions:
- GeoTIFF input format
- rasterio-compatible dataset
- at least 3 bands
- valid georeferencing
- multispectral support for 3-4 band imagery

The inference service also checks that the uploaded raster is valid and rejects non-GeoTIFF or invalid-band inputs.

## 4. Preprocessing and normalization

### Patch extraction

The preprocessing script uses rasterio to read windowed patches from LR and HR rasters.

Key behavior:
- patch windows are sampled on the LR grid
- corresponding HR windows are extracted using a scale factor mapping
- partial edge windows are skipped
- masks are respected so nodata/invalid pixels are handled safely

### Normalization logic

The project normalizes data per band using a robust dynamic range strategy:
- masked arrays are converted to finite numeric values
- invalid pixels are replaced with NaN and then zeroed
- each band is normalized to [0,1] using the band’s own selected range

The API version uses per-band percentile normalization (2nd and 98th percentile) before inference.

This helps:
- suppress outlier influence
- reduce dynamic-range mismatch between scenes
- stabilize model training and inference

## 5. Model architecture

### SatelliteSRNet

The model is implemented in [model.py](../model.py). It is a residual CNN with a sub-pixel reconstruction step.

Architecture outline:
- Input: tensor shaped as (N, C, H, W)
- Head: Conv2d from input channels to feature channels
- Residual trunk: multiple residual blocks
- Upsampling head: Conv2d followed by PixelShuffle
- Output: Conv2d + Sigmoid

Residual block structure:
- 3x3 convolution
- ReLU
- 3x3 convolution
- identity skip connection

This design is intended to:
- preserve spatial structure
- propagate high-frequency information
- stabilize deeper feature extraction
- reconstruct high-resolution output efficiently

### PixelShuffle and super-resolution logic

The model does not simply resize an image; it learns feature maps and rearranges them into higher-resolution pixel grids using nn.PixelShuffle(). This is a classic super-resolution strategy that increases spatial resolution by reordering depth channels into spatial dimensions.

## 6. Training pipeline

### Training data format

The patch extraction process saves arrays in compressed NPZ format with keys:
- lr_patches
- hr_patches

The arrays are expected to be shaped as:
- (N, C, H, W)

### Training loop

In [train.py](../train.py), the model is trained with:
- torch.utils.data.Dataset
- DataLoader with batch_size=16
- shuffle enabled
- drop_last=True
- Adam optimizer
- MSELoss objective

The code checks for enough examples before training and saves the best model state when the epoch loss improves.

### Evaluation metrics during training

Each epoch computes:
- epoch loss
- PSNR over the batch predictions

This gives a direct signal for how well the network is reconstructing the target high-resolution patches.

## 7. Inference service architecture

The service in [src/core/inference.py](../src/core/inference.py) handles actual inference requests.

### Model loading

At import time, the service checks whether a model checkpoint exists at:
- SRM_MODEL_PATH environment variable
- or best_model.pth in the project root

If present, the checkpoint is loaded into a model instance. The code supports:
- standard SatelliteSRNet checkpoints
- legacy architecture checkpoints with a compatibility wrapper

This is an important robustness feature because earlier generated weights may not match the current architecture exactly.

### Device selection

The service chooses:
- CUDA if available
- otherwise CPU

This keeps inference portable across local machines, workstations, and cloud environments.

## 8. Memory-safe tiled inference

Large raster processing is a major challenge in geospatial ML.

The service implements tiled inference using a tile size of 512 and a scale factor of 4.

Process:
- determine image height and width
- compute tile grid covering the whole image
- process each tile independently
- pad each tile to tile boundaries
- run the model on padded tile
- resize and crop the output back to the tile’s original footprint
- stitch tile outputs to the global output canvas

This avoids loading the full large image into memory at once.

### Why tile processing matters

Without tiling:
- large 100MB+ rasters may blow memory limits
- GPU allocations become unstable
- direct full-image inference can fail on constrained hardware

The coded approach keeps the model inference bounded and scalable.

## 9. Baseline comparison and metrics

The system compares the model output against a bicubic upsample of the input data.

### PSNR

Peak Signal-to-Noise Ratio measures pixel-wise fidelity. Higher values indicate stronger reconstruction quality.

### SSIM

Structural Similarity Index measures structural consistency between the reference and prediction. It accounts for luminance, contrast, and structure rather than only pixel-level error.

The service computes both using scikit-image:
- peak_signal_noise_ratio
- structural_similarity

## 10. Color-statistics matching

The output from the neural model is adjusted using the reference baseline’s mean and standard deviation. This is handled by `_match_color_statistics`.

The purpose is to reduce color and brightness mismatch between:
- upsampled bicubic reference
- model-generated super-resolved output

This makes the final result visually and numerically more consistent, especially when comparing predictions to a baseline image.

## 11. Georeferencing and output preservation

This is a critical part of the project. A model output is interesting only if it remains spatially valid.

The service extracts and preserves:
- source transform
- source bounds
- source CRS
- output width and height
- output profile from rasterio

The output GeoTIFF is written with updated dimensions and transform values, but the source georeferencing properties remain aligned to the original spatial extent.

This enables downstream GIS usage without losing location information.

## 12. Vector extraction pipeline

After super-resolving the raster, the service converts image content into vector geometry using OpenCV and Shapely.

### Process flow

1. Select first three channels as display image
2. Convert to grayscale
3. Apply Gaussian adaptive threshold
4. Run Canny edge detection
5. Combine masks
6. Find contours
7. Filter low-area contours
8. Convert contour arrays to polygon geometries
9. Save as GeoJSON

### GeoJSON output schema

The output GeoDataFrame usually contains:
- feature_type: "detected_boundary"
- pixel_area: contour area
- geometry: polygon or multipolygon

This yields map-ready vector data representing detected structures or boundaries.

## 13. Streaming dashboard and UX

The dashboard in [src/ui/dashboard.py](../src/ui/dashboard.py) is built with Streamlit and provides interactive processing status.

It includes:
- file uploader for GeoTIFF upload
- optional demo loading
- progress stages
- metric cards
- original vs output preview
- map overlay using pydeck
- download buttons for TIFF and GeoJSON

The interface also supports a local inference mode for Streamlit-only deployment, using the same core pipeline logic.

## 14. API contract

The backend exposes a prediction endpoint:
- POST /predict
- accepts a GeoTIFF upload
- returns JSON metadata including:
  - output_path
  - geojson_path
  - geojson_feature_count
  - scale_factor
  - width and height
  - bounds
  - CRS
  - PSNR
  - SSIM

This allows UI and external integrations to consume the generated outputs in a structured way.

## 15. Deployment model

The project supports two main deployment styles:

### API architecture
- FastAPI backend runs the heavy computation
- Streamlit connects to it over HTTP

### Direct Streamlit deployment
- dashboard runs local inference in-process
- no separate API server required

This flexibility is useful for free hosting environments or local experimentation.

## 16. Practical interpretation of the result

The end-user value is not just “the image looks bigger.” Instead, the pipeline aims to produce:
- clearer geospatial imagery
- more usable raster outputs
- more accurate map-like structural interpretation
- GIS-ready vector boundaries

This makes it well suited for:
- remote sensing workflows
- urban feature extraction
- map enhancement tasks
- visual comparison and analysis of satellite scenes

## 17. Summary

SatFeed is a complete geospatial ML pipeline that combines:
- training data preparation
- residual CNN super-resolution
- georeferenced raster reconstruction
- quality metrics
- OpenCV-based feature extraction
- GeoJSON transformation
- Streamlit-based visualization and downloads

The core logic is intentionally practical: optimize for real-world satellite imagery, keep GIS output faithful to spatial metadata, and give end users a visual and vector representation of the enhanced data.

# SatFeed High-Level Explanation

## What this project does

SatFeed is a satellite image super-resolution and GIS intelligence system. Its goal is to take a lower-resolution multispectral GeoTIFF, improve its spatial resolution, and then convert the result into a GIS-friendly output that can be used for mapping and feature detection.

In plain English:

- The project starts with a medium-resolution satellite image.
- It improves the image quality and resolution using a deep learning model.
- It keeps the original geospatial metadata so the output remains usable in GIS software.
- It extracts vector features such as boundaries and detected structures.
- It presents the result in a dashboard with quality metrics and download links.

## The core idea

Satellite data is often medium resolution. For many geospatial tasks, that is not enough for accurate mapping. SatFeed tries to learn a mapping from low-resolution raster patches to higher-resolution patches using a neural network.

The network is trained on paired LR/HR patches that are extracted from georeferenced raster datasets. Once trained, the model can apply that learned mapping to new satellite input.

## Main technologies used

### Deep learning
- Python
- PyTorch
- Torch nn modules
- Residual blocks
- PixelShuffle upsampling

### Raster and geospatial processing
- Rasterio
- NumPy
- GDAL-style GeoTIFF handling
- Affine transforms
- CRS preservation

### Image processing and geometry
- OpenCV
- Canny edge detection
- adaptive thresholding
- contour extraction
- Shapely polygons
- GeoPandas GeoJSON output

### Frontend / app layer
- Streamlit for dashboard UI
- FastAPI for backend inference service

### Metrics and validation
- PSNR
- SSIM

## How the full process works

### 1. Prepare training data

The system extracts paired low-resolution and high-resolution patch sets from matching GeoTIFFs. Each patch is normalized per band to a consistent numeric range, often [0,1]. The patch archive is saved as a compressed .npz file.

This step is handled by the dataset generation pipeline in [data_preprocessing.py](../data_preprocessing.py) and the training pipeline in [train.py](../train.py).

### 2. Train the model

A model called SatelliteSRNet is defined in [model.py](../model.py). It uses residual blocks to preserve image details and a PixelShuffle-based upsampling step to increase resolution.

The training loop loads LR/HR patches, runs them through the model, computes the loss, updates the weights, and saves the checkpoint when the model improves.

### 3. Inference on new data

When a user uploads a GeoTIFF, the system:
- reads the raster
- ensures it has enough bands for RGB processing
- normalizes pixel values
- tiles the image for memory efficiency
- performs model inference on each tile
- stitches the tiles back together

This is done in [src/core/inference.py](../src/core/inference.py).

### 4. Preserve GIS quality and metadata

The output does not just become a generic image. The original spatial metadata is retained:
- affine transform
- width/height
- CRS
- geospatial bounds

This ensures the super-resolved image can still be used in GIS tools and mapping workflows.

### 5. Compare with a baseline

The model result is compared with a bicubic interpolation baseline. This gives a more realistic measure of quality than comparing only the model output to a blank canvas.

The system reports:
- PSNR: pixel fidelity measurement
- SSIM: structural similarity measurement

### 6. Extract vector features

After the image is super-resolved, the system runs image-processing techniques to detect likely boundaries, structures, and footprint-like regions. It does this by:
- converting to grayscale
- thresholding image intensities
- applying edge detection
- finding contours
- simplifying and creating polygons
- exporting GeoJSON

This is a GIS-oriented output layer generated from the enhanced image.

### 7. Present through the dashboard

The Streamlit dashboard loads the uploaded or demo raster and shows:
- original image preview
- super-resolved image preview
- metrics
- map overlay of extracted features
- downloadable GeoTIFF and GeoJSON

## Models used

### SatelliteSRNet
This is the main model used in training and inference.

It is a residual convolutional neural network designed for multispectral raster upsampling. It focuses on restoring high-frequency spatial details in satellite imagery while preserving spectral consistency.

Architectural ideas used:
- residual learning
- convolutional feature extraction
- upsampling by sub-pixel rearrangement
- output activation with sigmoid for normalized values

### Compatibility layer for legacy checkpoints

The inference service also includes a compatibility wrapper for older checkpoint architectures. This allows older model files that were trained with an earlier scaling design to still be loaded.

## Techniques used

### Percentile normalization
Each band is normalized using 2nd and 98th percentiles, which makes the image robust to outliers and helps the model learn consistently.

### Tile-based inference
Very large rasters are split into smaller tiles. Each tile is processed separately and stitched back together. This reduces GPU and memory pressure.

### Color-statistics matching
The model output is adjusted to match the mean and variance of the baseline reference image, improving visual consistency and making the output more robust when comparing model output against bicubic interpolation.

### Contour-based vectorization
The vector layer is produced by converting image edges and shapes into polygons. This is a classic GIS approach to turn image-based features into map-ready geometry.

## End-to-end summary

The project follows this general path:

1. Input raster is uploaded
2. Data is normalized and prepared
3. Model upsamples the image
4. Output is georeferenced and saved as GeoTIFF
5. Feature extraction generates GeoJSON
6. Metrics are computed to validate quality
7. UI shows previews and allows downloads

This combination of machine learning, geospatial processing, and GIS feature extraction makes SatFeed a satellite super-resolution + mapping pipeline rather than just a simple image upscaling tool.

## In one sentence

SatFeed learns how to sharpen low-resolution satellite imagery, keeps the geospatial meaning of the image intact, and converts the enhanced result into useful GIS products like GeoTIFFs and vector maps.

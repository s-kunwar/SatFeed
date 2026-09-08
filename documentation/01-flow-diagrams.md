# SatFeed Flow Diagrams

This document shows the end-to-end flow of the SatFeed system: from raw GeoTIFF upload to enhanced output and GIS-ready vector extraction.

## 1) System-level end-to-end flow

```mermaid
flowchart TD
    A[User uploads GeoTIFF] --> B[Validate raster and band count]
    B --> C[Read RGB bands]
    C --> D[Percentile normalization]
    D --> E[Tile large raster into 512x512 windows]
    E --> F[Run PyTorch SR model per tile]
    F --> G[Reconstruct full super-resolved raster]
    G --> H[Match output color statistics to baseline]
    H --> I[Compute PSNR and SSIM]
    I --> J[Write georeferenced GeoTIFF]
    J --> K[Run OpenCV contour extraction]
    K --> L[Create GeoJSON polygons]
    L --> M[Display preview + metrics + downloads]
```

## 2) Training flow

```mermaid
flowchart TD
    A[Pair LR and HR GeoTIFFs] --> B[Extract aligned patches]
    B --> C[Normalize band values to [0,1]]
    C --> D[Save archive as .npz]
    D --> E[Load dataset in train.py]
    E --> F[SatelliteSRNet forward pass]
    F --> G[Compute MSE loss]
    G --> H[Backprop + Adam optimizer]
    H --> I[Save checkpoint if loss improves]
    I --> J[best_model.pth]
```

## 3) Inference flow inside the API

```mermaid
flowchart TD
    A[POST /predict with GeoTIFF] --> B[Check model exists]
    B --> C[Open raster with rasterio]
    C --> D{Raster count >= 3 bands?}
    D -- No --> E[HTTP 400: invalid input]
    D -- Yes --> F[Downsample only if very large]
    F --> G[Prepare input_tensor]
    G --> H[Clear torch memory]
    H --> I[_infer_in_tiles]
    I --> J[Baseline bicubic + model output arrays]
    J --> K[Color-statistics matching]
    K --> L[Convert to uint8 for metrics]
    L --> M[Compute PSNR + SSIM]
    M --> N[Write super-resolved GeoTIFF]
    N --> O[_extract_vector_mapping]
    O --> P[Write GeoJSON]
    P --> Q[Return metadata and paths]
```

## 4) Tile-based inference flow

```mermaid
flowchart LR
    A[Input raster tensor] --> B[height, width]
    B --> C[For each tile in grid]
    C --> D[Pad tile to tile_size boundary]
    D --> E[Run model on tile]
    E --> F[Upsample output by scale factor]
    F --> G[Store result in output canvas]
    G --> H[Tile stitching]
    H --> I[Full super-resolved image]
```

## 5) Vector extraction flow

```mermaid
flowchart TD
    A[Super-resolved image] --> B[Use first 3 channels as RGB]
    B --> C[Convert to grayscale]
    C --> D[Adaptive threshold + Canny edge detection]
    D --> E[Combine binary masks]
    E --> F[Find contours]
    F --> G[Filter small contours by minimum area]
    G --> H[Convert contours to polygons]
    H --> I[Fix invalid geometry]
    I --> J[Write GeoJSON]
```

## 6) Dashboard flow

```mermaid
flowchart TD
    A[Streamlit app loads] --> B[User selects GeoTIFF or demo]
    B --> C[Run SatFeed Super-Resolution button]
    C --> D[Progress stage updates]
    D --> E[Local or API inference]
    E --> F[Save result to session state]
    F --> G[Display metrics cards]
    F --> H[Show original vs enhanced image]
    F --> I[Render GeoJSON overlay on map]
    F --> J[Enable downloads for TIFF and GeoJSON]
```

## 7) Model architecture flow

```mermaid
flowchart TD
    A[Input tensor C x H x W] --> B[Conv2d + ReLU]
    B --> C[Residual blocks]
    C --> D[Conv2d + residual skip]
    D --> E[Conv2d -> PixelShuffle upsampling]
    E --> F[Output conv and Sigmoid]
    F --> G[Higher-resolution tensor]
```

## 8) Data lifecycle

```mermaid
flowchart TD
    A[Raw satellite raster] --> B[Patch extraction]
    B --> C[Normalization]
    C --> D[Training]
    D --> E[Checkpoint]
    E --> F[Inference]
    F --> G[Output GeoTIFF]
    F --> H[GeoJSON features]
    G --> I[GIS usage]
    H --> I
```

## 9) Complete operational summary

```mermaid
flowchart LR
    A[Low-resolution multispectral GeoTIFF] --> B[SatFeed pipeline]
    B --> C[Super-resolved GeoTIFF]
    B --> D[Feature vectors]
    B --> E[Metrics]
    C --> F[GIS analysis / mapping]
    D --> F
    E --> G[Quality reporting]
```

## Notes

- The system uses a 2x residual upscaling model in the training code and a 4x inference pipeline in the service layer.
- Large rasters are processed in tiles to avoid memory blowups.
- The model output is normalized and then converted to georeferenced output preserving source transform and CRS.
- The vector extraction stage is heuristic image processing rather than a learned segmentation model.

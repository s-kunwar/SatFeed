# SatFeed Components and Parameters

## 1. Core system components

SatFeed is built as a geospatial AI pipeline with several distinct components working together.

### 1.1 Data input layer

This is the first step in the system:
- GeoTIFF raster upload
- multispectral satellite imagery input
- support for at least 3-band RGB input
- optional 4-band RGB + NIR support in the training pipeline

The input is expected to be georeferenced raster data, not just a standard JPEG/PNG image. That is why the system relies on rasterio, affine transforms, and CRS metadata.

### 1.2 Preprocessing layer

This component prepares raw pixels for the model:
- band extraction
- nodata handling
- invalid value removal
- percentile normalization
- patch extraction
- tensor conversion for deep learning

This is the layer that turns raw raster data into clean model-ready arrays.

### 1.3 Model layer

The model layer contains the neural network, which is responsible for learning the mapping from low-resolution to high-resolution satellite imagery.

The main architecture is:
- SatelliteSRNet
- residual convolutional blocks
- sub-pixel reconstruction using PixelShuffle
- output in normalized [0, 1] range

This layer is the heart of the super-resolution step.

### 1.4 Inference layer

The inference layer is responsible for running the model on new data efficiently.

It includes:
- tile-based raster processing
- memory-bounded inference
- output reconstruction from tiles
- color-statistics matching
- quality computation (PSNR/SSIM)

This is what makes the pipeline practical for large real-world rasters.

### 1.5 Geospatial output layer

This layer preserves the GIS meaning of the output image:
- transform retention
- bounds retention
- raster width/height preservation
- CRS retention
- GeoTIFF writing

This ensures the enhanced output remains spatially valid for mapping software.

### 1.6 Vectorization layer

This layer converts enhanced raster output into GIS vector features:
- grayscale conversion
- thresholding
- edge extraction
- contour detection
- polygon generation
- GeoJSON export

The output is a vector representation of detected features, which can then be overlayed on maps.

### 1.7 Frontend / dashboard layer

The dashboard layer presents everything to the user:
- upload panel
- progress indicators
- preview comparison
- metrics
- map overlay
- downloads

It is built using Streamlit and pydeck.

## 2. GIS-related components

The project is strongly GIS-aware. The following are the core spatial components:

### 2.1 Raster dataset handling

The system uses rasterio to read and write GeoTIFF data. This supports:
- reading band values
- assigning geospatial transform
- managing nodata values
- writing georeferenced output

### 2.2 Coordinate Reference System (CRS)

The system retains the original spatial coordinate system from the input. This is essential because the output must align with the real world in GIS software.

### 2.3 Affine transform

Affine transforms define how raster pixels map to real-world coordinates. SatFeed preserves this so the super-resolved output remains aligned with the image footprint and surrounding map layers.

### 2.4 Bounds and extent

The code tracks the source bounds and uses them in output generation. This keeps the spatial extent consistent.

### 2.5 GeoJSON vector output

The extracted feature polygons are exported as GeoJSON. This makes the result usable in GIS tools, web mapping platforms, and geospatial analysis pipelines.

## 3. Channel-related explanation

### 3.1 Number of channels in the model

The model is designed for multispectral raster data, typically with multiple bands.

Typical cases:
- 3-band RGB input: common for visible imagery
- 4-band RGB + NIR: supported by training and model design

In the model definition, the input channel count can be configured through the in_channels argument. The default is 4, but the training and inference logic supports other values based on the dataset.

### 3.2 Why multiple channels matter

Each band carries different spectral information:
- red band: vegetation stress / red reflectance
- green band: vegetation and land surface response
- blue band: atmospheric and water-related response
- NIR band: vegetation health / land cover separation

Using multiple channels lets the model learn more than just visible appearance. It can preserve spectral relationships that matter for remote sensing tasks.

### 3.3 Output channels

The output channels are usually aligned with the target imagery channels. In many cases, the output is similar to the input number of channels, preserving the multispectral structure.

The code supports output channels matching either the input or checkpoint-specific values depending on the trained model.

## 4. Main parameters and configuration

### 4.1 Patch size

The preprocessing script uses a patch size default of 64 in the LR raster domain. This defines the local regions from which training pairs are generated.

Examples:
- patch_size=64
- LR patch = 64 x 64
- HR patch = 128 x 128 if the scale factor is 2

### 4.2 Scale factor

The project supports scale-based super-resolution. The main architecture is structured around upsampling factors such as 2 and 4.

Examples:
- scale_factor = 2 in the patch extraction module
- scale_factor = 4 in the API inference pipeline

This means the system can upscale imagery by multiple factors depending on the deployment and model checkpoint.

### 4.3 Tile size

The inference layer uses a tile size of 512 for processing large rasters. This controls the chunk size used while reconstructing the image.

This is a practical memory-management parameter.

### 4.4 Feature channels

In the model definition, the network uses a feature dimension such as 64. This determines the internal number of feature maps used inside residual blocks.

Larger values can improve capacity, but they also increase memory and computational cost.

### 4.5 Number of residual blocks

The model uses multiple residual blocks internally. This allows the model to learn deeper representations without losing signal quality through deeper network instability.

### 4.6 Normalization percentiles

The project uses 2nd and 98th percentiles for dynamic-range normalization. These values are robust against outliers and make model input stable.

## 5. Image quality metrics

The system evaluates the result with:

### 5.1 PSNR

- Measures pixel-wise reconstruction quality
- higher is better
- useful for general image fidelity comparison

### 5.2 SSIM

- measures structural similarity
- more aligned with human perception of image quality
- useful for evaluating whether the result preserves important structures

## 6. Model and data assumptions

The project assumes:
- paired LR/HR training data exists
- georeferencing metadata is important
- bands are ordered and usable for RGB inference
- output quality should be judged not just visually but also with objective metrics
- extracted vector features are derived from image structure and not from a separate semantic segmentation model

## 7. Practical components summary

In short, SatFeed has the following main components:

- GIS raster layer: reads and writes georeferenced GeoTIFFs
- ML model layer: learns super-resolution mapping
- normalization layer: stabilizes data input
- tiling layer: handles large rasters efficiently
- metric layer: validates reconstruction quality
- vector extraction layer: converts the enhanced image into map-ready features
- dashboard layer: makes the result interactive and downloadable

## 8. One-sentence definition

SatFeed is a geospatial deep-learning system that combines multispectral satellite raster processing, super-resolution CNN inference, GIS metadata preservation, and vector extraction to turn low-resolution imagery into higher-quality, map-ready geospatial outputs.

import rasterio
import numpy as np
from rasterio.transform import from_origin

# Create synthetic Low-Res raster (e.g., 4 bands, 256x256)
lr_data = np.random.rand(4, 256, 256).astype(np.float32)
transform_lr = from_origin(0, 0, 20, 20) # 20m resolution simulation
with rasterio.open(
    'sample_lr.tif', 'w', driver='GTiff',
    height=256, width=256, count=4, dtype=rasterio.float32,
    crs='EPSG:4326', transform=transform_lr
) as dst:
    dst.write(lr_data)

# Create synthetic High-Res raster (Scale factor 2 -> 512x512)
hr_data = np.random.rand(4, 512, 512).astype(np.float32)
transform_hr = from_origin(0, 0, 10, 10) # 10m resolution simulation
with rasterio.open(
    'sample_hr.tif', 'w', driver='GTiff',
    height=512, width=512, count=4, dtype=rasterio.float32,
    crs='EPSG:4326', transform=transform_hr
) as dst:
    dst.write(hr_data)

print("Dummy GeoTIFFs created successfully!")
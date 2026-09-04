import rasterio
from PIL import Image

with rasterio.open('austin1.tif') as src:
    print(f"Number of bands/channels: {src.count}")



with Image.open('austin1.tif') as img:
    print(f"Image mode: {img.mode}, Shape/Size: {img.size}")
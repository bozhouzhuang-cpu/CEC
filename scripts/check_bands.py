#!/usr/bin/env python3
"""
Check number of bands in all .tif files in a folder
"""

import os
import rasterio

def check_bands_in_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith(".tif"):
            file_path = os.path.join(folder_path, file)
            try:
                with rasterio.open(file_path) as src:
                    print(f"{file}: {src.count} bands, Size = {src.width}x{src.height}, CRS = {src.crs}")
            except Exception as e:
                print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    # 👇 Change this path to your Site_1 folder
    folder = "tests/test_dataset"
    check_bands_in_folder(folder)

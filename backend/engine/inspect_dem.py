import os
import sys
from pathlib import Path

# Use GDAL's bundled data files when running from a Conda environment.
gdal_data = Path(sys.prefix) / "Library" / "share" / "gdal"
if (gdal_data / "gdalvrt.xsd").is_file():
    os.environ.setdefault("GDAL_DATA", str(gdal_data))

import matplotlib
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont


project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "data" / "raw"
dem_files = sorted(raw_folder.glob("*.tif")) + sorted(raw_folder.glob("*.tiff"))

if not dem_files:
    raise FileNotFoundError(f"No DEM file found in {raw_folder}")

file_path = dem_files[0]
output_folder = project_folder / "outputs"
output_folder.mkdir(parents=True, exist_ok=True)
output_path = output_folder / "jezero_elevation.png"

with rasterio.open(file_path) as dem:
    elevation = dem.read(1, masked=True)

    print("\n===== JEZERO DEM INFORMATION =====")
    print("File:", file_path.name)
    print("Width:", dem.width, "pixels")
    print("Height:", dem.height, "pixels")
    print("Resolution:", dem.res, "meters")
    print("CRS:", dem.crs)
    print("Minimum elevation:", elevation.min(), "meters")
    print("Maximum elevation:", elevation.max(), "meters")

    minimum = float(elevation.min())
    maximum = float(elevation.max())
    normalized = np.ma.clip((elevation - minimum) / (maximum - minimum), 0, 1)

    # Use Matplotlib's terrain palette, while Pillow writes the pixels because
    # Matplotlib's figure save path crashes natively in this Conda environment.
    terrain = matplotlib.colormaps["terrain"]
    map_rgba = terrain(normalized.filled(0), bytes=True)
    map_image = Image.fromarray(map_rgba[:, :, :3], mode="RGB")

    bar_width = 28
    gap = 18
    right_margin = 118
    top_margin = 70
    bottom_margin = 35
    canvas = Image.new(
        "RGB",
        (map_image.width + gap + bar_width + right_margin,
         map_image.height + top_margin + bottom_margin),
        "white",
    )
    canvas.paste(map_image, (0, top_margin))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 12), "Jezero Crater - Mars Elevation Map", fill="black", font=font)

    bar_x = map_image.width + gap
    color_values = np.linspace(1, 0, map_image.height)
    color_rgba = terrain(color_values, bytes=True)
    color_strip = Image.fromarray(color_rgba[:, :3].reshape(map_image.height, 1, 3), mode="RGB")
    canvas.paste(color_strip.resize((bar_width, map_image.height)), (bar_x, top_margin))
    draw.rectangle(
        (bar_x, top_margin, bar_x + bar_width - 1, top_margin + map_image.height - 1),
        outline="black",
    )
    for fraction, value in ((0, maximum), (0.5, (minimum + maximum) / 2), (1, minimum)):
        y = top_margin + round(fraction * (map_image.height - 1))
        draw.line((bar_x + bar_width, y, bar_x + bar_width + 5, y), fill="black")
        draw.text((bar_x + bar_width + 9, y - 4), f"{value:.0f} m", fill="black", font=font)

    canvas.save(output_path, format="PNG")

if not output_path.is_file():
    raise RuntimeError(f"Map image was not created: {output_path}")

image_size = output_path.stat().st_size
if image_size <= 0:
    raise RuntimeError(f"Map image is empty: {output_path}")

print("\nMap saved successfully!")
print("Location:", output_path)
print("File size:", f"{image_size:,} bytes")

# Open the finished PNG in the system's default image viewer when available.
if hasattr(os, "startfile"):
    try:
        os.startfile(output_path)
        print("Opened the map in the default image viewer.")
    except OSError as exc:
        print("Could not open an image viewer:", exc)

print("SCRIPT FINISHED!")

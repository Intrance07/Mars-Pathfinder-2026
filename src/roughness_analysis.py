
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter, binary_erosion


def main():
    print("Starting terrain roughness analysis...", flush=True)

    project_folder = Path(__file__).resolve().parent.parent
    raw_folder = project_folder / "data" / "raw"
    output_folder = project_folder / "outputs"
    output_folder.mkdir(parents=True, exist_ok=True)

    dem_files = sorted(raw_folder.glob("*.tif")) + sorted(raw_folder.glob("*.tiff"))

    if not dem_files:
        raise FileNotFoundError("No DEM file found in data/raw")

    file_path = dem_files[0]
    output_path = output_folder / "jezero_roughness.png"

    print(f"DEM found: {file_path.name}", flush=True)

    with rasterio.open(file_path) as dem:
        elevation = dem.read(1, masked=True).astype("float32")

        valid_mask = (
            ~np.ma.getmaskarray(elevation)
            & np.isfinite(elevation.data)
        )

        data = elevation.filled(0).astype("float32")
        valid = valid_mask.astype("float32")

        print("STEP 1: DEM loaded", flush=True)

        # نافذة 5x5 بكسلات = 100x100 متر تقريباً
        window_size = 5

        # حساب مجموع الارتفاعات ومجموع مربعاتها
        count = uniform_filter(
            valid, size=window_size, mode="constant"
        )

        mean = uniform_filter(
            data, size=window_size, mode="constant"
        ) / np.maximum(count, 1e-10)

        mean_square = uniform_filter(
            data**2, size=window_size, mode="constant"
        ) / np.maximum(count, 1e-10)

        variance = np.maximum(mean_square - mean**2, 0)

        roughness = np.sqrt(variance)

        # الاحتفاظ بالمناطق التي تحتوي نافذتها بالكامل على بيانات صالحة
        valid_roughness = binary_erosion(
            valid_mask,
            structure=np.ones((window_size, window_size), dtype=bool),
            border_value=0
        )

        roughness = np.ma.array(
            roughness,
            mask=~valid_roughness
        )

        print("STEP 2: Roughness calculated", flush=True)

        values = roughness.compressed()

        if values.size == 0:
            raise ValueError("No valid roughness pixels found.")

        minimum = float(values.min())
        maximum = float(values.max())
        average = float(values.mean())
        percentile_95 = float(np.percentile(values, 95))

        print("STEP 3: Statistics calculated", flush=True)

        # رسم الخريطة
        cmap = matplotlib.colormaps["terrain"]
        height, width = roughness.shape

        display_max = max(percentile_95, 0.01)

        rgb = np.empty((height, width, 3), dtype=np.uint8)

        for start in range(0, height, 128):
            end = min(start + 128, height)

            values_chunk = np.clip(
                roughness.data[start:end] / display_max,
                0, 1
            )

            rows = cmap(values_chunk, bytes=True)[:, :, :3]

            mask_chunk = np.ma.getmaskarray(roughness)[start:end]
            rows[mask_chunk] = (255, 255, 255)

            rgb[start:end] = rows

        map_image = Image.fromarray(rgb, mode="RGB")

        canvas = Image.new(
            "RGB",
            (width + 130, height + 60),
            "white"
        )

        canvas.paste(map_image, (0, 40))

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        draw.text(
            (8, 10),
            "Jezero Crater - Terrain Roughness",
            fill="black",
            font=font
        )

        bar_x = width + 20

        color_strip = Image.fromarray(
            cmap(np.linspace(1, 0, height), bytes=True)[:, None, :3],
            mode="RGB"
        ).resize((25, height))

        canvas.paste(color_strip, (bar_x, 40))

        draw.text((bar_x + 35, 40), f"{display_max:.2f} m", fill="black", font=font)
        draw.text((bar_x + 35, height // 2 + 40), "Roughness", fill="black", font=font)
        draw.text((bar_x + 35, height + 25), "0 m", fill="black", font=font)

        canvas.save(output_path, format="PNG")

    print("\n===== JEZERO ROUGHNESS ANALYSIS =====")
    print(f"Minimum: {minimum:.3f} meters")
    print(f"Maximum: {maximum:.3f} meters")
    print(f"Average: {average:.3f} meters")
    print(f"95th percentile: {percentile_95:.3f} meters")
    print("Map saved at:", output_path)
    print("ROUGHNESS ANALYSIS FINISHED!", flush=True)

    if hasattr(os, "startfile"):
        os.startfile(output_path)


if __name__ == "__main__":
    main()

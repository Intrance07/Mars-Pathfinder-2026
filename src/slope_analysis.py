
import os
import sys
import traceback
from pathlib import Path

# Make GDAL's Conda support files discoverable
gdal_data = Path(sys.prefix) / "Library" / "share" / "gdal"
if (gdal_data / "gdalvrt.xsd").is_file():
    os.environ.setdefault("GDAL_DATA", str(gdal_data))

import matplotlib
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion


def main():
    print("Starting slope analysis...", flush=True)

    project_folder = Path(__file__).resolve().parent.parent
    raw_folder = project_folder / "data" / "raw"
    output_folder = project_folder / "outputs"
    output_folder.mkdir(parents=True, exist_ok=True)

    dem_files = sorted(raw_folder.glob("*.tif")) + sorted(raw_folder.glob("*.tiff"))

    if not dem_files:
        raise FileNotFoundError(f"No DEM file found in {raw_folder}")

    file_path = dem_files[0]
    output_path = output_folder / "jezero_slope.png"

    print(f"DEM file found: {file_path.name}", flush=True)

    with rasterio.open(file_path) as dem:
        elevation = dem.read(1, masked=True).astype("float32")
        pixel_size_x, pixel_size_y = dem.res

        if pixel_size_x <= 0 or pixel_size_y <= 0:
            raise ValueError(f"Invalid DEM pixel resolution: {dem.res}")

        print("STEP 1: DEM loaded", flush=True)
        print("Dimensions:", elevation.shape, flush=True)
        print("Resolution (x, y):", (pixel_size_x, pixel_size_y), "meters", flush=True)

        valid_mask = ~np.ma.getmaskarray(elevation) & np.isfinite(elevation.data)
        valid_count = int(valid_mask.sum())

        print(f"Valid pixels: {valid_count:,}", flush=True)

        if valid_count == 0:
            raise ValueError("DEM contains no valid elevation pixels.")

        # Calculate elevation gradient
        print("STEP 2: Calculating elevation gradient...", flush=True)

        elevation_filled = elevation.filled(0)

        gradient_y, gradient_x = np.gradient(
            elevation_filled,
            pixel_size_y,
            pixel_size_x
        )

        del elevation_filled

        print("STEP 2: Gradient calculated", flush=True)

        # Calculate slope in degrees
        print("STEP 3: Calculating slope in degrees...", flush=True)

        slope_values = np.degrees(
            np.arctan(np.hypot(gradient_x, gradient_y))
        )

        del gradient_x, gradient_y

        # Exclude nodata-adjacent pixels
        valid_slope_mask = binary_erosion(
            valid_mask,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0
        )

        del valid_mask

        slope = np.ma.array(
            slope_values,
            mask=~valid_slope_mask,
            copy=False
        )

        del slope_values, valid_slope_mask

        print("STEP 3: Slope calculated", flush=True)

        # Extract slope statistics
        print("STEP 3A: Extracting slope statistics...", flush=True)

        valid_slopes = slope.compressed()

        if valid_slopes.size == 0:
            raise ValueError("No valid slope pixels found.")

        minimum_slope = float(valid_slopes.min())
        maximum_slope = float(valid_slopes.max())
        average_slope = float(valid_slopes.mean())

        print("STEP 3B: Statistics calculated", flush=True)

        # ==========================================
        # STEP 3C: SLOPE DISTRIBUTION ANALYSIS
        # ==========================================

        print("\n===== SLOPE DISTRIBUTION =====", flush=True)

        ranges = [
            ("0 - 5 degrees", 0, 5),
            ("5 - 10 degrees", 5, 10),
            ("10 - 15 degrees", 10, 15),
            ("15 - 20 degrees", 15, 20),
            ("20 - 30 degrees", 20, 30),
            ("Above 30 degrees", 30, 90.000001)
        ]

        total_pixels = valid_slopes.size

        for label, lower, upper in ranges:
            count = np.sum(
                (valid_slopes >= lower) &
                (valid_slopes < upper)
            )

            percentage = (count / total_pixels) * 100

            print(
                f"{label}: {percentage:.2f}% "
                f"({count:,} pixels)",
                flush=True
            )

        # ==========================================
        # STEP 4: RENDER SLOPE MAP
        # ==========================================

        print("\nSTEP 4A: Rendering slope map...", flush=True)

        cmap = matplotlib.colormaps["terrain"]

        height, width = slope.shape
        rgb = np.empty((height, width, 3), dtype=np.uint8)

        slope_mask = np.ma.getmaskarray(slope)

        for row_start in range(0, height, 128):
            row_end = min(row_start + 128, height)

            values = np.clip(
                slope.data[row_start:row_end] / 30.0,
                0.0,
                1.0
            )

            rows = cmap(values, bytes=True)[:, :, :3]

            rows[slope_mask[row_start:row_end]] = (255, 255, 255)

            rgb[row_start:row_end] = rows

        map_image = Image.fromarray(rgb, mode="RGB")
        del rgb

        # Create map canvas
        bar_width = 28
        gap = 18
        right_margin = 90
        top_margin = 36

        canvas = Image.new(
            "RGB",
            (width + gap + bar_width + right_margin, height + top_margin + 24),
            "white"
        )

        canvas.paste(map_image, (0, top_margin))

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        draw.text(
            (8, 10),
            "Jezero Crater - Slope Map",
            fill="black",
            font=font
        )

        # Color scale
        bar_x = width + gap

        color_strip = Image.fromarray(
            cmap(
                np.linspace(1, 0, height),
                bytes=True
            )[:, None, :3].reshape(height, 1, 3),
            mode="RGB"
        )

        canvas.paste(
            color_strip.resize((bar_width, height)),
            (bar_x, top_margin)
        )

        draw.rectangle(
            (
                bar_x,
                top_margin,
                bar_x + bar_width - 1,
                top_margin + height - 1
            ),
            outline="black"
        )

        for fraction, label in (
            (0, "30 deg"),
            (0.5, "15 deg"),
            (1, "0 deg")
        ):
            y = top_margin + round(fraction * (height - 1))

            draw.line(
                (bar_x + bar_width, y, bar_x + bar_width + 5, y),
                fill="black"
            )

            draw.text(
                (bar_x + bar_width + 9, y - 4),
                label,
                fill="black",
                font=font
            )

        # Save PNG
        print("STEP 4B: Writing PNG...", flush=True)

        canvas.save(output_path, format="PNG")

    # Verify saved image
    if not output_path.is_file():
        raise RuntimeError(f"Map image was not created: {output_path}")

    file_size = output_path.stat().st_size

    if file_size <= 0:
        raise RuntimeError(f"Map image is empty: {output_path}")

    with Image.open(output_path) as saved_image:
        saved_image.verify()

    # Final results
    print("\n===== JEZERO SLOPE ANALYSIS =====")
    print(f"Minimum slope: {minimum_slope:.3f} degrees")
    print(f"Maximum slope: {maximum_slope:.3f} degrees")
    print(f"Average slope: {average_slope:.3f} degrees")
    print("Map saved at:", output_path)
    print(f"PNG size: {file_size:,} bytes", flush=True)

    if hasattr(os, "startfile"):
        try:
            os.startfile(output_path)
            print("Opened the map in the default image viewer.", flush=True)
        except OSError as exc:
            print("Could not open an image viewer:", exc, flush=True)

    print("\nSCRIPT FINISHED!", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            "ERROR: Slope analysis failed. Full traceback:",
            file=sys.stderr,
            flush=True
        )
        traceback.print_exc()
        raise

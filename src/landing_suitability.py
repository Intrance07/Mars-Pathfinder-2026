
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, binary_erosion
from PIL import Image, ImageDraw, ImageFont


def main():
    print("Starting landing suitability analysis...", flush=True)

    project = Path(__file__).resolve().parent.parent
    raw_folder = project / "data" / "raw"
    output_folder = project / "outputs"
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_folder.glob("*.tif")) + sorted(raw_folder.glob("*.tiff"))

    if not files:
        raise FileNotFoundError("No DEM file found.")

    with rasterio.open(files[0]) as dem:
        elevation = dem.read(1, masked=True).astype("float32")
        pixel_y, pixel_x = dem.res

        valid = (
            ~np.ma.getmaskarray(elevation)
            & np.isfinite(elevation.data)
        )

        print("STEP 1: DEM loaded", flush=True)

        # --------------------------------
        # 1. Calculate slope
        # --------------------------------

        filled = elevation.filled(0)

        gradient_y, gradient_x = np.gradient(
            filled,
            pixel_y,
            pixel_x
        )

        slope = np.degrees(
            np.arctan(np.hypot(gradient_x, gradient_y))
        )

        slope_valid = binary_erosion(
            valid,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0
        )

        print("STEP 2: Slope calculated", flush=True)

        # --------------------------------
        # 2. Calculate roughness
        # --------------------------------

        valid_float = valid.astype("float32")
        data = filled.astype("float32")

        window = 5

        count = uniform_filter(
            valid_float,
            size=window,
            mode="constant"
        )

        safe_count = np.maximum(count, 1e-10)

        mean = uniform_filter(
            data,
            size=window,
            mode="constant"
        ) / safe_count

        mean_square = uniform_filter(
            data ** 2,
            size=window,
            mode="constant"
        ) / safe_count

        roughness = np.sqrt(
            np.maximum(mean_square - mean ** 2, 0)
        )

        roughness_valid = binary_erosion(
            valid,
            structure=np.ones((window, window), dtype=bool),
            border_value=0
        )

        print("STEP 3: Roughness calculated", flush=True)

        # --------------------------------
        # 3. Combine valid areas
        # --------------------------------

        combined_valid = slope_valid & roughness_valid

        # --------------------------------
        # 4. Classify landing suitability
        # --------------------------------

        # 0 = No data
        # 1 = More suitable
        # 2 = Caution
        # 3 = Unsuitable under current criteria

        classification = np.zeros(
            elevation.shape,
            dtype=np.uint8
        )

        more_suitable = (
            (slope <= 5) &
            (roughness <= 2)
        )

        caution = (
            (slope <= 15) &
            (roughness <= 5)
        )

        classification[
            combined_valid & more_suitable
        ] = 1

        classification[
            combined_valid & caution & ~more_suitable
        ] = 2

        classification[
            combined_valid & ~caution
        ] = 3

        print("STEP 4: Classification completed", flush=True)

        # --------------------------------
        # 5. Calculate area percentages
        # --------------------------------

        total = int(combined_valid.sum())

        if total == 0:
            raise ValueError("No valid combined pixels.")

        labels = {
            1: "More suitable",
            2: "Caution",
            3: "Unsuitable"
        }

        print("\n===== LANDING SUITABILITY =====")

        for code, label in labels.items():
            count = int(np.sum(classification == code))
            percentage = count / total * 100

            print(
                f"{label}: {percentage:.2f}% "
                f"({count:,} pixels)"
            )

        # --------------------------------
        # 6. Create classification map
        # --------------------------------

        height, width = classification.shape

        rgb = np.full(
            (height, width, 3),
            255,
            dtype=np.uint8
        )

        rgb[classification == 1] = (50, 180, 90)
        rgb[classification == 2] = (245, 200, 60)
        rgb[classification == 3] = (210, 65, 65)

        image = Image.fromarray(rgb, mode="RGB")

        canvas = Image.new(
            "RGB",
            (width, height + 100),
            "white"
        )

        canvas.paste(image, (0, 45))

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        draw.text(
            (10, 10),
            "Jezero Crater - Landing Suitability",
            fill="black",
            font=font
        )

        draw.text(
            (10, height + 52),
            "GREEN: More suitable",
            fill=(30, 140, 70),
            font=font
        )

        draw.text(
            (220, height + 52),
            "YELLOW: Caution",
            fill=(180, 130, 0),
            font=font
        )

        draw.text(
            (390, height + 52),
            "RED: Unsuitable",
            fill=(190, 40, 40),
            font=font
        )

        output_path = output_folder / "jezero_landing_suitability.png"
        canvas.save(output_path)

    print("\nSTEP 5: Map saved")
    print("Output:", output_path)
    print("LANDING SUITABILITY ANALYSIS FINISHED!", flush=True)

    if hasattr(os, "startfile"):
        os.startfile(output_path)


if __name__ == "__main__":
    main()

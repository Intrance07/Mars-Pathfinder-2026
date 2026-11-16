"""Estimate landing suitability around one DEM pixel (row, col)."""

import argparse
import math
import os
import sys
import traceback
from numbers import Real
from pathlib import Path

# Make the GDAL support files bundled with Conda discoverable before Rasterio loads GDAL.
gdal_data = Path(sys.prefix) / "Library" / "share" / "gdal"
if (gdal_data / "gdalvrt.xsd").is_file():
    os.environ.setdefault("GDAL_DATA", str(gdal_data))

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, uniform_filter
from rasterio.windows import Window


PIXEL_SIZE_M = 20.0
ROUGHNESS_WINDOW = 5
MAX_RENDER_DIMENSION = 301
CLASS_NAMES = {1: "More Suitable", 2: "Caution", 3: "Unsuitable"}
CLASS_COLORS = {
    1: (50, 180, 90),
    2: (245, 200, 60),
    3: (210, 65, 65),
}
ASSESSMENT_RULE = (
    "The most prevalent suitability class in the circle; ties resolve to the "
    "more conservative class (Unsuitable > Caution > More Suitable)."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate landing suitability around a DEM pixel coordinate."
    )
    parser.add_argument("--row", type=int, required=True, help="DEM row index (0-based).")
    parser.add_argument("--col", type=int, required=True, help="DEM column index (0-based).")
    parser.add_argument(
        "--radius-m",
        type=float,
        default=500.0,
        help="Evaluation circle radius in meters (default: 500).",
    )
    return parser.parse_args()


def _validate_inputs(row, col, radius_m):
    if isinstance(row, bool) or not isinstance(row, int):
        raise ValueError("row must be an integer DEM row index.")
    if isinstance(col, bool) or not isinstance(col, int):
        raise ValueError("col must be an integer DEM column index.")
    if isinstance(radius_m, bool) or not isinstance(radius_m, Real):
        raise ValueError("radius_m must be a numeric value.")
    radius_m = float(radius_m)
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and greater than zero.")
    return radius_m


def _find_dem(raw_folder):
    dem_files = sorted(raw_folder.glob("*.tif")) + sorted(raw_folder.glob("*.tiff"))
    if not dem_files:
        raise FileNotFoundError(f"No DEM GeoTIFF found in {raw_folder}")
    return dem_files[0]


def _render_context(
    classification,
    row,
    col,
    radius_m,
    pixel_size_x,
    pixel_size_y,
    *,
    offset=(0, 0),
    full_shape=None,
):
    """Keep a compact classification preview around the evaluated circle.

    This is a sampled class-code grid, not an elevation/DEM array. The default
    500 m assessment uses a 51x51 preview at 20 m/pixel. Larger radii are
    sampled so neither JSON responses nor the renderer carry the full DEM.
    """
    row_radius = math.ceil(radius_m / pixel_size_y)
    col_radius = math.ceil(radius_m / pixel_size_x)
    full_height, full_width = full_shape or classification.shape
    row_start = max(0, row - row_radius)
    row_end = min(full_height, row + row_radius + 1)
    col_start = max(0, col - col_radius)
    col_end = min(full_width, col + col_radius + 1)
    preview_height = row_end - row_start
    preview_width = col_end - col_start
    stride = max(
        1,
        math.ceil(max(preview_height, preview_width) / MAX_RENDER_DIMENSION),
    )
    local_row_start = row_start - offset[0]
    local_row_end = row_end - offset[0]
    local_col_start = col_start - offset[1]
    local_col_end = col_end - offset[1]
    grid = classification[
        local_row_start:local_row_end:stride,
        local_col_start:local_col_end:stride,
    ]
    return {
        "classification_grid": grid.tolist(),
        "row_start": int(row_start),
        "col_start": int(col_start),
        "pixel_stride": int(stride),
        "center_row": float((row - row_start) / stride),
        "center_col": float((col - col_start) / stride),
        "radius_y_pixels": float(radius_m / pixel_size_y / stride),
        "radius_x_pixels": float(radius_m / pixel_size_x / stride),
        "class_names": {str(code): name for code, name in CLASS_NAMES.items()},
    }


def evaluate_landing_point(row, col, radius_m=500.0, *, verbose=False, dem_path=None):
    """Evaluate a circular landing area and return JSON-compatible summaries.

    The returned result contains scalar statistics and a compact class-code
    preview for rendering. It never returns DEM, slope, or roughness arrays.
    Set ``verbose=True`` to print the progress messages used by the CLI.
    """
    radius_m = _validate_inputs(row, col, radius_m)
    project = Path(__file__).resolve().parent.parent
    raw_folder = project / "data" / "raw"
    dem_path = Path(dem_path) if dem_path is not None else _find_dem(raw_folder)
    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM GeoTIFF not found: {dem_path}")

    if verbose:
        print(f"STEP 1: Opening DEM: {dem_path.name}", flush=True)

    with rasterio.open(dem_path) as dem:
        height, width = dem.height, dem.width
        pixel_size_x, pixel_size_y = map(abs, dem.res)

        if row < 0 or row >= height or col < 0 or col >= width:
            raise ValueError(
                f"Pixel (row={row}, col={col}) is outside the DEM bounds "
                f"(rows 0..{height - 1}, cols 0..{width - 1})."
            )
        if not np.isclose(pixel_size_x, PIXEL_SIZE_M) or not np.isclose(
            pixel_size_y, PIXEL_SIZE_M
        ):
            raise ValueError(
                f"Expected {PIXEL_SIZE_M:g} m pixels; DEM resolution is "
                f"({pixel_size_x:g}, {pixel_size_y:g}) m."
            )

        # Only compute terrain layers in the evaluation bounding box plus the
        # two-cell roughness halo. The halo preserves the existing 5x5 local
        # statistics and central-difference gradients for every evaluated cell.
        row_radius = math.ceil(radius_m / pixel_size_y)
        col_radius = math.ceil(radius_m / pixel_size_x)
        eval_row_start = max(0, row - row_radius)
        eval_row_end = min(height, row + row_radius + 1)
        eval_col_start = max(0, col - col_radius)
        eval_col_end = min(width, col + col_radius + 1)
        halo = ROUGHNESS_WINDOW // 2
        row_start = max(0, eval_row_start - halo)
        row_end = min(height, eval_row_end + halo)
        col_start = max(0, eval_col_start - halo)
        col_end = min(width, eval_col_end + halo)
        elevation = dem.read(
            1,
            window=Window(col_start, row_start, col_end - col_start, row_end - row_start),
            masked=True,
        ).astype("float32")
        valid = ~np.ma.getmaskarray(elevation) & np.isfinite(elevation.data)
        if verbose:
            print(
                f"STEP 1: DEM loaded ({height} rows x {width} cols, "
                f"{pixel_size_x:g} m pixels)",
                flush=True,
            )

        local_row = row - row_start
        local_col = col - col_start
        if not valid[local_row, local_col]:
            raise ValueError(
                f"The selected point (row={row}, col={col}) has no valid DEM data."
            )

        if verbose:
            print("STEP 2: Calculating slope using 20 m x/y spacing...", flush=True)
        filled = elevation.filled(0)
        # Keep the existing row/y then column/x gradient calculation unchanged.
        gradient_y, gradient_x = np.gradient(filled, pixel_size_y, pixel_size_x)
        slope = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
        slope_valid = binary_erosion(
            valid, structure=np.ones((3, 3), dtype=bool), border_value=0
        )
        del filled, gradient_x, gradient_y
        if verbose:
            print("STEP 2: Slope calculated in degrees", flush=True)
            print(
                f"STEP 3: Calculating roughness ({ROUGHNESS_WINDOW}x{ROUGHNESS_WINDOW} "
                "pixel local elevation standard deviation)...",
                flush=True,
            )

        valid_float = valid.astype("float32")
        data = elevation.filled(0).astype("float32")
        count = uniform_filter(valid_float, size=ROUGHNESS_WINDOW, mode="constant")
        safe_count = np.maximum(count, 1e-10)
        mean = uniform_filter(data, size=ROUGHNESS_WINDOW, mode="constant") / safe_count
        mean_square = uniform_filter(
            data**2, size=ROUGHNESS_WINDOW, mode="constant"
        ) / safe_count
        roughness = np.sqrt(np.maximum(mean_square - mean**2, 0))
        roughness_valid = binary_erosion(
            valid,
            structure=np.ones((ROUGHNESS_WINDOW, ROUGHNESS_WINDOW), dtype=bool),
            border_value=0,
        )
        combined_valid = slope_valid & roughness_valid
        del valid_float, data, count, safe_count, mean, mean_square
        if verbose:
            print("STEP 3: Roughness calculated in meters", flush=True)
            print("STEP 4: Applying existing suitability thresholds...", flush=True)

        # Keep the existing thresholds and NODATA mask behavior unchanged.
        classification = np.zeros(elevation.shape, dtype=np.uint8)
        more_suitable = (slope <= 5) & (roughness <= 2)
        caution = (slope <= 15) & (roughness <= 5)
        classification[combined_valid & more_suitable] = 1
        classification[combined_valid & caution & ~more_suitable] = 2
        classification[combined_valid & ~caution] = 3
        del more_suitable, caution
        if verbose:
            print("STEP 4: Suitability classification complete", flush=True)

        if classification[local_row, local_col] == 0:
            raise ValueError(
                f"The selected point (row={row}, col={col}) does not have "
                "valid slope and roughness data (it may be near a NoData boundary)."
            )

        radius_pixels = radius_m / PIXEL_SIZE_M
        rows = np.arange(eval_row_start, eval_row_end)[:, None]
        cols = np.arange(eval_col_start, eval_col_end)[None, :]
        distance_squared = (
            ((rows - row) * pixel_size_y) ** 2
            + ((cols - col) * pixel_size_x) ** 2
        )
        in_circle = distance_squared <= radius_m**2
        local_eval_row_start = eval_row_start - row_start
        local_eval_row_end = eval_row_end - row_start
        local_eval_col_start = eval_col_start - col_start
        local_eval_col_end = eval_col_end - col_start
        local_classification = classification[
            local_eval_row_start:local_eval_row_end,
            local_eval_col_start:local_eval_col_end,
        ]
        local_valid = (local_classification > 0) & in_circle
        valid_region_count = int(local_valid.sum())
        if valid_region_count == 0:
            raise ValueError("The evaluation circle contains no valid classified pixels.")

        counts = {
            code: int(np.count_nonzero(local_valid & (local_classification == code)))
            for code in CLASS_NAMES
        }
        percentages = {
            code: count / valid_region_count * 100 for code, count in counts.items()
        }
        slope_region = slope[
            local_eval_row_start:local_eval_row_end,
            local_eval_col_start:local_eval_col_end,
        ]
        roughness_region = roughness[
            local_eval_row_start:local_eval_row_end,
            local_eval_col_start:local_eval_col_end,
        ]
        mean_slope = float(slope_region[local_valid].mean())
        mean_roughness = float(roughness_region[local_valid].mean())

        # Preserve dominant-class selection and conservative tie-breaking.
        final_code = max(counts, key=lambda code: (counts[code], code))
        final_assessment = CLASS_NAMES[final_code]
        render_context = _render_context(
            classification,
            row,
            col,
            radius_m,
            pixel_size_x,
            pixel_size_y,
            offset=(row_start, col_start),
            full_shape=(height, width),
        )

    result = {
        "success": True,
        "row": int(row),
        "col": int(col),
        "radius_m": float(radius_m),
        "pixel_size_m": [float(pixel_size_x), float(pixel_size_y)],
        "valid_pixel_count": valid_region_count,
        "class_counts": {CLASS_NAMES[code]: counts[code] for code in CLASS_NAMES},
        "class_percentages": {
            CLASS_NAMES[code]: float(percentages[code]) for code in CLASS_NAMES
        },
        "mean_slope_deg": mean_slope,
        "mean_roughness_m": mean_roughness,
        "final_assessment": final_assessment,
        "assessment_rule": ASSESSMENT_RULE,
        "render_context": render_context,
    }

    if verbose:
        print("STEP 5: Circular-region statistics calculated", flush=True)
        print("\n===== LANDING POINT ESTIMATE =====")
        print(f"Point (row, col): ({row}, {col})")
        print(f"Radius: {radius_m:g} m ({radius_pixels:g} pixels at 20 m/pixel)")
        print(f"Valid classified pixels in circle: {valid_region_count:,}")
        for code, name in CLASS_NAMES.items():
            print(f"{name}: {percentages[code]:.2f}% ({counts[code]:,} pixels)")
        print(f"Mean slope: {mean_slope:.3f} degrees")
        print(f"Mean roughness: {mean_roughness:.3f} meters")
        print(f"Final assessment: {final_assessment}")
        print(f"Assessment rule: {ASSESSMENT_RULE}", flush=True)
    return result


def render_landing_assessment(result, output_path):
    """Render and save the assessment using its compact class preview only."""
    if not isinstance(result, dict) or not result.get("success"):
        raise ValueError("result must be a successful evaluate_landing_point result.")
    context = result.get("render_context")
    if not isinstance(context, dict) or not context.get("classification_grid"):
        raise ValueError("result does not contain the classification preview needed to render.")

    grid = np.asarray(context["classification_grid"], dtype=np.uint8)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("result contains an invalid classification preview.")
    stride = int(context["pixel_stride"])
    image_scale = max(1, min(8, 800 // max(grid.shape)))
    rgb = np.full((*grid.shape, 3), 255, dtype=np.uint8)
    for code, color in CLASS_COLORS.items():
        rgb[grid == code] = color
    map_image = Image.fromarray(rgb, mode="RGB").resize(
        (grid.shape[1] * image_scale, grid.shape[0] * image_scale),
        Image.Resampling.NEAREST,
    )

    top_margin = 42
    bottom_margin = 32
    right_panel = 250
    canvas = Image.new(
        "RGB",
        (map_image.width + right_panel, map_image.height + top_margin + bottom_margin),
        "white",
    )
    canvas.paste(map_image, (0, top_margin))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    center_x = (context["center_col"] + 0.5) * image_scale
    center_y = top_margin + (context["center_row"] + 0.5) * image_scale
    radius_x = context["radius_x_pixels"] * image_scale
    radius_y = context["radius_y_pixels"] * image_scale
    circle_box = (
        center_x - radius_x,
        center_y - radius_y,
        center_x + radius_x,
        center_y + radius_y,
    )
    draw.ellipse(circle_box, outline="black", width=max(1, image_scale))
    draw.ellipse(circle_box, outline=(0, 255, 255), width=max(1, image_scale // 2))
    draw.ellipse(
        (center_x - 7, center_y - 7, center_x + 7, center_y + 7),
        fill="white",
        outline="black",
        width=2,
    )

    panel_x = map_image.width + 18
    draw.text((panel_x, 12), "Landing point assessment", fill="black", font=font)
    draw.text(
        (panel_x, 36),
        f"row={result['row']}, col={result['col']}",
        fill="black",
        font=font,
    )
    draw.text((panel_x, 54), f"Radius: {result['radius_m']:g} m", fill="black", font=font)
    draw.text(
        (panel_x, 82), f"Result: {result['final_assessment']}", fill="black", font=font
    )
    legend_y = 120
    for code, name in CLASS_NAMES.items():
        draw.rectangle(
            (panel_x, legend_y, panel_x + 14, legend_y + 14),
            fill=CLASS_COLORS[code],
        )
        draw.text((panel_x + 22, legend_y + 2), name, fill="black", font=font)
        legend_y += 22
    draw.line(
        (panel_x, legend_y + 12, panel_x + 26, legend_y + 12),
        fill=(0, 190, 190),
        width=3,
    )
    draw.text((panel_x + 34, legend_y + 7), "Evaluation radius", fill="black", font=font)
    draw.text(
        (panel_x, legend_y + 40),
        "Estimated only; not a safety guarantee.",
        fill="black",
        font=font,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def _verify_png(output_path):
    output_path = Path(output_path)
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Assessment map was not saved correctly: {output_path}")
    with Image.open(output_path) as result_image:
        result_image.verify()
    return output_path.stat().st_size


def main():
    args = parse_args()
    print("Starting landing point evaluation...", flush=True)
    print(f"Requested point: row={args.row}, col={args.col}", flush=True)
    print(f"Evaluation radius: {args.radius_m:g} m", flush=True)

    project = Path(__file__).resolve().parent.parent
    output_path = project / "outputs" / "landing_point_assessment.png"
    result = evaluate_landing_point(args.row, args.col, args.radius_m, verbose=True)

    print("STEP 6: Creating classified map with point and radius overlay...", flush=True)
    render_landing_assessment(result, output_path)
    size = _verify_png(output_path)
    print("STEP 7: Assessment map saved and verified", flush=True)
    print("Output:", output_path)
    print(f"PNG size: {size:,} bytes")
    print(
        "IMPORTANT: This is an estimate based on DEM slope and roughness; "
        "it does not guarantee landing safety.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERROR: Landing point evaluation failed. Full traceback:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1) from None

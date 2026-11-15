"""Preliminary terrain assessment for unique Jezero science-site coordinates."""

import csv
import json
import math
import os
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from textwrap import wrap

# This project assumes GeoJSON coordinates are Mars 2000 sphere lon/lat.
# The assumption is inherited from validate_science_sites.py and is not confirmed.
gdal_data = Path(sys.prefix) / "Library" / "share" / "gdal"
if (gdal_data / "gdalvrt.xsd").is_file():
    os.environ.setdefault("GDAL_DATA", str(gdal_data))

import matplotlib
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from rasterio.windows import Window
from scipy.ndimage import binary_erosion, uniform_filter

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DEM_PATH = RAW_DIR / "M20_JezeroCrater_CTXDEM_20m (1).tif"
SITES_PATH = RAW_DIR / "jezero_10_scientific_sites.geojson"
CSV_PATH = OUTPUT_DIR / "scientific_site_terrain_assessment.csv"
PNG_PATH = OUTPUT_DIR / "scientific_site_terrain_assessment.png"
RADIUS_M = 500.0
ROUGHNESS_WINDOW = 5
SOURCE_CRS = CRS.from_proj4(
    "+proj=longlat +a=3396190 +b=3396190 +no_defs +type=crs"
)
CLASS_NAMES = {1: "More Suitable", 2: "Caution", 3: "Unsuitable"}
CLASS_COLORS = {1: (50, 180, 90), 2: (245, 200, 60), 3: (210, 65, 65)}
SITE_COLORS = [(20, 90, 220), (170, 40, 190), (0, 145, 145), (220, 90, 20), (75, 75, 75), (130, 120, 0)]


def unique_sites_from_geojson(path):
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    features = document.get("features", [])
    grouped = OrderedDict()
    invalid = 0
    for number, feature in enumerate(features, 1):
        props = feature.get("properties") or {} if isinstance(feature, dict) else {}
        geom = feature.get("geometry") or {} if isinstance(feature, dict) else {}
        coords = geom.get("coordinates", []) if isinstance(geom, dict) else []
        name = str(props.get("site_name") or f"Feature {number}") if isinstance(props, dict) else f"Feature {number}"
        if not isinstance(geom, dict) or geom.get("type") != "Point" or not isinstance(coords, (list, tuple)) or len(coords) < 2:
            print(f"[WARN] {name}: invalid Point geometry; skipped.")
            invalid += 1
            continue
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError, OverflowError):
            lon = lat = math.nan
        if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
            print(f"[WARN] {name}: invalid longitude/latitude; skipped.")
            invalid += 1
            continue
        key = (lon, lat)
        if key not in grouped:
            grouped[key] = {"longitude": lon, "latitude": lat, "names": []}
        grouped[key]["names"].append(name)
    return features, list(grouped.values()), invalid


def inverse_pixel(transform, x, y):
    # Use the inverse affine instead of dem.index(): Rasterio's index() crashes
    # natively in this environment. Floor matches its default pixel indexing rule.
    return (~transform) * (x, y)  # column, row (continuous pixel-edge coordinates)


def analyze_site(dem, site, transformer):
    lon, lat = site["longitude"], site["latitude"]
    x, y = transformer.transform(lon, lat)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("Coordinate transformation returned non-finite values")
    col_f, row_f = inverse_pixel(dem.transform, x, y)
    row, col = math.floor(row_f), math.floor(col_f)
    inside = 0 <= row < dem.height and 0 <= col < dem.width
    site.update(x=x, y=y, col_f=col_f, row_f=row_f, in_bounds=inside)
    result = {
        "site_names": "; ".join(site["names"]), "longitude": lon, "latitude": lat,
        "projected_x_m": x, "projected_y_m": y, "row": row, "col": col,
        "status": "OUTSIDE_DEM" if not inside else "ASSESSED", "radius_m": RADIUS_M,
        "valid_dem_pixels": 0, "valid_analysis_pixels": 0,
        "mean_slope_deg": None, "max_slope_deg": None,
        "mean_roughness_m": None, "max_roughness_m": None, "mean_elevation_m": None,
        "more_suitable_pct": None, "caution_pct": None, "unsuitable_pct": None,
    }
    if not inside:
        return result

    # Window covers the circle plus a two-pixel halo required by the 5x5 roughness kernel.
    corners = [inverse_pixel(dem.transform, x + dx, y + dy)
               for dx in (-RADIUS_M, RADIUS_M) for dy in (-RADIUS_M, RADIUS_M)]
    col0 = max(0, math.floor(min(p[0] for p in corners)) - 2)
    col1 = min(dem.width, math.ceil(max(p[0] for p in corners)) + 2)
    row0 = max(0, math.floor(min(p[1] for p in corners)) - 2)
    row1 = min(dem.height, math.ceil(max(p[1] for p in corners)) + 2)
    elev = dem.read(1, window=Window(col0, row0, col1-col0, row1-row0), masked=True).astype("float32")
    valid = ~np.ma.getmaskarray(elev) & np.isfinite(elev.data)

    rr = np.arange(row0, row1)[:, None]
    cc = np.arange(col0, col1)[None, :]
    t = dem.transform
    cx = t.a * (cc + .5) + t.b * (rr + .5) + t.c
    cy = t.d * (cc + .5) + t.e * (rr + .5) + t.f
    circle = (cx-x)**2 + (cy-y)**2 <= RADIUS_M**2
    valid_dem_circle = valid & circle
    result["valid_dem_pixels"] = int(valid_dem_circle.sum())
    if result["valid_dem_pixels"]:
        result["mean_elevation_m"] = float(elev.data[valid_dem_circle].mean())

    px, py = map(abs, dem.res)
    if px <= 0 or py <= 0:
        raise ValueError(f"Invalid DEM resolution: {dem.res}")

    # Match slope_analysis.py: finite-difference gradient with meter spacing, then atan magnitude in degrees.
    filled = elev.filled(0)
    grad_y, grad_x = np.gradient(filled, py, px)
    slope = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))
    slope_valid = binary_erosion(valid, structure=np.ones((3, 3), dtype=bool), border_value=0)

    # Match roughness_analysis.py / landing_suitability.py: 5x5 local elevation standard deviation.
    valid_float = valid.astype("float32")
    data = filled.astype("float32")
    count = uniform_filter(valid_float, size=ROUGHNESS_WINDOW, mode="constant")
    safe = np.maximum(count, 1e-10)
    mean = uniform_filter(data, size=ROUGHNESS_WINDOW, mode="constant") / safe
    mean_sq = uniform_filter(data**2, size=ROUGHNESS_WINDOW, mode="constant") / safe
    roughness = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    rough_valid = binary_erosion(valid, structure=np.ones((ROUGHNESS_WINDOW, ROUGHNESS_WINDOW), dtype=bool), border_value=0)

    valid_analysis = slope_valid & rough_valid & circle
    n = int(valid_analysis.sum())
    result["valid_analysis_pixels"] = n
    if not n:
        result["status"] = "NO_VALID_TERRAIN_DATA_IN_CIRCLE"
        return result

    slopes = slope[valid_analysis]
    rough = roughness[valid_analysis]
    result["mean_slope_deg"] = float(slopes.mean())
    result["max_slope_deg"] = float(slopes.max())
    result["mean_roughness_m"] = float(rough.mean())
    result["max_roughness_m"] = float(rough.max())

    # Use unchanged criteria from landing_suitability.py.
    more = (slope <= 5) & (roughness <= 2)
    caution = (slope <= 15) & (roughness <= 5)
    counts = {
        1: int(np.count_nonzero(valid_analysis & more)),
        2: int(np.count_nonzero(valid_analysis & caution & ~more)),
        3: int(np.count_nonzero(valid_analysis & ~caution)),
    }
    result["more_suitable_pct"] = 100 * counts[1] / n
    result["caution_pct"] = 100 * counts[2] / n
    result["unsuitable_pct"] = 100 * counts[3] / n

    if not valid[row-row0, col-col0]:
        result["status"] = "CENTER_NODATA_NEIGHBORHOOD_ASSESSED"
    elif result["valid_dem_pixels"] > n:
        result["status"] = "ASSESSED_WITH_NODATA_OR_EDGE_CELLS_EXCLUDED"
    return result


def write_csv(results):
    columns = ["site_names", "longitude", "latitude", "projected_x_m", "projected_y_m", "row", "col", "status", "radius_m", "valid_dem_pixels", "valid_analysis_pixels", "mean_slope_deg", "max_slope_deg", "mean_roughness_m", "max_roughness_m", "mean_elevation_m", "more_suitable_pct", "caution_pct", "unsuitable_pct"]
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


def create_map(dem, sites, results):
    map_w, map_h, panel_w = 1080, 680, 390
    inside_sites = [s for s in sites if s.get("in_bounds")]
    if inside_sites:
        extents = []
        for s in inside_sites:
            cr = RADIUS_M / abs(dem.res[0]) + 80
            rr = RADIUS_M / abs(dem.res[1]) + 80
            extents.append((s["col_f"]-cr, s["row_f"]-rr, s["col_f"]+cr, s["row_f"]+rr))
        c0 = max(0, math.floor(min(e[0] for e in extents)))
        r0 = max(0, math.floor(min(e[1] for e in extents)))
        c1 = min(dem.width, math.ceil(max(e[2] for e in extents)))
        r1 = min(dem.height, math.ceil(max(e[3] for e in extents)))
        tile = dem.read(1, window=Window(c0, r0, c1-c0, r1-r0), masked=True).astype("float32")
        mask = ~np.ma.getmaskarray(tile) & np.isfinite(tile.data)
        vals = tile.data[mask]
        lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0., 1.)
        if hi <= lo:
            hi = lo + 1.
        colors = matplotlib.colormaps["terrain"](np.clip((tile.filled(lo)-lo)/(hi-lo), 0, 1), bytes=True)[:, :, :3]
        colors[~mask] = (255, 255, 255)
        background = Image.fromarray(colors, mode="RGB")
        scale = min(map_w/background.width, map_h/background.height)
        bw, bh = max(1, round(background.width*scale)), max(1, round(background.height*scale))
        background = background.resize((bw, bh), Image.Resampling.BILINEAR)
        left, top = 20+(map_w-bw)//2, 50+(map_h-bh)//2
    else:
        c0 = r0 = 0
        background = Image.new("RGB", (map_w, map_h), (238, 238, 238))
        scale, bw, bh, left, top = 1., map_w, map_h, 20, 50

    canvas = Image.new("RGB", (map_w+panel_w+40, 790), "white")
    canvas.paste(background, (left, top))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((20, 18), "Jezero Crater - Scientific Site Terrain Assessment", fill="black", font=font)

    for i, (site, result) in enumerate(zip(sites, results), 1):
        if not site.get("in_bounds"):
            continue
        color = SITE_COLORS[(i-1) % len(SITE_COLORS)]
        x = left + (site["col_f"]-c0)*scale
        y = top + (site["row_f"]-r0)*scale
        rx = RADIUS_M/abs(dem.res[0])*scale
        ry = RADIUS_M/abs(dem.res[1])*scale
        box = (x-rx, y-ry, x+rx, y+ry)
        draw.ellipse(box, outline="black", width=5)
        draw.ellipse(box, outline=color, width=3)
        draw.ellipse((x-12, y-12, x+12, y+12), fill="white", outline=color, width=3)
        draw.text((x-3, y-5), str(i), fill="black", font=font)

    panel_x = map_w + 42
    draw.line((panel_x-12, 52, panel_x-12, 720), fill=(190, 190, 190))
    draw.text((panel_x, 58), "Unique coordinates / 500 m", fill="black", font=font)
    y = 84
    for i, (site, result) in enumerate(zip(sites, results), 1):
        color = SITE_COLORS[(i-1) % len(SITE_COLORS)]
        draw.ellipse((panel_x, y, panel_x+14, y+14), fill=color, outline="black")
        draw.text((panel_x+22, y+2), f"{i}. {result['site_names']}", fill="black", font=font)
        y += 19
        if result["mean_slope_deg"] is None:
            detail = result["status"]
        else:
            detail = f"Slope mean/max {result['mean_slope_deg']:.1f}/{result['max_slope_deg']:.1f} deg; roughness {result['mean_roughness_m']:.1f}/{result['max_roughness_m']:.1f} m"
        for line in wrap(detail, 50):
            draw.text((panel_x+22, y), line, fill=(70, 70, 70), font=font)
            y += 13
        y += 8

    draw.line((20, 750, 42, 750), fill=(0, 180, 190), width=3)
    draw.text((48, 744), "500 m assessment radius", fill="black", font=font)
    draw.text((265, 744), "Preliminary terrain assessment; not a safety guarantee.", fill="black", font=font)
    canvas.save(PNG_PATH, format="PNG")


def main():
    print("="*72)
    print("SCIENTIFIC SITE TERRAIN ASSESSMENT")
    print("="*72)
    for path in (DEM_PATH, SITES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
    features, sites, invalid = unique_sites_from_geojson(SITES_PATH)
    print(f"[STEP 1] GeoJSON features: {len(features)}; unique valid coordinates: {len(sites)}; invalid: {invalid}")
    if not sites:
        raise ValueError("No valid Point sites found in GeoJSON.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("[STEP 2] Opening DEM and creating coordinate transformer...", flush=True)
    with rasterio.open(DEM_PATH) as dem:
        if dem.crs is None:
            raise ValueError("DEM has no CRS.")
        print(f"DEM: {dem.width} x {dem.height}; resolution {dem.res} m")
        print(f"DEM CRS: {dem.crs}")
        print("CRS assumption: GeoJSON lon/lat are Mars 2000 sphere (radius 3,396,190 m), inherited from validate_science_sites.py; not confirmed scientifically.", flush=True)
        transformer = Transformer.from_crs(SOURCE_CRS, dem.crs, always_xy=True)

        for i, site in enumerate(sites, 1):
            names = "; ".join(site["names"])
            print(f"\n[STEP 3.{i}] {names}", flush=True)
            try:
                result = analyze_site(dem, site, transformer)
            except Exception as error:
                print(f"[ERROR] {names}: {error}", flush=True)
                traceback.print_exc()
                result = {"site_names": names, "longitude": site["longitude"], "latitude": site["latitude"], "projected_x_m": site.get("x"), "projected_y_m": site.get("y"), "row": None, "col": None, "status": f"ERROR: {error}", "radius_m": RADIUS_M, "valid_dem_pixels": 0, "valid_analysis_pixels": 0, "mean_slope_deg": None, "max_slope_deg": None, "mean_roughness_m": None, "max_roughness_m": None, "mean_elevation_m": None, "more_suitable_pct": None, "caution_pct": None, "unsuitable_pct": None}
                site["in_bounds"] = False
            results.append(result)
            print(f"Status: {result['status']}; projected X/Y: {result['projected_x_m']} / {result['projected_y_m']} m; row/col: {result['row']} / {result['col']}")
            if result["mean_slope_deg"] is None:
                print("Terrain metrics unavailable.")
            else:
                print(f"Slope mean/max: {result['mean_slope_deg']:.3f} / {result['max_slope_deg']:.3f} degrees")
                print(f"Roughness mean/max: {result['mean_roughness_m']:.3f} / {result['max_roughness_m']:.3f} m")
                print(f"Mean elevation: {result['mean_elevation_m']:.3f} m; analysis pixels: {result['valid_analysis_pixels']:,}")
                print(f"More Suitable: {result['more_suitable_pct']:.2f}% | Caution: {result['caution_pct']:.2f}% | Unsuitable: {result['unsuitable_pct']:.2f}%")

        print("\n[STEP 4] Writing CSV...", flush=True)
        write_csv(results)
        print("[STEP 5] Rendering site overview map...", flush=True)
        create_map(dem, sites, results)

    if not CSV_PATH.is_file() or CSV_PATH.stat().st_size == 0:
        raise RuntimeError(f"CSV missing or empty: {CSV_PATH}")
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(sites):
        raise RuntimeError(f"CSV contains {len(rows)} rows; expected {len(sites)} unique sites.")
    if not PNG_PATH.is_file() or PNG_PATH.stat().st_size == 0:
        raise RuntimeError(f"PNG missing or empty: {PNG_PATH}")
    with Image.open(PNG_PATH) as image:
        image.verify()
    print("\n[STEP 6] Outputs verified")
    print(f"CSV: {CSV_PATH} ({CSV_PATH.stat().st_size:,} bytes; {len(rows)} site rows)")
    print(f"PNG: {PNG_PATH} ({PNG_PATH.stat().st_size:,} bytes)")
    print("Assessment is preliminary; terrain metrics do not guarantee landing safety. CRS remains an assumption.")
    print("SCIENTIFIC SITE TERRAIN ASSESSMENT COMPLETED.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL ERROR: Full traceback follows:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1) from None
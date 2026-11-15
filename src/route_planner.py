"""Prototype A* route planner across Jezero DEM terrain."""

import argparse
import heapq
import json
import math
import os
import sys
import traceback
from pathlib import Path

# CRS assumption follows validate_science_sites.py; it is not independently confirmed.
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
OUT_DIR = PROJECT_DIR / "outputs"
DEM_PATH = RAW_DIR / "M20_JezeroCrater_CTXDEM_20m (1).tif"
SITES_PATH = RAW_DIR / "jezero_10_scientific_sites.geojson"
GEOJSON_PATH = OUT_DIR / "route_planner_routes.geojson"
PNG_PATH = OUT_DIR / "route_planner_map.png"
SOURCE_CRS = CRS.from_proj4("+proj=longlat +a=3396190 +b=3396190 +no_defs +type=crs")
ROUGHNESS_WINDOW = 5
CLASS_NAMES = {1: "More Suitable", 2: "Caution", 3: "Unsuitable"}
CLASS_COLORS = {0: (235, 235, 235), 1: (50, 180, 90), 2: (245, 200, 60), 3: (210, 65, 65)}


def arguments():
    parser = argparse.ArgumentParser(description="Plan two prototype routes using A* on a local DEM corridor.")
    parser.add_argument("--start-row", type=int, help="Landing pixel row (0-based); use with --start-col.")
    parser.add_argument("--start-col", type=int, help="Landing pixel column (0-based); use with --start-row.")
    parser.add_argument("--start-lon", type=float, help="Landing Mars longitude in degrees; use with --start-lat.")
    parser.add_argument("--start-lat", type=float, help="Landing Mars latitude in degrees; use with --start-lon.")
    parser.add_argument("--target-site", required=True, help="Scientific site name from the GeoJSON (aliases accepted).")
    parser.add_argument("--corridor-buffer-m", type=float, default=3000.0, help="Search corridor half-width (default: 3000 m). Increase if no route is found.")
    parser.add_argument("--slope-weight", type=float, default=2.0, help="Slope contribution to cell cost (default: 2).")
    parser.add_argument("--roughness-weight", type=float, default=2.0, help="Roughness contribution to cell cost (default: 2).")
    parser.add_argument("--caution-penalty", type=float, default=4.0, help="Extra Caution-cell cost multiplier (default: 4).")
    args = parser.parse_args()

    row_mode = args.start_row is not None and args.start_col is not None
    lonlat_mode = args.start_lon is not None and args.start_lat is not None
    partial_pixel = (args.start_row is None) != (args.start_col is None)
    partial_lonlat = (args.start_lon is None) != (args.start_lat is None)
    if partial_pixel:
        parser.error("Provide both --start-row and --start-col.")
    if partial_lonlat:
        parser.error("Provide both --start-lon and --start-lat.")
    if row_mode == lonlat_mode:
        parser.error("Provide exactly one start method: --start-row/--start-col OR --start-lon/--start-lat.")
    if lonlat_mode and not (
        math.isfinite(args.start_lon)
        and math.isfinite(args.start_lat)
        and -180 <= args.start_lon <= 180
        and -90 <= args.start_lat <= 90
    ):
        parser.error("Mars start coordinates must be finite, with longitude in [-180, 180] and latitude in [-90, 90].")
    return args


def get_unique_sites(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped = {}
    for feature in data.get("features", []):
        if not isinstance(feature, dict):
            continue
        geom = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        coords = geom.get("coordinates", []) if isinstance(geom, dict) else []
        if not isinstance(geom, dict) or geom.get("type") != "Point" or len(coords) < 2:
            continue
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        key = (lon, lat)
        grouped.setdefault(key, {"longitude": lon, "latitude": lat, "names": []})["names"].append(str(props.get("site_name") or "Unknown site"))
    return list(grouped.values())


def to_pixel(transform, x, y):
    return (~transform) * (x, y)  # column, row


def terrain_layers(elevation, pixel_x, pixel_y):
    """Use the project slope, roughness, and landing-classification methodology."""
    valid = ~np.ma.getmaskarray(elevation) & np.isfinite(elevation.data)
    filled = elevation.filled(0)
    gy, gx = np.gradient(filled, pixel_y, pixel_x)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope_valid = binary_erosion(valid, structure=np.ones((3, 3), dtype=bool), border_value=0)

    vf = valid.astype("float32")
    data = filled.astype("float32")
    count = uniform_filter(vf, size=ROUGHNESS_WINDOW, mode="constant")
    safe = np.maximum(count, 1e-10)
    mean = uniform_filter(data, size=ROUGHNESS_WINDOW, mode="constant") / safe
    mean_square = uniform_filter(data**2, size=ROUGHNESS_WINDOW, mode="constant") / safe
    roughness = np.sqrt(np.maximum(mean_square - mean**2, 0))
    rough_valid = binary_erosion(valid, structure=np.ones((ROUGHNESS_WINDOW, ROUGHNESS_WINDOW), dtype=bool), border_value=0)
    combined = slope_valid & rough_valid

    # Exact thresholds from landing_suitability.py.
    more = (slope <= 5) & (roughness <= 2)
    caution = (slope <= 15) & (roughness <= 5)
    classes = np.zeros(elevation.shape, dtype=np.uint8)
    classes[combined & more] = 1
    classes[combined & caution & ~more] = 2
    classes[combined & ~caution] = 3
    del filled, gx, gy, vf, data, count, safe, mean, mean_square, slope_valid, rough_valid
    return slope, roughness, combined, classes


def make_corridor(transform, shape, offset, start_xy, goal_xy, buffer_m):
    row0, col0 = offset
    h, w = shape
    rows = np.arange(row0, row0+h)[:, None] + 0.5
    cols = np.arange(col0, col0+w)[None, :] + 0.5
    t = transform
    xx = t.a*cols + t.b*rows + t.c
    yy = t.d*cols + t.e*rows + t.f
    ax, ay = start_xy
    bx, by = goal_xy
    vx, vy = bx-ax, by-ay
    denominator = vx*vx + vy*vy
    if denominator == 0:
        return np.ones(shape, dtype=bool)
    fraction = np.clip(((xx-ax)*vx + (yy-ay)*vy) / denominator, 0, 1)
    dx = xx - (ax + fraction*vx)
    dy = yy - (ay + fraction*vy)
    return dx*dx + dy*dy <= buffer_m*buffer_m


def astar(passable, cell_weight, transform, start, goal):
    """Eight-neighbor A* with metric step lengths and no diagonal corner cutting."""
    h, w = passable.shape
    if not passable[start] or not passable[goal]:
        return None
    if start == goal:
        return [start]
    a, b, d, e = transform.a, transform.b, transform.d, transform.e

    def metric_delta(dr, dc):
        dx, dy = a*dc + b*dr, d*dc + e*dr
        return math.hypot(dx, dy)

    def heuristic(r, c):
        dr, dc = goal[0]-r, goal[1]-c
        return metric_delta(dr, dc)

    costs = np.full((h, w), np.inf, dtype=np.float64)
    parents = np.full((h, w), -1, dtype=np.int64)
    closed = np.zeros((h, w), dtype=bool)
    start_id, goal_id = start[0]*w+start[1], goal[0]*w+goal[1]
    costs[start] = 0.0
    queue = [(heuristic(*start), 0.0, start_id)]
    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        _, current_cost, node = heapq.heappop(queue)
        r, c = divmod(node, w)
        if closed[r, c] or current_cost > costs[r, c]:
            continue
        if node == goal_id:
            path = []
            while node != -1:
                rr, cc = divmod(node, w)
                path.append((rr, cc))
                if node == start_id:
                    break
                node = int(parents[rr, cc])
            path.reverse()
            return path if path and path[0] == start and path[-1] == goal else None
        closed[r, c] = True

        for dr, dc in moves:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not closed[nr, nc]):
                continue
            if dr and dc and (not passable[r+dr, c] or not passable[r, c+dc]):
                continue
            step = metric_delta(dr, dc)
            candidate = current_cost + step * (cell_weight[r, c] + cell_weight[nr, nc]) * 0.5
            if candidate < costs[nr, nc]:
                costs[nr, nc] = candidate
                parents[nr, nc] = r*w+c
                heapq.heappush(queue, (candidate+heuristic(nr, nc), candidate, nr*w+nc))
    return None


def metrics(path, offset, slope, roughness, classes, transform, cell_weight):
    row0, col0 = offset
    global_path = [(r+row0, c+col0) for r, c in path]
    route_slope = np.array([slope[r, c] for r, c in path], dtype=float)
    route_rough = np.array([roughness[r, c] for r, c in path], dtype=float)
    distance = 0.0
    cost = 0.0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        dr, dc = r2-r1, c2-c1
        dx, dy = transform.a*dc + transform.b*dr, transform.d*dc + transform.e*dr
        step = math.hypot(dx, dy)
        distance += step
        cost += step * (cell_weight[r1, c1]+cell_weight[r2, c2])*0.5
    return {
        "path_local": path,
        "path_global": global_path,
        "distance_m": distance,
        "mean_slope_deg": float(route_slope.mean()),
        "mean_roughness_m": float(route_rough.mean()),
        "weighted_cost_m": cost,
        "caution_cells": int(sum(classes[r, c] == 2 for r, c in path)),
        "unsuitable_cells": 0,
    }


def build_geojson_document(result):
    """Build the existing CLI GeoJSON representation from in-memory route results."""
    features = []
    start = result["start"]
    target = result["target"]
    for route in result["routes"]:
        properties = {key: value for key, value in route.items() if key != "coordinates"}
        properties.update({
            "target_site_names": target["site_names"],
            "start_row": start["row"], "start_col": start["col"],
            "target_row": target["row"], "target_col": target["col"],
            "coordinate_order": "Mars longitude, latitude (assumed Mars 2000 sphere)",
        })
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": route["coordinates"]},
            "properties": properties,
        })
    return {
        "type": "FeatureCollection",
        "name": "Jezero A* prototype routes",
        "coordinate_reference_assumption": result["assumptions"]["note"],
        "dem_crs": result["dem"]["crs"],
        "alternative_available": result["alternative_available"],
        "alternative_message": result["alternative_message"],
        "features": features,
    }

def render_map(classes, offset, start_local, goal_local, routes, path):
    row0, col0 = offset
    rgb = np.full((*classes.shape, 3), (235, 235, 235), dtype=np.uint8)
    for code, color in CLASS_COLORS.items():
        if code:
            rgb[classes == code] = color
    image = Image.fromarray(rgb, mode="RGB")
    max_w, max_h = 1250, 800
    scale = min(max_w/image.width, max_h/image.height)
    width, height = max(1, round(image.width*scale)), max(1, round(image.height*scale))
    image = image.resize((width, height), Image.Resampling.NEAREST)
    top, left = 55, 20
    panel_w = 300
    canvas = Image.new("RGB", (width+panel_w+50, height+120), "white")
    canvas.paste(image, (left, top))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((20, 18), "A* routes over slope/roughness suitability", fill="black", font=font)

    def xy(local):
        return (left+(local[1]+0.5)*scale, top+(local[0]+0.5)*scale)

    colors = [(0, 190, 240), (235, 80, 35)]
    for index, route in enumerate(routes):
        points = [xy(point) for point in route["path_local"]]
        if len(points) > 1:
            draw.line(points, fill=(0, 0, 0), width=7)
            draw.line(points, fill=colors[index], width=4)
    for point, color, label in ((start_local, (20, 140, 40), "S"), (goal_local, (130, 30, 170), "T")):
        x, y = xy(point)
        draw.ellipse((x-9, y-9, x+9, y+9), fill="white", outline=color, width=4)
        draw.text((x-3, y-5), label, fill="black", font=font)

    panel_x = width+42
    draw.line((panel_x-14, 48, panel_x-14, height+50), fill=(190, 190, 190))
    draw.text((panel_x, 55), "Route comparison", fill="black", font=font)
    y = 82
    for i, route in enumerate(routes):
        draw.line((panel_x, y+4, panel_x+25, y+4), fill=colors[i], width=4)
        draw.text((panel_x+34, y), route["route_id"], fill="black", font=font)
        y += 18
        for line in [
            f"Distance: {route['distance_m']/1000:.2f} km",
            f"Mean slope: {route['mean_slope_deg']:.2f} deg",
            f"Mean roughness: {route['mean_roughness_m']:.2f} m",
            f"Weighted cost: {route['weighted_cost_m']/1000:.2f} km-eq",
            f"Caution cells: {route['caution_cells']}",
        ]:
            draw.text((panel_x+34, y), line, fill=(70, 70, 70), font=font)
            y += 15
        y += 12
    legend_y = height+72
    for code, name in CLASS_NAMES.items():
        color = CLASS_COLORS[code]
        draw.rectangle((20, legend_y, 34, legend_y+14), fill=color, outline="black")
        draw.text((40, legend_y+3), name, fill="black", font=font)
        legend_y += 18
    draw.text((340, height+83), "Prototype planning only; not a guarantee of safe traversal.", fill="black", font=font)
    canvas.save(path, format="PNG")


def calculate_routes(
    *,
    target_site,
    start_row=None,
    start_col=None,
    start_lon=None,
    start_lat=None,
    corridor_buffer_m=3000.0,
    slope_weight=2.0,
    roughness_weight=2.0,
    caution_penalty=4.0,
):
    """Calculate the preferred A* route and an optional distinct alternative."""
    row_mode = start_row is not None and start_col is not None
    lonlat_mode = start_lon is not None and start_lat is not None
    if (start_row is None) != (start_col is None):
        raise ValueError("Provide both start_row and start_col.")
    if (start_lon is None) != (start_lat is None):
        raise ValueError("Provide both start_lon and start_lat.")
    if row_mode == lonlat_mode:
        raise ValueError("Provide exactly one start method: row/col or longitude/latitude.")
    if lonlat_mode and not (
        isinstance(start_lon, (int, float)) and not isinstance(start_lon, bool)
        and isinstance(start_lat, (int, float)) and not isinstance(start_lat, bool)
        and math.isfinite(start_lon) and math.isfinite(start_lat)
        and -180 <= start_lon <= 180 and -90 <= start_lat <= 90
    ):
        raise ValueError("Mars start longitude/latitude must be finite and within [-180, 180] / [-90, 90].")
    if not isinstance(target_site, str) or not target_site.strip():
        raise ValueError("target_site must be a non-empty site name.")
    if corridor_buffer_m <= 0 or not math.isfinite(corridor_buffer_m):
        raise ValueError("corridor_buffer_m must be finite and greater than zero.")
    if any(not math.isfinite(v) or v < 0 for v in (slope_weight, roughness_weight, caution_penalty)):
        raise ValueError("Cost weights and caution penalty must be finite and non-negative.")
    for required_path in (DEM_PATH, SITES_PATH):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    sites = get_unique_sites(SITES_PATH)
    requested = target_site.strip().casefold()
    target_group = next(
        (site for site in sites if any(name.casefold() == requested for name in site["names"])),
        None,
    )
    if target_group is None:
        available = [" / ".join(site["names"]) for site in sites]
        raise ValueError(f"Unknown target site {target_site!r}. Available unique sites: {available}")
    target_names = "; ".join(target_group["names"])

    with rasterio.open(DEM_PATH) as dem:
        if dem.crs is None:
            raise ValueError("DEM has no CRS.")
        transformer = Transformer.from_crs(SOURCE_CRS, dem.crs, always_xy=True)
        inverse_transformer = Transformer.from_crs(dem.crs, SOURCE_CRS, always_xy=True)

        if row_mode:
            if not isinstance(start_row, int) or isinstance(start_row, bool) or not isinstance(start_col, int) or isinstance(start_col, bool):
                raise ValueError("start_row and start_col must be integer pixel indices.")
            start = (start_row, start_col)
            if not (0 <= start[0] < dem.height and 0 <= start[1] < dem.width):
                raise ValueError(f"Start pixel {start} is outside DEM bounds.")
            start_x, start_y = dem.transform * (start[1]+0.5, start[0]+0.5)
            input_mode = "row_col"
            input_longitude = input_latitude = None
        else:
            try:
                start_x_input, start_y_input = transformer.transform(start_lon, start_lat, errcheck=True)
            except Exception as error:
                raise ValueError(f"Could not transform Mars start coordinates ({start_lon}, {start_lat}) to DEM CRS: {error}") from error
            if not (math.isfinite(start_x_input) and math.isfinite(start_y_input)):
                raise ValueError("Transformed start coordinates are not finite.")
            start_col_f, start_row_f = to_pixel(dem.transform, start_x_input, start_y_input)
            start = (math.floor(start_row_f), math.floor(start_col_f))
            if not (0 <= start[0] < dem.height and 0 <= start[1] < dem.width):
                raise ValueError(f"Mars start coordinates ({start_lon}, {start_lat}) are outside DEM bounds (converted to row={start[0]}, col={start[1]}).")
            input_mode = "longitude_latitude"
            input_longitude, input_latitude = float(start_lon), float(start_lat)
            start_x, start_y = dem.transform * (start[1]+0.5, start[0]+0.5)

        start_sample = dem.read(1, window=Window(start[1], start[0], 1, 1), masked=True)[0, 0]
        if bool(getattr(start_sample, "mask", False)) or not math.isfinite(float(start_sample)):
            raise ValueError(f"Start pixel row={start[0]}, col={start[1]} contains NoData or a non-finite elevation.")

        tx, ty = transformer.transform(target_group["longitude"], target_group["latitude"], errcheck=True)
        target_col_f, target_row_f = to_pixel(dem.transform, tx, ty)
        target = (math.floor(target_row_f), math.floor(target_col_f))
        if not (0 <= target[0] < dem.height and 0 <= target[1] < dem.width):
            raise ValueError(f"Target {target_names} is outside DEM at pixel {target}.")
        if start == target:
            raise ValueError("Start and target map to the same DEM cell.")

        px, py = map(abs, dem.res)
        pad_cols = math.ceil(corridor_buffer_m/px)+3
        pad_rows = math.ceil(corridor_buffer_m/py)+3
        row0 = max(0, min(start[0], target[0])-pad_rows)
        row1 = min(dem.height, max(start[0], target[0])+pad_rows+1)
        col0 = max(0, min(start[1], target[1])-pad_cols)
        col1 = min(dem.width, max(start[1], target[1])+pad_cols+1)
        elevation = dem.read(1, window=Window(col0, row0, col1-col0, row1-row0), masked=True).astype("float32")
        slope, roughness, combined_valid, classes = terrain_layers(elevation, px, py)
        local_start = (start[0]-row0, start[1]-col0)
        local_goal = (target[0]-row0, target[1]-col0)
        if classes[local_start] == 0 or classes[local_goal] == 0:
            raise ValueError("Start or target has no valid slope/roughness neighborhood (possibly NODATA-adjacent).")
        if classes[local_start] == 3 or classes[local_goal] == 3:
            raise ValueError("Start or target cell is classified Unsuitable by the existing thresholds.")

        corridor = make_corridor(
            dem.transform, elevation.shape, (row0, col0),
            (start_x, start_y), (tx, ty), corridor_buffer_m,
        )
        passable = combined_valid & (classes != 3) & corridor
        passable[local_start] = True
        passable[local_goal] = True
        cell_weight = (
            1.0 + slope_weight*(slope/15.0)
            + roughness_weight*(roughness/5.0)
            + (classes == 2)*caution_penalty
        ).astype(np.float32)

        path1 = astar(passable, cell_weight, dem.transform, local_start, local_goal)
        if path1 is None:
            raise RuntimeError("No route found in this corridor. Increase corridor_buffer_m and retry.")
        alternate_allowed = passable.copy()
        for point in path1[4:-4]:
            alternate_allowed[point] = False
        path2 = astar(alternate_allowed, cell_weight, dem.transform, local_start, local_goal)
        if path2 is None or path2 == path1:
            alternate_weight = cell_weight.copy()
            for point in path1[4:-4]:
                alternate_weight[point] += 25.0
            path2 = astar(passable, alternate_weight, dem.transform, local_start, local_goal)
        alternative_available = path2 is not None and path2 != path1
        if not alternative_available:
            path2 = None
        alternative_message = (
            None if alternative_available
            else "A distinct second route could not be found inside the selected corridor."
        )

        resolved_lon, resolved_lat = inverse_transformer.transform(start_x, start_y)
        target_record = {
            "site_names": target_names,
            "longitude": float(target_group["longitude"]),
            "latitude": float(target_group["latitude"]),
            "projected_x_m": float(tx), "projected_y_m": float(ty),
            "row": int(target[0]), "col": int(target[1]),
        }
        start_record = {
            "input_method": input_mode,
            "input_longitude": input_longitude,
            "input_latitude": input_latitude,
            "longitude": float(resolved_lon), "latitude": float(resolved_lat),
            "projected_x_m": float(start_x), "projected_y_m": float(start_y),
            "row": int(start[0]), "col": int(start[1]),
        }
        public_routes = []
        render_routes = []
        route_paths = [("Route 1 (lowest weighted cost)", path1)]
        if path2 is not None:
            route_paths.append(("Route 2 (alternative)", path2))
        for route_id, path in route_paths:
            route_metrics = metrics(path, (row0, col0), slope, roughness, classes, dem.transform, cell_weight)
            coordinates = []
            for row, col in route_metrics["path_global"]:
                x, y = dem.transform * (col+0.5, row+0.5)
                lon, lat = inverse_transformer.transform(x, y)
                coordinates.append([float(lon), float(lat)])
            public_routes.append({
                "route_id": route_id,
                "coordinates": coordinates,
                "distance_m": float(route_metrics["distance_m"]),
                "mean_slope_deg": float(route_metrics["mean_slope_deg"]),
                "mean_roughness_m": float(route_metrics["mean_roughness_m"]),
                "weighted_cost_m": float(route_metrics["weighted_cost_m"]),
                "caution_cells": int(route_metrics["caution_cells"]),
                "unsuitable_cells": int(route_metrics["unsuitable_cells"]),
                "slope_weight": float(slope_weight),
                "roughness_weight": float(roughness_weight),
                "caution_penalty": float(caution_penalty),
                "corridor_buffer_m": float(corridor_buffer_m),
            })
            render_routes.append({"route_id": route_id, "path_local": path, **route_metrics})

        return {
            "success": True,
            "status": "success",
            "start": start_record,
            "target": target_record,
            "routes": public_routes,
            "alternative_available": alternative_available,
            "alternative_message": alternative_message,
            "assumptions": {
                "source_crs": "Mars 2000 reference sphere longitude/latitude (radius 3,396,190 m)",
                "scientifically_confirmed": False,
                "note": "The GeoJSON coordinate interpretation follows validate_science_sites.py and still requires confirmation against the source data.",
            },
            "dem": {"name": DEM_PATH.name, "crs": dem.crs.to_string(), "resolution_m": [float(px), float(py)]},
            "_render_context": {
                "classes": classes,
                "offset": (row0, col0),
                "start_local": local_start,
                "goal_local": local_goal,
                "routes": render_routes,
            },
        }


def main():
    args = arguments()
    result = calculate_routes(
        target_site=args.target_site,
        start_row=args.start_row,
        start_col=args.start_col,
        start_lon=args.start_lon,
        start_lat=args.start_lat,
        corridor_buffer_m=args.corridor_buffer_m,
        slope_weight=args.slope_weight,
        roughness_weight=args.roughness_weight,
        caution_penalty=args.caution_penalty,
    )

    print("="*72)
    print("JEZERO MARS PATHFINDER - A* ROUTE PLANNER (PROTOTYPE)")
    print(f"Unique science-site coordinate groups in GeoJSON: {len(get_unique_sites(SITES_PATH))}")
    print(f"Target: {result['target']['site_names']}")
    print(f"Landing point method: {result['start']['input_method']}")
    if result["start"]["input_method"] == "longitude_latitude":
        print("CRS assumption: Mars 2000 sphere lon/lat; unconfirmed scientifically.")
    print(f"Resolved landing cell: row={result['start']['row']}, col={result['start']['col']}")
    print("[STEP 1] Local DEM corridor and existing terrain layers processed.")
    print(f"[STEP 2] Search window respects a {args.corridor_buffer_m:g} m corridor; existing suitability thresholds used.")
    if result["alternative_available"]:
        print("[STEP 3] A* preferred and distinct alternative routes calculated.")
    else:
        print(f"[STEP 3] Preferred route calculated; alternative unavailable: {result['alternative_message']}")
    print("\n===== ROUTE COMPARISON =====")
    for route in result["routes"]:
        print(route["route_id"])
        print(f"  Distance: {route['distance_m']:.1f} m")
        print(f"  Mean slope: {route['mean_slope_deg']:.3f} degrees")
        print(f"  Mean roughness: {route['mean_roughness_m']:.3f} m")
        print(f"  Weighted cost: {route['weighted_cost_m']:.1f} m-equivalent")
        print(f"  Caution cells: {route['caution_cells']}; Unsuitable cells: {route['unsuitable_cells']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geojson_document = build_geojson_document(result)
    GEOJSON_PATH.write_text(json.dumps(geojson_document, indent=2), encoding="utf-8")
    context = result["_render_context"]
    render_map(context["classes"], context["offset"], context["start_local"], context["goal_local"], context["routes"], PNG_PATH)

    if not GEOJSON_PATH.is_file() or GEOJSON_PATH.stat().st_size == 0:
        raise RuntimeError(f"GeoJSON output missing or empty: {GEOJSON_PATH}")
    saved_geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    expected_routes = len(result["routes"])
    if len(saved_geojson.get("features", [])) != expected_routes:
        raise RuntimeError(f"Expected {expected_routes} route feature(s) in GeoJSON output.")
    if not PNG_PATH.is_file() or PNG_PATH.stat().st_size == 0:
        raise RuntimeError(f"PNG output missing or empty: {PNG_PATH}")
    with Image.open(PNG_PATH) as image:
        image.verify()
    print("\n[STEP 4] Existing CLI outputs saved and verified")
    print(f"GeoJSON: {GEOJSON_PATH} ({GEOJSON_PATH.stat().st_size:,} bytes; {expected_routes} route(s))")
    print(f"PNG: {PNG_PATH} ({PNG_PATH.stat().st_size:,} bytes)")
    print("Prototype routes are DEM-based planning estimates and do not guarantee safe traversal.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL ERROR: Full traceback follows:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1) from None

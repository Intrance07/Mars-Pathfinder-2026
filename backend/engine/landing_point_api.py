"""In-memory longitude/latitude interface for landing-point assessment."""

import json
import math
from decimal import Decimal
from numbers import Real
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window

try:  # Support both `import src.landing_point_api` and a src/ sys.path import.
    from .landing_point_evaluator import evaluate_landing_point
    from .route_planner import DEM_PATH, SOURCE_CRS, to_pixel
except ImportError:
    from landing_point_evaluator import evaluate_landing_point
    from route_planner import DEM_PATH, SOURCE_CRS, to_pixel


CRS_ASSUMPTION = {
    "source_crs": "Mars 2000 Sphere longitude/latitude (radius 3,396,190 m)",
    "longitude_direction": "Positive East",
    "scientifically_confirmed": False,
    "note": (
        "Uses the same CRS assumption as route_planner.py. The original reference "
        "for user-provided Mars coordinates has not been scientifically confirmed."
    ),
}


def evaluate_landing_point_lonlat(longitude, latitude, radius_m=500.0):
    """Transform Mars longitude/latitude to a DEM pixel, then assess that point.

    The route planner's DEM path, geographic CRS assumption, Transformer axis
    order, and inverse-Affine pixel conversion are reused. No terrain metrics
    are recalculated here; those remain the responsibility of
    ``evaluate_landing_point``.
    """
    try:
        longitude = _validate_number(longitude, "longitude", -180.0, 180.0)
        latitude = _validate_number(latitude, "latitude", -90.0, 90.0)
        radius_m = _validate_radius(radius_m)

        dem_path = Path(DEM_PATH)
        if not dem_path.is_file():
            raise FileNotFoundError(f"DEM GeoTIFF not found: {dem_path}")

        with rasterio.open(dem_path) as dem:
            if dem.crs is None:
                raise ValueError("DEM has no CRS defined.")
            try:
                target_crs = CRS.from_user_input(dem.crs)
            except Exception as error:
                raise ValueError(f"DEM CRS is invalid: {error}") from error
            if not target_crs.axis_info or not target_crs.to_wkt().strip():
                raise ValueError("DEM CRS is invalid.")

            try:
                transformer = Transformer.from_crs(
                    SOURCE_CRS, target_crs, always_xy=True
                )
                x_m, y_m = transformer.transform(longitude, latitude, errcheck=True)
            except Exception as error:
                raise ValueError(
                    f"Could not transform Mars coordinates to DEM CRS: {error}"
                ) from error
            if not (math.isfinite(x_m) and math.isfinite(y_m)):
                raise ValueError("Transformed DEM coordinates are not finite.")

            try:
                col_float, row_float = to_pixel(dem.transform, x_m, y_m)
            except Exception as error:
                raise ValueError(f"Could not apply DEM Affine Transform: {error}") from error
            if not (math.isfinite(col_float) and math.isfinite(row_float)):
                raise ValueError("Transformed pixel coordinates are not finite.")

            # This matches route_planner.calculate_routes: floor the inverse
            # affine pixel-edge coordinates to select the containing cell.
            row = math.floor(row_float)
            col = math.floor(col_float)
            if not (0 <= row < dem.height and 0 <= col < dem.width):
                raise ValueError(
                    f"Mars coordinates are outside DEM bounds "
                    f"(converted to row={row}, col={col})."
                )

            sample = dem.read(
                1, window=Window(col, row, 1, 1), masked=True
            )[0, 0]
            if np.ma.is_masked(sample) or not math.isfinite(float(sample)):
                raise ValueError(
                    f"DEM pixel row={row}, col={col} contains NoData or a non-finite elevation."
                )

            dem_info = {
                "name": dem_path.name,
                "crs": target_crs.to_string(),
                "resolution_m": [float(abs(value)) for value in dem.res],
            }

        assessment = evaluate_landing_point(
            row, col, radius_m=radius_m, dem_path=dem_path
        )
        response = {
            "success": True,
            "status": "success",
            "input": {"longitude": longitude, "latitude": latitude},
            "pixel": {
                "row": int(row),
                "col": int(col),
                "projected_x_m": float(x_m),
                "projected_y_m": float(y_m),
            },
            "assessment": assessment,
            "assumptions": CRS_ASSUMPTION,
            "dem": dem_info,
        }
        return _assert_strict_json(_json_safe(response))
    except Exception as error:
        return _assert_strict_json(_json_safe({
            "success": False,
            "status": "error",
            "error": {"type": type(error).__name__, "message": str(error)},
            "input": {"longitude": longitude, "latitude": latitude},
            "assumptions": CRS_ASSUMPTION,
            "dem": {"name": Path(DEM_PATH).name},
        }))


def _validate_number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a numeric value, not {type(value).__name__}.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; NaN and Infinity are not allowed.")
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}] degrees.")
    return number


def _validate_radius(value):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError("radius_m must be a numeric value.")
    try:
        radius_m = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("radius_m must be a finite number.") from error
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and greater than zero.")
    return radius_m


def _json_safe(value):
    """Convert response contents to strict JSON primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (Real, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return f"<{type(value).__name__}>"


def _assert_strict_json(response):
    """Fail closed if an unexpected value escapes JSON normalization."""
    json.dumps(response, allow_nan=False)
    return response

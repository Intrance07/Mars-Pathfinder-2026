"""In-memory Python API for Mars Pathfinder route planning.

GeoJSON coordinates are interpreted as longitude/latitude on the Mars 2000
reference sphere, following validate_science_sites.py. This assumption is not
scientifically confirmed and must be checked against the source data.
"""

import math
from decimal import Decimal
from numbers import Real

try:  # Supports both `import src.route_api` and adding src/ to sys.path.
    from .route_planner import SearchWindowLimitError, calculate_routes
except ImportError:
    from route_planner import SearchWindowLimitError, calculate_routes


CRS_ASSUMPTION = {
    "source_crs": "Mars 2000 reference sphere longitude/latitude (radius 3,396,190 m)",
    "scientifically_confirmed": False,
    "note": "The GeoJSON coordinate interpretation follows validate_science_sites.py and still requires confirmation against the source data.",
}


def plan_routes(start_lon, start_lat, target_site, diagnostics_callback=None):
    """Return one or two route plans as a JSON-serializable dictionary, without writing files.

    Args:
        start_lon: Mars longitude in degrees under the project's current CRS assumption.
        start_lat: Mars latitude in degrees under the project's current CRS assumption.
        target_site: A scientific-site or sample name present in the sites GeoJSON.

    Returns:
        A dictionary with success/status, resolved start and target information,
        CRS assumption, and two route records with coordinate arrays and metrics.
        On failure, returns success=False and an explanatory error string.
    """
    try:
        start_lon = _validate_coordinate(start_lon, "start_lon", -180, 180)
    except ValueError as error:
        return _error(str(error), start_lon, start_lat, target_site)
    try:
        start_lat = _validate_coordinate(start_lat, "start_lat", -90, 90)
    except ValueError as error:
        return _error(str(error), start_lon, start_lat, target_site)
    if not isinstance(target_site, str) or not target_site.strip():
        return _error("target_site must be a non-empty site name.", start_lon, start_lat, target_site)

    try:
        calculate_kwargs = {
            "start_lon": start_lon,
            "start_lat": start_lat,
            "target_site": target_site,
        }
        if diagnostics_callback is not None:
            calculate_kwargs["diagnostics_callback"] = diagnostics_callback
        result = calculate_routes(**calculate_kwargs)
    except SearchWindowLimitError as error:
        response = _error(
            str(error), start_lon, start_lat, target_site,
            error_type="SearchWindowLimitError",
        )
        response["search_window"] = error.search_window
        return response
    except Exception as error:
        return _error(str(error), start_lon, start_lat, target_site, error_type=type(error).__name__)

    # _render_context contains NumPy arrays and is consumed only by the CLI map renderer.
    return _json_safe({
        "success": True,
        "status": "success",
        "start": result["start"],
        "target": result["target"],
        "routes": result["routes"],
        "alternative_available": result.get("alternative_available", len(result["routes"]) > 1),
        "alternative_message": result.get("alternative_message"),
        "assumptions": result["assumptions"],
        "dem": result["dem"],
        "files_written": False,
    })


def _error(message, start_lon, start_lat, target_site, error_type="ValidationError"):
    return _json_safe({
        "success": False,
        "status": "error",
        "error": {"type": error_type, "message": str(message)},
        "start": {"longitude": start_lon, "latitude": start_lat},
        "target": {"site_name": target_site},
        "routes": [],
        "assumptions": CRS_ASSUMPTION,
        "files_written": False,
    })


def _validate_coordinate(value, name, minimum, maximum):
    """Return a finite float coordinate or raise a clear validation error."""
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a numeric value, not {type(value).__name__}.")

    try:
        coordinate = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be convertible to a finite number.") from error

    if not math.isfinite(coordinate):
        raise ValueError(f"{name} must be finite; NaN and Infinity are not allowed.")
    if not minimum <= coordinate <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}] degrees.")
    return coordinate


def _json_safe(value):
    """Convert response values to strict-JSON-compatible Python primitives.

    Non-finite numbers become null, and unsupported input objects are represented
    by their type name so an error response never echoes a non-serializable object.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (Real, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    # Support NumPy scalars/arrays if a future route-engine response includes them.
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if converted is not value:
                return _json_safe(converted)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            converted = value.item()
            if converted is not value:
                return _json_safe(converted)
        except (TypeError, ValueError):
            pass
    return f"<{type(value).__name__}>"

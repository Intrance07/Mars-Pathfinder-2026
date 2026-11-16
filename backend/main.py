import json
import logging
import os
import sys
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
SITES_PATH = BASE_DIR / "data" / "raw" / "jezero_10_scientific_sites.geojson"

sys.path.insert(0, str(BASE_DIR / "engine"))
from route_api import plan_routes  # noqa: E402
from landing_point_api import evaluate_landing_point_lonlat  # noqa: E402

app = FastAPI(title="Mars Pathfinder API")

# One Uvicorn worker is configured in render.Dockerfile. Keep the heavy
# synchronous A* operation single-flight without blocking health/sites routes.
_route_planning_lock = threading.Lock()
_route_logger = logging.getLogger("uvicorn.error")


@lru_cache(maxsize=1)
def _known_site_log_ids():
    """Map trusted site names to bounded identifiers for request logging."""
    try:
        with open(SITES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    site_ids = {}
    for index, feature in enumerate(data.get("features", []), start=1):
        properties = feature.get("properties") or {}
        name = properties.get("site_name")
        if isinstance(name, str) and name and len(name) <= 80 and name.isprintable():
            site_ids.setdefault(name.casefold(), f"site_{index:02d}")
    return site_ids


def _safe_target_site_log_id(target_site):
    """Never place arbitrary client text into service logs."""
    if (
        not isinstance(target_site, str)
        or not target_site
        or len(target_site) > 80
        or not target_site.isprintable()
    ):
        return "unrecognized"
    return _known_site_log_ids().get(target_site.strip().casefold(), "unrecognized")

_default_cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_configured_cors_origins = os.getenv("API_ALLOWED_ORIGINS")
_cors_origins = (
    [
        origin.strip().rstrip("/")
        for origin in _configured_cors_origins.split(",")
        if origin.strip()
    ]
    if _configured_cors_origins is not None
    else _default_cors_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    start_lon: float
    start_lat: float
    target_site: str


class LandingRequest(BaseModel):
    longitude: float
    latitude: float
    radius_m: float = 500.0


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sites")
def sites():
    with open(SITES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "sites": [
            {
                "properties": feat.get("properties", {}),
                "geometry": feat.get("geometry"),
            }
            for feat in data.get("features", [])
        ]
    }


@app.post("/api/plan-routes")
def plan(req: RouteRequest):
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    search_window = {}

    if not _route_planning_lock.acquire(blocking=False):
        duration_ms = (time.perf_counter() - started_at) * 1000
        _route_logger.info(
            "route_plan request_id=%s duration_ms=%.1f result=busy "
            "window_height=unknown window_width=unknown window_cells=unknown",
            request_id,
            duration_ms,
        )
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "10"},
            content={
                "success": False,
                "status": "error",
                "error": {
                    "type": "RoutePlannerBusyError",
                    "message": "Route planning is already in progress. Retry in a few seconds.",
                },
                "routes": [],
            },
        )

    outcome = "internal_error"
    target_site_log_id = "unrecognized"

    def record_search_window(info):
        search_window.update(info)
        _route_logger.info(
            "route_plan_window request_id=%s height=%d width=%d cells=%d limit=%d",
            request_id,
            info["height"],
            info["width"],
            info["cell_count"],
            info["max_cells"],
        )

    try:
        target_site_log_id = _safe_target_site_log_id(req.target_site)
        result = plan_routes(
            req.start_lon,
            req.start_lat,
            req.target_site,
            diagnostics_callback=record_search_window,
        )
        if result.get("success"):
            outcome = "success"
        else:
            outcome = f"error:{result.get('error', {}).get('type', 'UnknownError')}"

        if result.get("error", {}).get("type") == "SearchWindowLimitError":
            return JSONResponse(status_code=422, content=result)
        return result
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        _route_logger.info(
            "route_plan request_id=%s target_site_id=%s duration_ms=%.1f result=%s "
            "window_height=%s window_width=%s window_cells=%s",
            request_id,
            target_site_log_id,
            duration_ms,
            outcome,
            search_window.get("height", "unknown"),
            search_window.get("width", "unknown"),
            search_window.get("cell_count", "unknown"),
        )
        _route_planning_lock.release()


@app.post("/api/landing-point")
def landing_point(req: LandingRequest):
    return evaluate_landing_point_lonlat(req.longitude, req.latitude, req.radius_m)

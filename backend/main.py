import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
SITES_PATH = BASE_DIR / "data" / "raw" / "jezero_10_scientific_sites.geojson"

sys.path.insert(0, str(BASE_DIR / "engine"))
from route_api import plan_routes  # noqa: E402
from landing_point_api import evaluate_landing_point_lonlat  # noqa: E402

app = FastAPI(title="Mars Pathfinder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
    return plan_routes(req.start_lon, req.start_lat, req.target_site)


@app.post("/api/landing-point")
def landing_point(req: LandingRequest):
    return evaluate_landing_point_lonlat(req.longitude, req.latitude, req.radius_m)
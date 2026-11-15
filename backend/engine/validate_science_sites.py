
import json
import math
import os
import sys
import traceback
from pathlib import Path

# Make the GDAL support files bundled with Conda discoverable before Rasterio loads GDAL.
gdal_data = Path(sys.prefix) / "Library" / "share" / "gdal"
if (gdal_data / "gdalvrt.xsd").is_file():
    os.environ.setdefault("GDAL_DATA", str(gdal_data))

import rasterio
import pyproj
from pyproj import CRS, Transformer
from rasterio.windows import Window


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEM_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "M20_JezeroCrater_CTXDEM_20m (1).tif"
)

SITES_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "jezero_10_scientific_sites.geojson"
)


# ==========================================
# MARS COORDINATE SYSTEM
# ==========================================

# Assumption:
# GeoJSON coordinates are Mars longitude/latitude
# on the Mars 2000 reference sphere.
#
# Mars reference sphere radius = 3,396,190 meters.
#
# This assumption must be confirmed against the
# original source of the scientific coordinates.

MARS_SOURCE_CRS = CRS.from_proj4(
    "+proj=longlat "
    "+a=3396190 "
    "+b=3396190 "
    "+no_defs "
    "+type=crs"
)


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("=" * 65)
    print("JEZERO SCIENTIFIC SITES - SPATIAL VALIDATION")
    print("=" * 65)

    # --------------------------------------
    # STEP 1: CHECK FILES
    # --------------------------------------

    print("\n[STEP 1] CHECKING FILES")

    for path in [DEM_PATH, SITES_PATH]:

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        print(f"Found: {path.name}")

    # --------------------------------------
    # STEP 2: READ GEOJSON
    # --------------------------------------

    print("\n[STEP 2] READING GEOJSON")

    with open(SITES_PATH, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    features = geojson.get("features", [])

    print(f"Dataset: {geojson.get('name', 'Unnamed')}")
    print(f"Total records: {len(features)}")

    # --------------------------------------
    # STEP 3: READ DEM AND CRS
    # --------------------------------------

    print("\n[STEP 3] READING DEM INFORMATION")

    with rasterio.open(DEM_PATH) as dem:

        print(f"DEM CRS: {dem.crs}")
        print(f"Dimensions: {dem.width} x {dem.height}")
        print(f"Resolution: {dem.res}")
        print(f"Bounds: {dem.bounds}")

        # ----------------------------------
        # STEP 4: CREATE COORDINATE TRANSFORMER
        # ----------------------------------

        print("\n[STEP 4] COORDINATE TRANSFORMATION")

        if dem.crs is None:
            raise ValueError("DEM has no CRS defined.")

        print("Source CRS: Mars 2000 Sphere (assumed)")
        print("Target CRS: DEM CRS")

        transformer = Transformer.from_crs(
            MARS_SOURCE_CRS,
            dem.crs,
            always_xy=True
        )

        print("Coordinate transformer created successfully.")

        # ----------------------------------
        # STEP 5: VALIDATE SCIENTIFIC SITES
        # ----------------------------------

        print("\n[STEP 5] SCIENTIFIC SITE VALIDATION")

        unique_coordinates = set()

        inside_count = 0
        outside_count = 0
        invalid_count = 0
        nodata_count = 0

        for feature in features:

            if not isinstance(feature, dict):
                print("\nRecord: INVALID FEATURE (expected a GeoJSON object)")
                invalid_count += 1
                continue

            properties = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}
            if not isinstance(properties, dict):
                properties = {}

            site_name = properties.get(
                "site_name",
                "Unknown"
            )

            coordinates = geometry.get("coordinates", []) if isinstance(geometry, dict) else []

            if (
                not isinstance(geometry, dict)
                or geometry.get("type") != "Point"
                or not isinstance(coordinates, (list, tuple))
                or len(coordinates) < 2
            ):
                print(f"\n{site_name}: INVALID GEOMETRY")
                invalid_count += 1
                continue

            try:
                lon, lat = map(float, coordinates[:2])
            except (TypeError, ValueError, OverflowError):
                print(f"\n{site_name}: INVALID COORDINATES (not numeric)")
                invalid_count += 1
                continue

            if not (
                math.isfinite(lon)
                and math.isfinite(lat)
                and -180 <= lon <= 180
                and -90 <= lat <= 90
            ):
                print(f"\n{site_name}: INVALID COORDINATES")
                invalid_count += 1
                continue

            unique_coordinates.add((lon, lat))

            # Transform Mars longitude/latitude
            # into the DEM projected coordinate system.

            try:
                x, y = transformer.transform(lon, lat)
            except Exception as error:
                print(f"\n{site_name}: COORDINATE TRANSFORMATION FAILED: {error}")
                invalid_count += 1
                continue

            if not (math.isfinite(x) and math.isfinite(y)):
                print(f"\n{site_name}: TRANSFORMED COORDINATES ARE NOT FINITE")
                invalid_count += 1
                continue

            # Convert with the inverse affine transform. Rasterio's dem.index()
            # crashes natively in the installed environment, while this is its
            # equivalent floor-based pixel conversion and also supports rotation.

            try:
                col_float, row_float = (~dem.transform) * (x, y)
                row = math.floor(row_float)
                col = math.floor(col_float)
            except Exception as error:
                print(f"\n{site_name}: PIXEL COORDINATE CONVERSION FAILED: {error}")
                invalid_count += 1
                continue

            # Check whether the pixel is inside the DEM.

            inside = (
                0 <= row < dem.height
                and 0 <= col < dem.width
            )

            print("\n" + "-" * 50)
            print(f"Site: {site_name}")
            print(f"Original Longitude: {lon}")
            print(f"Original Latitude: {lat}")
            print(f"Projected X: {x:.2f} m")
            print(f"Projected Y: {y:.2f} m")

            if not inside:

                print("STATUS: OUTSIDE DEM BOUNDS")
                print(f"Raster Row: {row}")
                print(f"Raster Column: {col}")

                outside_count += 1
                continue

            inside_count += 1

            print("STATUS: INSIDE DEM BOUNDS")
            print(f"Raster Row: {row}")
            print(f"Raster Column: {col}")

            # Read elevation at the corresponding pixel.

            try:
                pixel = dem.read(
                    1,
                    window=Window(col, row, 1, 1),
                    masked=True
                )
                sample = pixel[0, 0]
            except Exception as error:
                print(f"Elevation: READ ERROR ({error})")
                invalid_count += 1
                continue

            if (
                sample is None
                or bool(getattr(sample, "mask", False))
                or not math.isfinite(float(sample))
            ):

                print("Elevation: NODATA")

                nodata_count += 1
                continue

            elevation = float(sample)

            print(f"DEM Elevation: {elevation:.2f} m")

            # Compare with elevation recorded in GeoJSON.

            recorded_elevation = properties.get("elevation_m")

            if recorded_elevation is not None:

                try:
                    difference = abs(elevation - float(recorded_elevation))
                except (TypeError, ValueError, OverflowError):
                    print("GeoJSON elevation: INVALID VALUE (comparison skipped)")
                    continue

                print(
                    "Difference from GeoJSON elevation: "
                    f"{difference:.2f} m"
                )

        # ----------------------------------
        # STEP 6: SUMMARY
        # ----------------------------------

        print("\n" + "=" * 65)
        print("SPATIAL VALIDATION SUMMARY")
        print("=" * 65)

        print(f"Total records: {len(features)}")
        print(f"Unique coordinate pairs: {len(unique_coordinates)}")
        print(f"Inside DEM: {inside_count}")
        print(f"Outside DEM: {outside_count}")
        print(f"Invalid records: {invalid_count}")
        print(f"NODATA pixels: {nodata_count}")

        print("\nCOORDINATE SYSTEM ASSUMPTION:")
        print(
            "GeoJSON coordinates were interpreted as "
            "longitude/latitude on the Mars 2000 sphere."
        )

        print(
            "This assumption still requires confirmation "
            "against the original coordinate source."
        )

        print("\nSPATIAL VALIDATION COMPLETED.")


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(f"\nERROR: {error}")
        traceback.print_exc()
        raise SystemExit(1) from None

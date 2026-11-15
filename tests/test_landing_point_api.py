"""Tests for lon/lat to pixel conversion and landing assessment integration."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer

from src import landing_point_api as api
from src import route_planner as planner


TEST_CRS = CRS.from_proj4(
    "+proj=eqc +lat_ts=18.4663 +lat_0=0 +lon_0=0 "
    "+a=3396190 +b=3396190 +units=m +no_defs"
)


class LandingPointApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the complete API and terrain evaluator once on the real DEM.
        cls.success = api.evaluate_landing_point_lonlat(
            77.7802176586129, 18.446480857911807, 500.0
        )

    def assert_strict_json(self, value):
        json.dumps(value, allow_nan=False)

    def test_real_dem_call_returns_json_safe_assessment(self):
        result = self.success
        self.assertTrue(result["success"], result.get("error"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["pixel"]["row"], 2500)
        self.assertEqual(result["pixel"]["col"], 2200)
        self.assertEqual(result["assessment"]["valid_pixel_count"], 1961)
        self.assertEqual(result["assessment"]["final_assessment"], "More Suitable")
        self.assertEqual(result["dem"]["name"], planner.DEM_PATH.name)
        self.assertFalse(result["assumptions"]["scientifically_confirmed"])
        self.assert_strict_json(result)

    def test_lonlat_pixel_conversion_matches_route_planner_affine_method(self):
        row0, col0 = 2500, 2200
        with rasterio.open(planner.DEM_PATH) as dem:
            x_m, y_m = dem.transform * (col0 + 0.5, row0 + 0.5)
            inverse = Transformer.from_crs(dem.crs, planner.SOURCE_CRS, always_xy=True)
            longitude, latitude = inverse.transform(x_m, y_m)
            forward = Transformer.from_crs(
                planner.SOURCE_CRS, dem.crs, always_xy=True
            )
            x_check, y_check = forward.transform(longitude, latitude, errcheck=True)
            col_float, row_float = planner.to_pixel(dem.transform, x_check, y_check)
            expected = (math.floor(row_float), math.floor(col_float))

        with patch.object(api, "evaluate_landing_point", return_value={"success": True}) as evaluate:
            result = api.evaluate_landing_point_lonlat(longitude, latitude)

        self.assertTrue(result["success"], result.get("error"))
        self.assertEqual((result["pixel"]["row"], result["pixel"]["col"]), expected)
        self.assertEqual(expected, (row0, col0))
        evaluate.assert_called_once_with(
            row0, col0, radius_m=500.0, dem_path=planner.DEM_PATH
        )

    def test_invalid_input_numbers_are_rejected_with_strict_json(self):
        cases = (
            ("77.7", 18.4),
            (True, 18.4),
            (math.nan, 18.4),
            (math.inf, 18.4),
            (77.7, -math.inf),
            (180.01, 18.4),
            (77.7, 90.01),
        )
        for longitude, latitude in cases:
            with self.subTest(longitude=repr(longitude), latitude=repr(latitude)):
                result = api.evaluate_landing_point_lonlat(longitude, latitude)
                self.assertFalse(result["success"])
                self.assertEqual(result["status"], "error")
                self.assert_strict_json(result)

    def test_location_outside_real_dem_is_rejected(self):
        result = api.evaluate_landing_point_lonlat(-180.0, 0.0)
        self.assertFalse(result["success"])
        self.assertIn("outside DEM bounds", result["error"]["message"])
        self.assert_strict_json(result)

    def test_nodata_center_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="landing_api_nodata_") as temp_dir:
            dem_path = Path(temp_dir) / "nodata_dem.tif"
            transform = Affine(20, 0, 4_372_800, 0, -20, 1_094_000)
            values = np.zeros((60, 60), dtype=np.float32)
            values[30, 30] = -9999.0
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=60,
                width=60,
                count=1,
                dtype="float32",
                crs=TEST_CRS,
                transform=transform,
                nodata=-9999.0,
            ) as dem:
                x_m, y_m = transform * (30.5, 30.5)
                inverse = Transformer.from_crs(TEST_CRS, planner.SOURCE_CRS, always_xy=True)
                longitude, latitude = inverse.transform(x_m, y_m)
                dem.write(values, 1)

            with (
                patch.object(api, "DEM_PATH", dem_path),
                patch.object(api, "evaluate_landing_point") as evaluate,
            ):
                result = api.evaluate_landing_point_lonlat(longitude, latitude)

        self.assertFalse(result["success"])
        self.assertIn("contains NoData", result["error"]["message"])
        evaluate.assert_not_called()
        self.assert_strict_json(result)

    def test_missing_dem_crs_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="landing_api_no_crs_") as temp_dir:
            dem_path = Path(temp_dir) / "no_crs_dem.tif"
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=4,
                width=4,
                count=1,
                dtype="float32",
                transform=Affine(20, 0, 1000, 0, -20, 2000),
            ) as dem:
                dem.write(np.zeros((4, 4), dtype=np.float32), 1)

            with patch.object(api, "DEM_PATH", dem_path):
                result = api.evaluate_landing_point_lonlat(77.0, 18.0)

        self.assertFalse(result["success"])
        self.assertIn("DEM has no CRS", result["error"]["message"])
        self.assert_strict_json(result)


if __name__ == "__main__":
    unittest.main()

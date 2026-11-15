"""Tests for the separated landing assessment and map renderer."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import rasterio
from affine import Affine
from pyproj import CRS

from src import landing_point_evaluator as evaluator


class LandingPointEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the project's real DEM and the recorded reference point once.
        cls.result = evaluator.evaluate_landing_point(2500, 2200, 500.0)

    def test_real_dem_reference_statistics_are_preserved(self):
        result = self.result
        self.assertTrue(result["success"])
        self.assertEqual((result["row"], result["col"]), (2500, 2200))
        self.assertEqual(result["radius_m"], 500.0)
        self.assertEqual(result["valid_pixel_count"], 1961)
        self.assertAlmostEqual(
            result["class_percentages"]["More Suitable"], 89.91, delta=0.02
        )
        self.assertAlmostEqual(
            result["class_percentages"]["Caution"], 10.09, delta=0.02
        )
        self.assertAlmostEqual(
            result["class_percentages"]["Unsuitable"], 0.00, delta=0.001
        )
        self.assertAlmostEqual(result["mean_slope_deg"], 1.922, places=3)
        self.assertAlmostEqual(result["mean_roughness_m"], 0.886, places=3)
        self.assertEqual(result["final_assessment"], "More Suitable")

    def test_result_is_strict_json_and_contains_no_numpy_arrays(self):
        serialized = json.dumps(self.result, allow_nan=False)
        self.assertTrue(serialized)
        self.assertNotIn("elevation", self.result)
        self.assertNotIn("slope", self.result)
        self.assertNotIn("roughness", self.result)
        grid = self.result["render_context"]["classification_grid"]
        self.assertIsInstance(grid, list)
        self.assertLessEqual(len(grid), evaluator.MAX_RENDER_DIMENSION)
        self.assertLessEqual(len(grid[0]), evaluator.MAX_RENDER_DIMENSION)

    def test_renderer_saves_png_from_result_without_re_evaluating(self):
        with tempfile.TemporaryDirectory(prefix="landing_assessment_") as temp_dir:
            path = Path(temp_dir) / "assessment.png"
            with patch.object(
                evaluator.rasterio,
                "open",
                side_effect=AssertionError("renderer must not reopen or recalculate the DEM"),
            ):
                returned = evaluator.render_landing_assessment(self.result, path)
            self.assertEqual(returned, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                image.verify()

    def test_invalid_coordinates_and_radius_are_rejected(self):
        for row, col, radius in (
            (-1, 2200, 500.0),
            (2500, 2200, 0.0),
            (2500, 2200, math.nan),
            (2500, 2200, math.inf),
        ):
            with self.subTest(row=row, col=col, radius=radius):
                with self.assertRaises(ValueError):
                    evaluator.evaluate_landing_point(row, col, radius)

    def test_nodata_center_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="landing_nodata_") as temp_dir:
            dem_path = Path(temp_dir) / "nodata_dem.tif"
            values = np.zeros((20, 20), dtype=np.float32)
            values[10, 10] = -9999.0
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=20,
                width=20,
                count=1,
                dtype="float32",
                crs=CRS.from_proj4("+proj=eqc +a=3396190 +b=3396190 +units=m +no_defs"),
                transform=Affine(20, 0, 1000, 0, -20, 2000),
                nodata=-9999.0,
            ) as dem:
                dem.write(values, 1)

            with self.assertRaisesRegex(ValueError, "has no valid DEM data"):
                evaluator.evaluate_landing_point(10, 10, 100.0, dem_path=dem_path)


if __name__ == "__main__":
    unittest.main()

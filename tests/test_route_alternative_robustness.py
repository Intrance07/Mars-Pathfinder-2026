"""Edge-case tests for route_planner's current second-route strategy.

Each test creates a tiny synthetic GeoTIFF and GeoJSON under a temporary
directory, patches only route_planner's input paths/terrain layer result, and
calls the real calculate_routes/A* functions. No project data or outputs are
written or changed.
"""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer

from src import route_planner as planner


HEIGHT = 60
WIDTH = 60
TEST_CRS = CRS.from_proj4(
    "+proj=eqc +lat_ts=18.4663 +lat_0=0 +lon_0=0 "
    "+a=3396190 +b=3396190 +units=m +no_defs"
)
TEST_TRANSFORM = Affine(20, 0, 4_372_800, 0, -20, 1_094_000)


def cell_center_lonlat(row, col):
    x, y = TEST_TRANSFORM * (col + 0.5, row + 0.5)
    inverse = Transformer.from_crs(TEST_CRS, planner.SOURCE_CRS, always_xy=True)
    return inverse.transform(x, y)


class RouteAlternativeRobustnessTests(unittest.TestCase):
    def run_synthetic_plan(self, classes, start, goal):
        """Call the real route calculation against isolated synthetic inputs."""
        classes = np.asarray(classes, dtype=np.uint8)
        self.assertEqual(classes.shape, (HEIGHT, WIDTH))
        start_lon, start_lat = cell_center_lonlat(*start)
        goal_lon, goal_lat = cell_center_lonlat(*goal)
        call_log = []

        def fake_terrain_layers(elevation, pixel_x, pixel_y):
            # Controlled class fixtures let tests isolate routing topology.
            shape = elevation.shape
            self.assertEqual(shape, classes.shape)
            slope = np.zeros(shape, dtype=np.float32)
            roughness = np.zeros(shape, dtype=np.float32)
            valid = classes != 0
            return slope, roughness, valid, classes.copy()

        actual_astar = planner.astar

        def recording_astar(passable, cell_weight, transform, astar_start, astar_goal):
            path = actual_astar(passable, cell_weight, transform, astar_start, astar_goal)
            call_log.append({"passable": passable.copy(), "path": path})
            return path

        with tempfile.TemporaryDirectory(prefix="mars_route_edge_") as temp_dir:
            temp_path = Path(temp_dir)
            dem_path = temp_path / "synthetic_dem.tif"
            sites_path = temp_path / "synthetic_sites.geojson"

            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=HEIGHT,
                width=WIDTH,
                count=1,
                dtype="float32",
                crs=TEST_CRS,
                transform=TEST_TRANSFORM,
                nodata=-9999.0,
            ) as raster:
                raster.write(np.zeros((HEIGHT, WIDTH), dtype=np.float32), 1)

            sites_path.write_text(
                json.dumps({
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [goal_lon, goal_lat]},
                        "properties": {"site_name": "Synthetic Target"},
                    }],
                }),
                encoding="utf-8",
            )

            with (
                patch.object(planner, "DEM_PATH", dem_path),
                patch.object(planner, "SITES_PATH", sites_path),
                patch.object(planner, "terrain_layers", side_effect=fake_terrain_layers),
                patch.object(planner, "astar", side_effect=recording_astar),
            ):
                try:
                    result = planner.calculate_routes(
                        start_lon=start_lon,
                        start_lat=start_lat,
                        target_site="Synthetic Target",
                        corridor_buffer_m=3000.0,
                    )
                    return result, call_log, None
                except Exception as error:
                    return None, call_log, error

    def assert_first_path_is_valid(self, call_log, classes):
        self.assertTrue(call_log, "A* was not called")
        first_path = call_log[0]["path"]
        self.assertIsNotNone(first_path, "Expected a valid first route")
        for row, col in first_path:
            self.assertNotEqual(int(classes[row, col]), 3, "First path crossed Unsuitable terrain")

    def test_very_close_start_and_goal_returns_only_the_valid_first_route(self):
        classes = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        result, calls, error = self.run_synthetic_plan(classes, (30, 30), (30, 31))

        self.assertIsNotNone(result, str(error))
        self.assertIsNone(error)
        self.assertEqual(len(result["routes"]), 1)
        self.assertFalse(result["alternative_available"])
        self.assertIn("distinct second route", result["alternative_message"])
        self.assert_first_path_is_valid(calls, classes)
        self.assertEqual(len(calls[0]["path"]), 2)
        self.assertEqual(calls[0]["path"], calls[-1]["path"])

    def test_route_shorter_than_eight_steps_returns_first_route_without_duplicate(self):
        classes = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        result, calls, error = self.run_synthetic_plan(classes, (30, 20), (30, 26))

        self.assertIsNotNone(result, str(error))
        self.assertIsNone(error)
        self.assertEqual(len(result["routes"]), 1)
        self.assertFalse(result["alternative_available"])
        self.assertIn("distinct second route", result["alternative_message"])
        self.assert_first_path_is_valid(calls, classes)
        self.assertLess(len(calls[0]["path"]), 8)
        self.assertEqual(calls[0]["path"], calls[-1]["path"])

    def test_narrow_two_cell_corridor_can_produce_a_distinct_alternative(self):
        classes = np.full((HEIGHT, WIDTH), 3, dtype=np.uint8)
        classes[29:31, 2:58] = 1
        result, calls, error = self.run_synthetic_plan(classes, (30, 5), (30, 54))

        self.assertIsNotNone(result, str(error))
        self.assertIsNone(error)
        self.assertEqual(len(result["routes"]), 2)
        self.assertTrue(result["alternative_available"])
        self.assertIsNone(result["alternative_message"])
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["path"], calls[1]["path"])
        for path in (calls[0]["path"], calls[1]["path"]):
            for row, col in path:
                self.assertNotEqual(int(classes[row, col]), 3, "Route crossed Unsuitable terrain")

    def test_goal_surrounded_by_unsuitable_cells_has_no_first_route(self):
        classes = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        goal = (30, 40)
        classes[goal] = 1
        for row in range(goal[0] - 1, goal[0] + 2):
            for col in range(goal[1] - 1, goal[1] + 2):
                if (row, col) != goal:
                    classes[row, col] = 3
        result, calls, error = self.run_synthetic_plan(classes, (30, 10), goal)

        self.assertIsNone(result)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("No route found", str(error))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["path"])

    def test_complete_unsuitable_barrier_means_no_first_route(self):
        classes = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        classes[:, 30] = 3
        result, calls, error = self.run_synthetic_plan(classes, (30, 10), (30, 50))

        self.assertIsNone(result)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("No route found", str(error))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["path"])

    def test_valid_first_route_without_independent_alternative_is_returned_once(self):
        classes = np.full((HEIGHT, WIDTH), 3, dtype=np.uint8)
        classes[30, 3:57] = 1  # A unique, one-cell-wide traversable corridor.
        result, calls, error = self.run_synthetic_plan(classes, (30, 8), (30, 52))

        self.assertIsNotNone(result, str(error))
        self.assertIsNone(error)
        self.assertEqual(len(result["routes"]), 1)
        self.assertFalse(result["alternative_available"])
        self.assertIn("distinct second route", result["alternative_message"])
        self.assert_first_path_is_valid(calls, classes)
        self.assertEqual(len(calls), 3)
        self.assertIsNone(calls[1]["path"], "Hard exclusion should remove the only route")
        self.assertEqual(calls[0]["path"], calls[-1]["path"], "Penalty fallback finds the same sole path")
        for row, col in calls[-1]["path"]:
            self.assertNotEqual(int(classes[row, col]), 3, "Fallback path crossed Unsuitable terrain")

        route_context = result["_render_context"]["routes"]
        self.assertEqual(len(route_context), 1, "Renderer context must not contain a duplicate route")
        geojson = planner.build_geojson_document(result)
        self.assertEqual(len(geojson["features"]), 1)
        self.assertFalse(geojson["alternative_available"])
        self.assertIn("distinct second route", geojson["alternative_message"])

    def test_cli_saves_one_route_without_replacing_project_outputs(self):
        classes = np.full((HEIGHT, WIDTH), 3, dtype=np.uint8)
        classes[30, 3:57] = 1
        result, _, error = self.run_synthetic_plan(classes, (30, 8), (30, 52))
        self.assertIsNotNone(result, str(error))

        args = Namespace(
            target_site="Synthetic Target",
            start_row=None,
            start_col=None,
            start_lon=result["start"]["input_longitude"],
            start_lat=result["start"]["input_latitude"],
            corridor_buffer_m=3000.0,
            slope_weight=2.0,
            roughness_weight=2.0,
            caution_penalty=4.0,
        )
        with tempfile.TemporaryDirectory(prefix="mars_route_cli_") as temp_dir:
            output_dir = Path(temp_dir) / "outputs"
            geojson_path = output_dir / "routes.geojson"
            png_path = output_dir / "routes.png"
            with (
                patch.object(planner, "arguments", return_value=args),
                patch.object(planner, "calculate_routes", return_value=result),
                patch.object(planner, "OUT_DIR", output_dir),
                patch.object(planner, "GEOJSON_PATH", geojson_path),
                patch.object(planner, "PNG_PATH", png_path),
            ):
                planner.main()

            geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
            self.assertEqual(len(geojson["features"]), 1)
            self.assertFalse(geojson["alternative_available"])
            self.assertTrue(png_path.is_file())
            self.assertGreater(png_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

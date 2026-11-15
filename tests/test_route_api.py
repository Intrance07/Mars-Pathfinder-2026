"""Strict JSON and input-validation tests for the in-memory route API."""

import json
import math
import unittest
from decimal import Decimal
from unittest.mock import patch

from src import route_api


class RouteApiTests(unittest.TestCase):
    def assert_strict_json(self, response):
        json.dumps(response, allow_nan=False)

    def test_real_route_response_is_strict_json(self):
        response = route_api.plan_routes(
            77.7802176586129,
            18.446480857911807,
            "Montagnac",
        )

        self.assertTrue(response["success"], response.get("error"))
        self.assertEqual(len(response["routes"]), 2)
        self.assertTrue(response["alternative_available"])
        self.assertIsNone(response["alternative_message"])
        self.assert_strict_json(response)
        self.assertFalse(response["files_written"])

    def test_api_keeps_first_route_when_alternative_is_unavailable(self):
        first_route = {"route_id": "Route 1 (lowest weighted cost)", "distance_m": 100.0}
        engine_result = {
            "start": {"row": 10, "col": 10},
            "target": {"site_names": "Synthetic Target"},
            "routes": [first_route],
            "alternative_available": False,
            "alternative_message": "A distinct second route could not be found inside the selected corridor.",
            "assumptions": route_api.CRS_ASSUMPTION,
            "dem": {"resolution_m": [20.0, 20.0]},
        }
        with patch.object(route_api, "calculate_routes", return_value=engine_result):
            response = route_api.plan_routes(77.7, 18.4, "Synthetic Target")

        self.assertTrue(response["success"])
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["routes"], [first_route])
        self.assertFalse(response["alternative_available"])
        self.assertIn("distinct second route", response["alternative_message"])
        self.assert_strict_json(response)

    def test_invalid_longitudes_return_strict_json_errors(self):
        invalid_values = (
            "77.7",
            True,
            object(),
            math.nan,
            math.inf,
            -math.inf,
            180.01,
            -180.01,
        )
        for value in invalid_values:
            with self.subTest(start_lon=type(value).__name__):
                response = route_api.plan_routes(value, 18.4, "Montagnac")
                self.assertFalse(response["success"])
                self.assertIn("start_lon", response["error"]["message"])
                self.assert_strict_json(response)

    def test_invalid_latitudes_return_strict_json_errors(self):
        invalid_values = (
            "18.4",
            True,
            object(),
            math.nan,
            math.inf,
            -math.inf,
            90.01,
            -90.01,
        )
        for value in invalid_values:
            with self.subTest(start_lat=type(value).__name__):
                response = route_api.plan_routes(77.7, value, "Montagnac")
                self.assertFalse(response["success"])
                self.assertIn("start_lat", response["error"]["message"])
                self.assert_strict_json(response)

    def test_invalid_target_is_reported_and_json_safe(self):
        for target in (None, "  ", object()):
            with self.subTest(target=type(target).__name__):
                response = route_api.plan_routes(77.7, 18.4, target)
                self.assertFalse(response["success"])
                self.assertIn("target_site", response["error"]["message"])
                self.assert_strict_json(response)

    def test_decimal_coordinates_are_accepted_as_numeric(self):
        with patch.object(route_api, "calculate_routes") as calculate:
            calculate.return_value = {
                "start": {},
                "target": {},
                "routes": [],
                "assumptions": route_api.CRS_ASSUMPTION,
                "dem": {},
            }
            response = route_api.plan_routes(Decimal("77.7"), Decimal("18.4"), "Montagnac")

        self.assertTrue(response["success"])
        calculate.assert_called_once_with(
            start_lon=77.7,
            start_lat=18.4,
            target_site="Montagnac",
        )
        self.assert_strict_json(response)

    def test_route_engine_exception_is_json_safe(self):
        with patch.object(route_api, "calculate_routes", side_effect=RuntimeError("route failed")):
            response = route_api.plan_routes(77.7, 18.4, "Montagnac")

        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["type"], "RuntimeError")
        self.assertEqual(response["error"]["message"], "route failed")
        self.assert_strict_json(response)


if __name__ == "__main__":
    unittest.main()

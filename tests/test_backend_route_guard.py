"""Endpoint-level tests for route-planner admission and resource safeguards."""

import json
import threading
import unittest
from unittest.mock import patch

from backend import main
import route_planner as backend_planner


def response_payload(response):
    if hasattr(response, "body"):
        return json.loads(response.body)
    return response


class BackendRouteGuardTests(unittest.TestCase):
    @staticmethod
    def route_payload():
        return {
            "start_lon": 77.7802176586129,
            "start_lat": 18.446480857911807,
            "target_site": "Montagnac",
        }

    def test_near_real_dem_route_succeeds_and_logs_window_metrics(self):
        request = main.RouteRequest(**self.route_payload())

        with self.assertLogs(main._route_logger, level="INFO") as captured_logs:
            result = main.plan(request)

        self.assertTrue(result["success"], result.get("error"))
        self.assertEqual(len(result["routes"]), 2)
        self.assertTrue(result["alternative_available"])
        self.assertAlmostEqual(result["routes"][0]["distance_m"], 20048.082245892125, places=6)
        self.assertAlmostEqual(result["routes"][1]["distance_m"], 20495.43289325506, places=6)
        log_text = "\n".join(captured_logs.output)
        self.assertIn("height=354", log_text)
        self.assertIn("width=1251", log_text)
        self.assertIn("cells=442854", log_text)
        self.assertIn("duration_ms=", log_text)
        self.assertIn("result=success", log_text)

    def test_busy_route_returns_503_and_other_endpoints_remain_available(self):
        entered = threading.Event()
        release = threading.Event()
        first_response = {}

        def slow_route(start_lon, start_lat, target_site, diagnostics_callback=None):
            if diagnostics_callback is not None:
                diagnostics_callback(
                    {"height": 123, "width": 234, "cell_count": 28782, "max_cells": 2_000_000}
                )
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release the simulated route request")
            return {"success": True, "status": "success", "routes": []}

        with patch.object(main, "plan_routes", side_effect=slow_route):
            thread = threading.Thread(
                target=lambda: first_response.update(
                    {"response": main.plan(main.RouteRequest(**self.route_payload()))}
                ),
                daemon=True,
            )
            thread.start()
            self.assertTrue(entered.wait(timeout=3), "first route request did not acquire the planner")

            busy = main.plan(main.RouteRequest(**self.route_payload()))
            health = main.health()
            sites = main.sites()

            release.set()
            thread.join(timeout=5)
            next_request = main.plan(main.RouteRequest(**self.route_payload()))

        self.assertFalse(thread.is_alive())
        self.assertEqual(busy.status_code, 503)
        self.assertEqual(busy.headers.get("Retry-After"), "10")
        busy_body = response_payload(busy)
        self.assertEqual(busy_body["error"]["type"], "RoutePlannerBusyError")
        self.assertIn("Retry", busy_body["error"]["message"])
        self.assertEqual(health, {"status": "ok"})
        self.assertEqual(len(sites["sites"]), 10)
        self.assertEqual(first_response["response"]["success"], True)
        # The lock is released after the completed calculation.
        self.assertTrue(next_request["success"])

    def test_oversized_real_dem_window_is_rejected_before_terrain_allocation(self):
        payload = {
            "start_lon": 78.57918390319612,
            "start_lat": 19.286981591943107,
            "target_site": "Montagnac",
        }

        with (
            self.assertLogs(main._route_logger, level="INFO") as captured_logs,
            patch.object(
                backend_planner,
                "terrain_layers",
                side_effect=AssertionError("terrain arrays must not be allocated for a rejected window"),
            ),
        ):
            response = main.plan(main.RouteRequest(**payload))
        body = response_payload(response)

        self.assertEqual(response.status_code, 422)
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["type"], "SearchWindowLimitError")
        self.assertEqual(
            body["search_window"],
            {
                "height": 2701,
                "width": 3353,
                "cell_count": 9_056_453,
                "max_cells": 2_000_000,
            },
        )
        self.assertIn("no route was computed", body["error"]["message"])
        log_text = "\n".join(captured_logs.output)
        self.assertIn("height=2701", log_text)
        self.assertIn("width=3353", log_text)
        self.assertIn("cells=9056453", log_text)
        self.assertIn("result=error:SearchWindowLimitError", log_text)
        self.assertIn("duration_ms=", log_text)

    def test_arbitrary_multiline_target_is_not_written_to_logs(self):
        hostile_target = "Montagnac\r\nFORGED_LOG secret=do-not-log"
        request = main.RouteRequest(**{
            **self.route_payload(),
            "target_site": hostile_target,
        })

        with self.assertLogs(main._route_logger, level="INFO") as captured_logs:
            result = main.plan(request)

        self.assertFalse(result["success"])
        log_text = "\n".join(captured_logs.output)
        self.assertIn("target_site_id=unrecognized", log_text)
        self.assertNotIn("Montagnac", log_text)
        self.assertNotIn("FORGED_LOG", log_text)
        self.assertNotIn("secret=do-not-log", log_text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecast_week1 import (
    format_submission_payload,
    generate_forecast_from_csv,
    normalize_history,
    run_baseline_forecast,
)


class ForecastWeek1Tests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2022-07-01", periods=35, freq="D")
        rows = []
        for idx, option_id in enumerate(["Option_A", "Option_B", "Option_C"], start=1):
            for i, date in enumerate(dates):
                base = idx * 3
                weekend_bonus = 2 if date.dayofweek >= 5 else 0
                rows.append(
                    {
                        "option_id": option_id,
                        "date": date.strftime("%Y-%m-%d"),
                        "sales_qty": base + weekend_bonus + (i % 4 == 0),
                    }
                )
        self.raw = pd.DataFrame(rows)

    def test_normalize_history_fills_missing_dates(self) -> None:
        raw = self.raw[self.raw["date"] != "2022-07-10"]
        history = normalize_history(raw)
        target = history[(history["option_id"] == "Option_A") & (history["date"] == "2022-07-10")]
        self.assertEqual(len(target), 1)
        self.assertEqual(float(target["target"].iloc[0]), 0.0)

    def test_run_baseline_forecast_produces_non_negative_horizon(self) -> None:
        history = normalize_history(self.raw)
        forecast_panel = run_baseline_forecast(history, horizon=28)
        self.assertEqual(len(forecast_panel), 3 * 28)
        self.assertTrue((forecast_panel["forecast"] >= 0).all())

    def test_submission_payload_matches_template_order(self) -> None:
        history = normalize_history(self.raw)
        forecast_panel = run_baseline_forecast(history, horizon=28)
        template = [
            {"option_id": "Option_C", "forecast": [0.0] * 28},
            {"option_id": "Option_A", "forecast": [0.0] * 28},
            {"option_id": "Option_B", "forecast": [0.0] * 28},
        ]
        payload = format_submission_payload(forecast_panel, template=template, horizon=28)
        self.assertEqual([row["option_id"] for row in payload], ["Option_C", "Option_A", "Option_B"])
        self.assertTrue(all(len(row["forecast"]) == 28 for row in payload))

    def test_csv_mode_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "history.csv"
            self.raw.to_csv(csv_path, index=False)
            artifacts = generate_forecast_from_csv(csv_path)
            self.assertEqual(len(artifacts.submission_payload), 3)
            self.assertEqual(len(artifacts.submission_payload[0]["forecast"]), 28)


if __name__ == "__main__":
    unittest.main()

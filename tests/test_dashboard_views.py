"""Dashboard data shaping (dashboard/views.py) — no Streamlit needed."""

from __future__ import annotations

import pytest

from views import (
    canary_verdict,
    drift_table,
    drivers_frame,
    filter_alerts,
    precision_by_band,
    velocity_table,
)


def _alert(tx, band, score, fraud, card="aaaa1111bbbb2222", ts="2026-01-01T00:00:00Z"):
    return {"transaction_id": tx, "risk_band": band, "score": score, "is_fraud": fraud,
            "card_key": card, "scored_at": ts}  # fmt: skip


ALERTS = [
    _alert(1, "high", 0.95, 1, ts="2026-01-01T00:00:01Z"),
    _alert(2, "medium", 0.60, 0, card="ffff0000eeee9999", ts="2026-01-01T00:00:02Z"),
    _alert(3, "medium", 0.45, 0, ts="2026-01-01T00:00:03Z"),
    _alert(4, "high", 0.85, 0, ts="2026-01-01T00:00:04Z"),
]


class TestFilters:
    def test_newest_first_and_band_filter(self):
        rows = filter_alerts(ALERTS, ["high"])
        assert [a["transaction_id"] for a in rows] == [4, 1]

    def test_label_score_and_card_query(self):
        assert [a["transaction_id"] for a in filter_alerts(ALERTS, ["high", "medium"],
                                                           label="fraud")] == [1]  # fmt: skip
        assert len(filter_alerts(ALERTS, ["high", "medium"], min_score=0.5)) == 3
        assert [a["transaction_id"] for a in filter_alerts(
            ALERTS, ["high", "medium"], card_query="FFFF00")] == [2]  # fmt: skip
        assert [a["transaction_id"] for a in filter_alerts(
            ALERTS, ["high", "medium"], card_query="3")] == [3]  # fmt: skip


def test_precision_by_band():
    table = precision_by_band(ALERTS).set_index("band")
    assert table.loc["🔴 high", "alerts"] == 2 and table.loc["🔴 high", "precision"] == 0.5
    assert table.loc["🟠 medium", "precision"] == 0.0


def test_velocity_table_rows_per_window():
    table = velocity_table({"txn_count_1h": 3.0, "amt_sum_1h": 30.0, "amt_mean_1h": 10.0})
    assert table.loc["1h"].tolist() == [3.0, 30.0, 10.0]
    assert table.loc["5m"].isna().all()


def test_drift_table_sorted_by_ratio_and_marks_top20():
    report = {"ratio": {"a": 0.2, "b": 1.4, "c": 0.7}, "important": ["b"],
              "features": {"a": 0.05, "b": 0.6, "c": 0.3},
              "finite_psi": {"a": 0.0, "b": 0.1, "c": 0.2}}  # fmt: skip
    table = drift_table(report, top=2)
    assert table["feature"].tolist() == ["b", "c"] and table["top-20"].tolist() == ["★", ""]
    assert drift_table({}).empty


def test_drivers_frame_is_signed_series_for_diverging_chart():
    drivers = [{"feature": "V201", "impact": 0.42}, {"feature": "amt_sum_1h", "impact": -0.13}]
    frame = drivers_frame(drivers)
    assert frame["V201"] == 0.42 and frame["amt_sum_1h"] == -0.13
    assert drivers_frame([]).empty


@pytest.mark.parametrize(
    ("canary", "state", "detail_bit"),
    [
        ({"challenger_version": None}, "none", "No challenger"),
        ({"champion_version": "5", "challenger_version": "5"}, "same", "current champion"),
        ({"challenger_version": "4", "events": 10}, "collecting", "Collecting"),
        (
            {
                "challenger_version": "4",
                "events": 5000,
                "p95_abs_delta": 0.118,
                "within_tolerance": False,
            },
            "fail",
            "blocked",
        ),  # fmt: skip
        (
            {
                "challenger_version": "4",
                "events": 5000,
                "p95_abs_delta": 0.01,
                "within_tolerance": True,
            },
            "pass",
            "would succeed",
        ),  # fmt: skip
    ],
)
def test_canary_verdict(canary, state, detail_bit):
    verdict = canary_verdict(canary)
    assert verdict["state"] == state and detail_bit in verdict["detail"]

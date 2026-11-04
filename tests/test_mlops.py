"""Canary + drift monitor tests (serving/canary.py, eval/drift.py, PLAN.md §10)."""

from __future__ import annotations

import numpy as np
import pytest

from streamguard.eval.drift import PSI_FLAG, DriftMonitor, reference_sample
from streamguard.eval.latency import hop_stats
from streamguard.serving.canary import MIN_EVENTS, TOLERANCE, CanaryWindow, summarize


class TestCanary:
    def test_close_models_within_tolerance(self):
        rng = np.random.default_rng(0)
        champion = rng.random(5000)
        window = CanaryWindow()
        window.record(champion, champion + rng.normal(0, 0.002, 5000),
                      champion > 0.5, champion > 0.5)  # fmt: skip
        summary = window.summary()
        assert summary["events"] == 5000 and summary["p95_abs_delta"] < TOLERANCE
        assert summary["alert_agreement"] == 1.0 and summary["within_tolerance"]

    def test_divergent_models_fail(self):
        deltas = np.full(MIN_EVENTS, 0.1)
        assert not summarize(deltas, np.ones(MIN_EVENTS, bool))["within_tolerance"]

    def test_too_few_events_never_pass(self):
        assert not summarize(np.zeros(10), np.ones(10, bool))["within_tolerance"]
        assert summarize(np.array([]), np.array([], bool))["events"] == 0


class TestDriftMonitor:
    def test_flags_only_important_shifted_features(self):
        rng = np.random.default_rng(0)
        train = rng.normal(size=(20_000, 3))
        monitor = DriftMonitor(reference_sample(train), ["a", "b", "c"], important=["a", "b"],
                               min_rows=1000)  # fmt: skip
        live = rng.normal(size=(3000, 3))
        live[:, 1] += 2.0  # important feature drifts
        live[:, 2] += 2.0  # unimportant feature drifts
        monitor.add(live)
        report = monitor.compute()
        assert report["flagged"] == ["b"] and report["drift_flag"]
        assert report["features"]["c"] > PSI_FLAG and report["features"]["a"] < PSI_FLAG

    def test_waits_for_enough_rows(self):
        monitor = DriftMonitor(np.zeros((100, 1)), ["a"], ["a"], min_rows=50)
        monitor.add(np.zeros((10, 1)))
        assert monitor.compute()["features"] == {}


def test_hop_stats_from_spans():
    fields = ["transaction_id", "created", "consumed", "snapshot_written",
              "features_ready", "scored", "published"]  # fmt: skip
    spans = [[i, 0.0, 0.1, 1.0, 1.2, 1.25, 1.3] for i in range(100)]
    spans.append([999, 0.0, 0.1, float("nan"), 1.2, 1.25, 1.3])  # feature miss: no Spark time
    stats = hop_stats(spans, fields)
    assert stats["end-to-end"]["p95"] == pytest.approx(1.3)
    assert stats["broker → Spark → Redis"]["n"] == 100
    assert stats["inference"]["p50"] == pytest.approx(0.05)

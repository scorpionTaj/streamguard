"""Canary + drift monitor tests (serving/canary.py, eval/drift.py, PLAN.md §10)."""

from __future__ import annotations

import numpy as np
import pytest

from streamguard.eval.drift import (
    PSI_FLAG,
    DriftMonitor,
    DriftReference,
    calibrate_thresholds,
    reference_sample,
)
from streamguard.eval.latency import hop_stats
from streamguard.eval.metrics import psi_finite
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


def _seasonal_train(rng, n=40_000):
    """Feature 0 calm; feature 1 alternates its mean between weekly-ish regimes."""
    x = rng.normal(size=(n, 3))
    x[:, 1] += np.where((np.arange(n) // 5_000) % 2 == 0, 0.0, 1.5)
    return x


class TestCalibration:
    def test_bursty_feature_gets_higher_threshold_with_floor(self):
        rng = np.random.default_rng(0)
        train = _seasonal_train(rng)
        thresholds = calibrate_thresholds(
            reference_sample(train), train, window=5_000, stride=2_500
        )
        assert thresholds[0] == pytest.approx(PSI_FLAG)  # calm → floor
        assert thresholds[1] > 2 * PSI_FLAG  # regime swings are normal for this feature

    def test_window_must_fit(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 2))
        assert calibrate_thresholds(x, x, window=1_000).tolist() == [PSI_FLAG, PSI_FLAG]


def _monitor(train, important, window=5_000):
    sample = reference_sample(train)
    kw = {"window": window, "stride": window // 2}
    ref = DriftReference(sample, ["a", "b", "c"], calibrate_thresholds(sample, train, **kw),
                         calibrate_thresholds(sample, train, statistic=psi_finite, **kw), window)  # fmt: skip
    return DriftMonitor(ref, important)


class TestDriftMonitor:
    def test_in_distribution_regime_is_not_drift(self):
        rng = np.random.default_rng(1)
        train = _seasonal_train(rng)
        monitor = _monitor(train, ["a", "b", "c"])
        live = rng.normal(size=(5_000, 3))
        live[:, 1] += 1.5  # a regime already seen during training
        monitor.add(live)
        report = monitor.compute()
        assert report["features"]["b"] > PSI_FLAG  # the old fixed rule would have fired
        assert not report["drift_flag"]

    def test_real_shift_on_important_feature_flags(self):
        rng = np.random.default_rng(2)
        train = _seasonal_train(rng)
        monitor = _monitor(train, ["a", "b"])
        live = rng.normal(size=(5_000, 3))
        live[:, 0] += 1.0  # calm feature moves: drift
        live[:, 2] += 3.0  # unimportant feature: reported, not flagged
        monitor.add(live)
        report = monitor.compute()
        assert report["flagged"] == ["a"] and report["drift_flag"]
        assert report["features"]["c"] / report["thresholds"]["c"] > 1.0

    def test_value_shift_in_mostly_missing_feature_flags(self):
        rng = np.random.default_rng(4)
        train = rng.normal(size=(40_000, 3))
        train[rng.random(40_000) < 0.85, 2] = np.nan  # like V187: 85 % missing
        monitor = _monitor(train, ["c"])
        live = rng.normal(size=(5_000, 3))
        live[rng.random(5_000) < 0.85, 2] = np.nan
        live[:, 2] += 0.7  # present values shift; missing share unchanged
        monitor.add(live)
        report = monitor.compute()
        assert report["features"]["c"] < report["thresholds"]["c"]  # overall PSI misses it
        assert report["flagged"] == ["c"]  # present-only PSI catches it

    def test_waits_for_a_full_window(self):
        rng = np.random.default_rng(3)
        monitor = _monitor(_seasonal_train(rng), ["a"])
        monitor.add(rng.normal(size=(4_999, 3)))
        report = monitor.compute()
        assert report["features"] == {} and report["rows"] == 4_999


def test_hop_stats_from_spans():
    fields = ["transaction_id", "created", "consumed", "snapshot_written",
              "features_ready", "scored", "published"]  # fmt: skip
    spans = [[i, 0.0, 0.1, 1.0, 1.2, 1.25, 1.3] for i in range(100)]
    spans.append([999, 0.0, 0.1, float("nan"), 1.2, 1.25, 1.3])  # feature miss: no Spark time
    stats = hop_stats(spans, fields)
    assert stats["end-to-end"]["p95"] == pytest.approx(1.3)
    assert stats["broker → Spark → Redis"]["n"] == 100
    assert stats["inference"]["p50"] == pytest.approx(0.05)

"""Threshold policy & PSI tests (eval/metrics.py, PLAN.md §8, §9)."""

from __future__ import annotations

import numpy as np
import pytest

from streamguard.eval.metrics import precision_at_k, psi, psi_finite, threshold_at_fpr


class TestThresholdAtFpr:
    def test_separable_scores_hit_recall_one(self):
        y_true = np.array([0] * 90 + [1] * 10)
        y_score = np.array([*np.linspace(0.0, 0.8, 90), *np.linspace(0.9, 1.0, 10)])
        result = threshold_at_fpr(y_true, y_score, max_fpr=0.05)
        assert result.recall == pytest.approx(1.0)
        assert result.fpr <= 0.05 + 1e-9

    def test_fpr_budget_respected_with_overlap(self):
        rng = np.random.default_rng(42)
        y_score = np.concatenate([rng.beta(2, 2, 1000), rng.beta(5, 2, 100)])
        y_true = np.array([0] * 1000 + [1] * 100)
        result = threshold_at_fpr(y_true, y_score, max_fpr=0.05)
        # recompute FPR at the chosen threshold — must hold
        flagged = y_score >= result.threshold
        fpr = ((y_true == 0) & flagged).sum() / (y_true == 0).sum()
        assert fpr <= 0.05 + 1e-9

    def test_clean_top_positive_beats_zero_budget(self):
        # max_fpr = 0 is feasible when the highest scorer is a true positive
        y_true = np.array([0, 0, 0, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])
        result = threshold_at_fpr(y_true, y_score, max_fpr=0.0)
        assert result.recall == pytest.approx(1.0) and result.fpr == 0.0

    def test_top_negative_gates_everything(self):
        # no threshold can flag anything without a false positive → nothing served
        y_true = np.array([0, 0, 1, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])
        result = threshold_at_fpr(y_true, y_score, max_fpr=0.0)
        assert result.recall == 0.0 and result.fpr == 0.0

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            threshold_at_fpr(np.array([1, 0]), np.array([0.5]))


class TestPsi:
    def test_identical_distribution_near_zero(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(0, 1, 10_000)
        assert psi(sample, sample) < 1e-6

    def test_shifted_distribution_flags(self):
        rng = np.random.default_rng(0)
        expected = rng.normal(0, 1, 10_000)
        actual = rng.normal(1.5, 1, 10_000)
        assert psi(expected, actual) > 0.25

    def test_missing_share_shift_flags(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 10_000)
        mostly_missing = np.where(rng.random(10_000) < 0.6, np.nan, values)
        assert psi(values, values) < 1e-6
        assert psi(values, mostly_missing) > 0.25
        assert np.isfinite(psi(np.full(100, np.nan), values))

    def test_present_only_psi_sees_value_shift_behind_missing_bin(self):
        rng = np.random.default_rng(0)
        reference = np.where(rng.random(20_000) < 0.85, np.nan, rng.normal(0, 1, 20_000))
        shifted = np.where(np.isnan(reference), np.nan, reference + 3.0)
        assert psi(reference, shifted) < 1.0  # missing bin dominates
        assert psi_finite(reference, shifted) > 3 * psi(reference, shifted)
        assert psi_finite(reference, reference) < 1e-6

    def test_dominant_value_shift_detected(self):
        # 93 % of present values identical (like V187): quantile edges would collapse
        rng = np.random.default_rng(0)
        reference = rng.choice([1.0, 2.0, 3.0], 5_000, p=[0.93, 0.05, 0.02])
        shifted = reference.copy()
        shifted[rng.choice(np.flatnonzero(reference == 1.0), 1_500, replace=False)] = 2.0
        assert psi(reference, reference) < 1e-6
        assert psi(reference, shifted) > 0.25
        assert psi(reference, np.full(100, 7.0)) > 1.0  # unseen value above the max

    def test_present_only_psi_needs_enough_values(self):
        sparse = np.full(1_000, np.nan)
        sparse[:50] = 1.0
        assert psi_finite(sparse, np.ones(1_000)) == 0.0

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            psi(np.array([1.0]), np.array([]))


class TestPrecisionAtK:
    def test_counts_top_k_only(self):
        y_true = np.array([1, 0, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
        assert precision_at_k(y_true, y_score, k=2) == pytest.approx(0.5)
        assert precision_at_k(y_true, y_score, k=3) == pytest.approx(2 / 3)

    def test_k_larger_than_sample(self):
        assert precision_at_k(np.array([1, 0]), np.array([0.6, 0.4]), k=50) == pytest.approx(0.5)

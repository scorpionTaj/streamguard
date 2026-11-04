"""Threshold policy & PSI tests (eval/metrics.py, PLAN.md §8, §9)."""

from __future__ import annotations

import numpy as np
import pytest

from streamguard.eval.metrics import precision_at_k, psi, threshold_at_fpr


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

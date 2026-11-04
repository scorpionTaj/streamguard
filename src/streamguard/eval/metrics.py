"""Evaluation metrics (implemented — pure numpy, no sklearn needed for these).

Primary metric: AUPRC (average precision) — see PLAN.md §8; published XGBoost
band on comparable card-fraud data is 0.78–0.88.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    recall: float
    fpr: float


def threshold_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.05
) -> ThresholdResult:
    """Highest-recall operating point with false-positive rate ≤ max_fpr.

    Classification rule: score ≥ threshold → flagged. Thresholds are the
    observed scores themselves, scanned from the lowest up; the last (lowest)
    threshold keeping FPR ≤ max_fpr maximizes recall.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape or y_true.size == 0:
        raise ValueError("y_true and y_score must be same-length non-empty arrays")

    total_neg = int((~y_true).sum())
    total_pos = int(y_true.sum())
    order = np.argsort(y_score, kind="stable")  # ascending
    scores, labels = y_score[order], y_true[order]

    # Ascending scan: FPR starts at 1.0 (everything flagged) and only decreases
    # as the threshold rises, so the first feasible threshold is the
    # highest-recall one.
    for i in range(scores.size):
        flagged = labels[i:]  # score >= scores[i]
        fp = int((~flagged).sum())
        fpr = fp / total_neg if total_neg else 0.0
        if fpr <= max_fpr:
            recall = int(flagged.sum()) / total_pos if total_pos else 0.0
            return ThresholdResult(threshold=float(scores[i]), recall=recall, fpr=fpr)

    # No threshold satisfies the budget (top scorer is a false positive).
    return ThresholdResult(threshold=float(scores[-1]) + 1.0, recall=0.0, fpr=0.0)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between reference and live distributions.

    Quantile-binned on the finite `expected` values; NaN (missing) is its own bin,
    so a shift in the missing share counts as drift. PSI > 0.2 is the retrain flag
    (PLAN.md §8).
    """
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if expected.size == 0 or actual.size == 0:
        raise ValueError("psi requires non-empty distributions")

    finite = expected[np.isfinite(expected)]
    if finite.size:
        edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 2:  # degenerate (constant) reference
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.array([-np.inf, np.inf])

    def proportions(values: np.ndarray) -> np.ndarray:
        missing = np.isnan(values)
        counts = np.histogram(values[~missing], bins=edges)[0]
        return np.append(counts, missing.sum()) / values.size

    e_prop = np.clip(proportions(expected), eps, None)
    a_prop = np.clip(proportions(actual), eps, None)
    return float(np.sum((e_prop - a_prop) * np.log(e_prop / a_prop)))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 50) -> float:
    """Share of true frauds among the k highest-scored events (alert-queue precision)."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape or y_true.size == 0 or k <= 0:
        raise ValueError("need same-length non-empty arrays and k > 0")
    top = np.argsort(-y_score, kind="stable")[: min(k, y_score.size)]
    return float(y_true[top].mean())

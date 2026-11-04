"""Autoencoder anomaly detector — milestone M5 (PLAN.md §9).

Contract: PyTorch MLP 85→64→32→8→32→64→85 with sigmoid output head, trained on
non-fraud train transactions only (standardized, median-imputed + missing-mask
channel), Adam lr=1e-3, batch 512, 100 epochs. `score(x)` returns the
reconstruction MSE; the scorer converts it to a percentile against the
validation-normal calibration histogram stored alongside the model.
"""

from __future__ import annotations


class FraudAutoencoder:
    """Skeleton — the class shape is the contract; implement in M5."""

    def fit(self, x_normal, x_val) -> None:
        raise NotImplementedError("M5: train reconstruction on non-fraud transactions")

    def score(self, x) -> float:
        raise NotImplementedError("M5: reconstruction MSE for one transaction")

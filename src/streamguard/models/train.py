"""Training & evaluation — milestone M1 (XGBoost) / M5 (autoencoder + ensemble).

Contract (PLAN.md §9):
- Temporal split 66.7/16.6/16.7 by TransactionDT quantiles — NEVER shuffle.
- XGBoost champion: objective=binary:logistic, eval_metric=aucpr,
  scale_pos_weight ≈ 27, max_depth 8, eta 0.05, early stopping 50 on val AUPRC.
- Autoencoder (M5): 85→64→32→8→32→64→85, trained on non-fraud train only;
  anomaly score = reconstruction MSE → val percentile.
- Ensemble: score = w·xgb + (1−w)·ae_percentile, w swept on val.
- Log to MLflow: params, metrics, PR curve, features.yaml hash, data hash.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for `make train`. — M1"""
    raise NotImplementedError("M1: temporal split + XGBoost baseline + MLflow logging")


if __name__ == "__main__":
    main()

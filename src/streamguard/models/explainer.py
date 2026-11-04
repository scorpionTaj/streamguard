"""Per-alert explanations (PLAN.md §9): top-5 signed SHAP contributions.

Uses XGBoost's native TreeSHAP (`pred_contribs=True`, the algorithm behind
shap.TreeExplainer) on the served model, in log-odds units. Runs ASYNC in the
scorer — never on the alert publish path.
"""

from __future__ import annotations

import numpy as np

from streamguard.schemas import TopDriver


def top_drivers(booster, matrix: np.ndarray, feature_names: list[str],
                k: int = 5) -> list[list[TopDriver]]:  # fmt: skip
    """For each row: the k features with the largest |SHAP| (bias term excluded)."""
    import xgboost as xgb

    if len(matrix) == 0:
        return []
    dmatrix = xgb.DMatrix(np.asarray(matrix, dtype=np.float32), feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)[:, :-1]  # last column = bias
    result = []
    for row in contribs:
        top = np.argsort(-np.abs(row), kind="stable")[:k]
        result.append([TopDriver(feature=feature_names[i], impact=float(row[i])) for i in top])
    return result

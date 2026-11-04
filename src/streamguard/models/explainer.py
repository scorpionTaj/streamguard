"""Per-alert SHAP explanations — milestone M5 (PLAN.md §9).

Contract: given the champion XGBoost model and one feature vector, return the
top-5 |SHAP| contributions as list[schemas.TopDriver] (signed values). Runs
ASYNC in the scorer — never on the alert publish critical path.
"""

from __future__ import annotations


def top_drivers(model, feature_vector, feature_names: list[str], k: int = 5) -> list:
    """Return top-k signed SHAP contributions for one transaction. — M5"""
    raise NotImplementedError("M5: shap.TreeExplainer → TopDriver list")


async def attach_drivers_async(alert) -> None:
    """Fill `alert.top_drivers` off the publish path. — M5"""
    raise NotImplementedError("M5: async SHAP task")

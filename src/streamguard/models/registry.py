"""MLflow registry helpers — milestone M3 (PLAN.md §10).

Contract: model name `streamguard-scorer` (env MODEL_NAME). Two aliases:
- champion: served by the scorer, drives alerts
- challenger: scored in canary mode, compared but not served (M6)
Promotion/rollback = moving the alias (one command). Every promotion records
score-delta stats from the canary window.
"""

from __future__ import annotations


def load_champion(uri: str):
    """Load the champion model as a pyfunc for the scorer. — M3"""
    raise NotImplementedError("M3: mlflow.pyfunc.load_model via alias")


def promote(challenger_version: str, delta_stats: dict) -> None:
    """Move champion alias to challenger after canary tolerance check. — M6"""
    raise NotImplementedError("M6: alias promotion + canary stats logging")


def rollback(previous_version: str) -> None:
    """Revert champion alias. — M6"""
    raise NotImplementedError("M6: alias revert")

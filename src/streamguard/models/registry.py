"""MLflow registry helpers (PLAN.md §9–10).

Model name `streamguard-scorer` (env MODEL_NAME). Two aliases:
- champion: served by the scorer, drives alerts
- challenger: scored in canary mode, compared but not served (M6)
Promotion/rollback = moving the alias (one command). Every registered version
carries its run's validation thresholds, so a model and its operating point
always travel together.

CLI: python -m streamguard.models.registry set-alias champion <version>
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

log = logging.getLogger("streamguard.registry")

MODEL_NAME = os.getenv("MODEL_NAME", "streamguard-scorer")
CHAMPION, CHALLENGER = "champion", "challenger"


@dataclass(frozen=True)
class Thresholds:
    """Risk-band floors from the version's validation run (PLAN.md §9)."""

    high: float  # FPR ≤ 1 % operating point
    medium: float  # FPR ≤ 5 % operating point (max-recall policy)
    review: float  # 90th-percentile validation score

    @classmethod
    def from_run_metrics(cls, metrics: dict[str, float]) -> Thresholds:
        return cls(
            high=metrics["val_threshold_at_fpr_1pct"],
            medium=metrics["val_threshold_at_fpr_5pct"],
            review=metrics["val_score_p90"],
        )


@dataclass(frozen=True)
class ServingModel:
    model: object  # models.ensemble.HybridModel (anything with .score(matrix) / .booster)
    version: str
    feature_names: list[str]
    thresholds: Thresholds


def _client():
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    return MlflowClient()


def ensure_champion(version: str | int) -> None:
    """Bootstrap: the first registered version becomes champion; later ones wait for M6."""
    from mlflow.exceptions import MlflowException

    client = _client()
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
        log.info("champion stays v%s (new v%s registered)", current.version, version)
    except MlflowException:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, str(version))
        log.info("v%s is now %s", version, CHAMPION)


def load_alias(alias: str = CHAMPION) -> ServingModel:
    """Load the aliased hybrid model (pyfunc) plus its run's thresholds."""
    import mlflow.pyfunc

    client = _client()
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    hybrid = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}").unwrap_python_model().model
    metrics = client.get_run(version.run_id).data.metrics
    return ServingModel(hybrid, str(version.version), list(hybrid.feature_names),
                        Thresholds.from_run_metrics(metrics))  # fmt: skip


def set_alias(alias: str, version: str) -> None:
    _client().set_registered_model_alias(MODEL_NAME, alias, version)
    log.info("%s → v%s", alias, version)


def promote(challenger_version: str, delta_stats: dict) -> None:
    """Move champion alias to challenger after canary tolerance check. — M6"""
    raise NotImplementedError("M6: alias promotion + canary stats logging")


def rollback(previous_version: str) -> None:
    """Revert champion alias. — M6"""
    raise NotImplementedError("M6: alias revert")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="StreamGuard model registry")
    sub = parser.add_subparsers(dest="cmd", required=True)
    alias = sub.add_parser("set-alias", help="point an alias at a registered version")
    alias.add_argument("alias", choices=[CHAMPION, CHALLENGER])
    alias.add_argument("version")
    args = parser.parse_args()
    if args.cmd == "set-alias":
        set_alias(args.alias, args.version)


if __name__ == "__main__":
    main()

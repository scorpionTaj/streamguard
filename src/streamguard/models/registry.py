"""MLflow registry helpers (PLAN.md §9–10).

Model name `streamguard-scorer` (env MODEL_NAME). Two aliases:
- champion: served by the scorer, drives alerts
- challenger: scored in canary mode, compared but not served (M6)
Promotion/rollback = moving the alias (one command). Every registered version
carries its run's validation thresholds, so a model and its operating point
always travel together.

CLI: python -m streamguard.models.registry {promote [--force] | rollback | set-alias <alias> <v>}
(`make promote` / `make rollback`; a running scorer picks the change up within ~30 s.)
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
    """Risk-band floors from the version's validation run (PLAN.md §9).

    Every band is an FPR-bounded validation operating point. The old third band
    (review, 90th-percentile val score ≈ the FPR 7.5 % point) was dropped after
    measurement: 8–9 % precision offline, 0 % live, 2.6 % of all traffic.
    """

    high: float  # FPR ≤ 1 % operating point
    medium: float  # FPR ≤ 5 % operating point (max-recall policy = alert gate)

    @classmethod
    def from_run_metrics(cls, metrics: dict[str, float]) -> Thresholds:
        return cls(
            high=metrics["val_threshold_at_fpr_1pct"],
            medium=metrics["val_threshold_at_fpr_5pct"],
        )


@dataclass(frozen=True)
class ServingModel:
    model: object  # models.ensemble.XgbModel (anything with .score(matrix) / .booster)
    version: str
    feature_names: list[str]
    thresholds: Thresholds
    run_id: str = ""  # training run (drift reference, metrics)


def _client():
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    return MlflowClient()


def _alias_version(client, alias: str) -> str | None:
    from mlflow.exceptions import MlflowException

    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
    except MlflowException:
        return None


def assign_new_version(version: str | int) -> str:
    """A fresh version becomes champion if there is none, else challenger (canary first)."""
    client = _client()
    alias = CHALLENGER if _alias_version(client, CHAMPION) else CHAMPION
    client.set_registered_model_alias(MODEL_NAME, alias, str(version))
    log.info("v%s is now %s", version, alias)
    return alias


def load_alias(alias: str = CHAMPION) -> ServingModel:
    """Load the aliased hybrid model (pyfunc) plus its run's thresholds."""
    import mlflow.pyfunc

    client = _client()
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    hybrid = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}").unwrap_python_model().model
    metrics = client.get_run(version.run_id).data.metrics
    return ServingModel(hybrid, str(version.version), list(hybrid.feature_names),
                        Thresholds.from_run_metrics(metrics), version.run_id)  # fmt: skip


def alias_versions() -> tuple[str | None, str | None]:
    """(champion, challenger) versions currently set in the registry — cheap to poll."""
    client = _client()
    return _alias_version(client, CHAMPION), _alias_version(client, CHALLENGER)


def load_optional(alias: str) -> ServingModel | None:
    """The aliased model, or None when the alias isn't set (e.g. no challenger)."""
    from mlflow.exceptions import MlflowException

    try:
        return load_alias(alias)
    except MlflowException:
        return None


def set_alias(alias: str, version: str) -> None:
    _client().set_registered_model_alias(MODEL_NAME, alias, version)
    log.info("%s → v%s", alias, version)


def promote(canary: dict, force: bool = False) -> str:
    """challenger → champion if the canary is within tolerance (serving/canary.py).

    The old champion is remembered on the new one (tag `previous_champion`) so
    `rollback()` is one alias move; the canary stats are recorded as version tags.
    """
    client = _client()
    champion, challenger = _alias_version(client, CHAMPION), _alias_version(client, CHALLENGER)
    if challenger is None:
        raise SystemExit("no challenger registered — `make train` creates one")
    if canary.get("challenger_version") not in (None, challenger):
        raise SystemExit(
            f"canary measured v{canary['challenger_version']}, challenger is v{challenger}"
        )
    if not (canary.get("within_tolerance") or force):
        raise SystemExit(f"canary not within tolerance: {canary}")
    for key in ("events", "p95_abs_delta", "mean_abs_delta", "alert_agreement"):
        if canary.get(key) is not None:
            client.set_model_version_tag(MODEL_NAME, challenger, f"canary_{key}", str(canary[key]))
    if champion:
        client.set_model_version_tag(MODEL_NAME, challenger, "previous_champion", champion)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, challenger)
    client.delete_registered_model_alias(MODEL_NAME, CHALLENGER)
    log.info("promoted v%s to champion (was v%s)", challenger, champion)
    return challenger


def rollback() -> str:
    """Point champion back at the version it replaced (one command, PLAN.md §9)."""
    client = _client()
    current = _alias_version(client, CHAMPION)
    if current is None:
        raise SystemExit("no champion to roll back")
    previous = client.get_model_version(MODEL_NAME, current).tags.get("previous_champion")
    if previous is None:
        raise SystemExit(f"v{current} has no previous_champion tag — nothing to roll back to")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, previous)
    log.info("rolled back champion v%s → v%s", current, previous)
    return previous


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="StreamGuard model registry")
    sub = parser.add_subparsers(dest="cmd", required=True)
    alias = sub.add_parser("set-alias", help="point an alias at a registered version")
    alias.add_argument("alias", choices=[CHAMPION, CHALLENGER])
    alias.add_argument("version")
    prom = sub.add_parser("promote", help="challenger → champion if the live canary passes")
    prom.add_argument("--api", default=os.getenv("SCORER_API", "http://localhost:8000"))
    prom.add_argument("--force", action="store_true", help="skip the tolerance check")
    sub.add_parser("rollback", help="champion → the version it replaced")
    args = parser.parse_args()
    if args.cmd == "set-alias":
        set_alias(args.alias, args.version)
    elif args.cmd == "promote":
        import httpx

        canary = httpx.get(f"{args.api}/canary", timeout=5).json()
        print(f"canary: {canary}")
        promote(canary, force=args.force)
    elif args.cmd == "rollback":
        rollback()


if __name__ == "__main__":
    main()

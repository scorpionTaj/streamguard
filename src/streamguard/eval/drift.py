"""PSI drift monitor (PLAN.md §8, §10).

Reference: a 10k-row sample of the TRAIN feature matrix, logged with every
training run (`drift_reference.npz`). Live: the scorer keeps the most recent
feature rows (DriftMonitor) and recomputes PSI per feature periodically →
Prometheus gauge `streamguard_feature_psi{feature}`. Any of the model's top-20
(gain) features with PSI > 0.2 raises the retrain flag.

Time compression note: at 200 tx/s and TIME_COMPRESSION 8800, the 20k-event
window ≈ 100 s real ≈ 10 simulated days — the plan's "hourly vs last 24h" in
simulated time is far shorter than one refresh, so the window is defined in events.

`make drift`: print the scorer's live report and, if flagged, tag the champion
version in MLflow (drift_flag / drift_features).
"""

from __future__ import annotations

import tempfile
import threading
from collections import deque
from pathlib import Path

import numpy as np
from prometheus_client import Gauge

from streamguard.eval.metrics import psi

PSI_FLAG = 0.2
TOP_K = 20
REFERENCE_ROWS = 10_000
REFERENCE_FILE = "drift_reference.npz"

FEATURE_PSI = Gauge("streamguard_feature_psi", "PSI vs training reference", ["feature"])
DRIFT_FLAG = Gauge("streamguard_drift_flag", "1 if a top-20 feature has PSI > 0.2")


def reference_sample(x_train: np.ndarray, rows: int = REFERENCE_ROWS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    take = rng.choice(len(x_train), min(rows, len(x_train)), replace=False)
    return np.asarray(x_train[np.sort(take)], dtype=np.float32)


def log_reference(run_id: str, x_train: np.ndarray, feature_names: list[str]) -> None:
    """Attach the drift reference to a training run (also used to backfill older runs)."""
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / REFERENCE_FILE
        np.savez_compressed(path, sample=reference_sample(x_train), names=np.array(feature_names))
        mlflow.MlflowClient().log_artifact(run_id, str(path))


def load_reference(run_id: str) -> tuple[np.ndarray, list[str]] | None:
    import mlflow

    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=REFERENCE_FILE)
    except Exception:  # older runs have no reference → drift monitoring disabled
        return None
    data = np.load(path)
    return data["sample"], data["names"].tolist()


def top_features(booster, k: int = TOP_K) -> list[str]:
    gain = booster.get_score(importance_type="gain")
    return sorted(gain, key=gain.get, reverse=True)[:k]


class DriftMonitor:
    """Rolling window of live feature rows → PSI per feature vs the training reference."""

    def __init__(self, reference: np.ndarray, names: list[str], important: list[str],
                 window: int = 20_000, min_rows: int = 2_000) -> None:  # fmt: skip
        self.reference, self.names, self.important = reference, names, set(important)
        self.rows: deque = deque(maxlen=window)
        self.min_rows = min_rows
        self.lock = threading.Lock()
        self.latest: dict = {"rows": 0, "features": {}, "flagged": [], "drift_flag": False}

    def add(self, matrix: np.ndarray) -> None:
        with self.lock:
            self.rows.extend(np.asarray(matrix, dtype=np.float32))

    def compute(self) -> dict:
        with self.lock:
            live = np.array(self.rows)
        if len(live) < self.min_rows:
            return self.latest
        scores = {name: psi(self.reference[:, i], live[:, i]) for i, name in enumerate(self.names)}
        flagged = sorted((n for n in self.important if scores.get(n, 0.0) > PSI_FLAG),
                         key=lambda n: -scores[n])  # fmt: skip
        for name, value in scores.items():
            FEATURE_PSI.labels(name).set(value)
        DRIFT_FLAG.set(1.0 if flagged else 0.0)
        self.latest = {
            "rows": int(len(live)),
            "features": dict(sorted(scores.items(), key=lambda kv: -kv[1])),
            "important": sorted(self.important),
            "flagged": flagged,
            "drift_flag": bool(flagged),
        }
        return self.latest


def main() -> None:
    """Entry point for `make drift`: live PSI report from the scorer + MLflow retrain flag."""
    import argparse
    import os

    import httpx
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="PSI drift report vs training reference")
    parser.add_argument("--api", default=os.getenv("SCORER_API", "http://localhost:8000"))
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    report = httpx.get(f"{args.api}/drift", timeout=30).json()
    if not report.get("features"):
        raise SystemExit(f"no drift data yet ({report.get('rows', 0)} live rows; needs ≥ 2000)")
    important = set(report.get("important", []))
    print(
        f"PSI over the last {report['rows']:,} scored events (flag > {PSI_FLAG}, ★ = top-{TOP_K})"
    )
    for name, value in list(report["features"].items())[: args.top]:
        mark = "★" if name in important else " "
        alarm = "  ← DRIFT" if value > PSI_FLAG and name in important else ""
        print(f"  {mark} {name:<26} {value:8.4f}{alarm}")

    from streamguard.models.registry import CHAMPION, MODEL_NAME, _client

    client = _client()
    version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    client.set_model_version_tag(MODEL_NAME, version, "drift_flag", str(report["drift_flag"]))
    client.set_model_version_tag(MODEL_NAME, version, "drift_features",
                                 ",".join(report["flagged"]) or "-")  # fmt: skip
    verdict = f"RETRAIN FLAG: {report['flagged']}" if report["drift_flag"] else "no drift flag"
    print(f"{verdict} → tagged champion v{version} in MLflow")


if __name__ == "__main__":
    main()

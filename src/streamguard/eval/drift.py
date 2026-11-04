"""PSI drift monitor (PLAN.md §8, §10).

Reference: a 10k-row sample of the TRAIN feature matrix plus per-feature PSI
thresholds, logged with every training run (`drift_reference.npz`).

Calibration: a fixed PSI > 0.2 rule false-alarmed on 18 % of 20k-event windows
INSIDE the training period (several top features swing to PSI ≈ 0.47 week to
week). So each feature's threshold = max(0.2, the largest PSI any training
window of the live window's size shows vs the reference): live drift is flagged
only when a feature looks worse than anything seen during training.

Live: the scorer keeps the last `window` feature rows (DriftMonitor); once the
window is full it recomputes PSI periodically → `streamguard_feature_psi` and
`streamguard_feature_drift_ratio` (PSI / threshold, ≥ 1 = drifted). Any top-20
(gain) feature with ratio > 1 raises the retrain flag.

Time compression note: at 200 tx/s and TIME_COMPRESSION 8800, the 20k-event
window ≈ 100 s real ≈ 10 simulated days, so the window is defined in events.

`make drift`: print the scorer's live report and, if flagged, tag the champion
version in MLflow (drift_flag / drift_features).
"""

from __future__ import annotations

import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from prometheus_client import Gauge

from streamguard.eval.metrics import psi

PSI_FLAG = 0.2  # floor for every calibrated threshold
TOP_K = 20
REFERENCE_ROWS = 10_000
WINDOW = 20_000  # live window = calibration window (events)
STRIDE = 10_000
REFERENCE_FILE = "drift_reference.npz"

FEATURE_PSI = Gauge("streamguard_feature_psi", "PSI vs training reference", ["feature"])
DRIFT_RATIO = Gauge("streamguard_feature_drift_ratio",
                    "PSI / calibrated threshold (≥ 1 = drifted)", ["feature"])  # fmt: skip
DRIFT_FLAG = Gauge("streamguard_drift_flag", "1 if a top-20 feature exceeds its threshold")


@dataclass(frozen=True)
class DriftReference:
    sample: np.ndarray  # reference rows (train sample)
    names: list[str]
    thresholds: np.ndarray  # per-feature PSI threshold
    window: int  # live window size the thresholds were calibrated for


def reference_sample(x_train: np.ndarray, rows: int = REFERENCE_ROWS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    take = rng.choice(len(x_train), min(rows, len(x_train)), replace=False)
    return np.asarray(x_train[np.sort(take)], dtype=np.float32)


def calibrate_thresholds(reference: np.ndarray, x_ordered: np.ndarray, window: int = WINDOW,
                         stride: int = STRIDE, floor: float = PSI_FLAG) -> np.ndarray:  # fmt: skip
    """Per-feature threshold = max(floor, largest PSI of any `window`-row slice of the
    time-ordered training matrix vs the reference) — the no-drift noise ceiling."""
    starts = range(0, max(1, len(x_ordered) - window + 1), stride)
    worst = np.zeros(reference.shape[1])
    for start in starts:
        chunk = x_ordered[start : start + window]
        for i in range(reference.shape[1]):
            worst[i] = max(worst[i], psi(reference[:, i], chunk[:, i]))
    return np.maximum(floor, worst)


def log_reference(run_id: str, x_train: np.ndarray, feature_names: list[str]) -> None:
    """Attach reference sample + calibrated thresholds to a training run (also backfills).

    `x_train` must be in event-time order (training windows mimic live windows).
    """
    import mlflow

    sample = reference_sample(x_train)
    thresholds = calibrate_thresholds(sample, x_train)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / REFERENCE_FILE
        np.savez_compressed(path, sample=sample, names=np.array(feature_names),
                            thresholds=thresholds, window=np.array(WINDOW))  # fmt: skip
        mlflow.MlflowClient().log_artifact(run_id, str(path))


def load_reference(run_id: str) -> DriftReference | None:
    import mlflow

    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=REFERENCE_FILE)
    except Exception:  # older runs have no reference → drift monitoring disabled
        return None
    data = np.load(path)
    names = data["names"].tolist()
    if "thresholds" not in data:  # pre-calibration artifact: fixed floor
        return DriftReference(data["sample"], names, np.full(len(names), PSI_FLAG), WINDOW)
    return DriftReference(data["sample"], names, data["thresholds"], int(data["window"]))


def top_features(booster, k: int = TOP_K) -> list[str]:
    gain = booster.get_score(importance_type="gain")
    return sorted(gain, key=gain.get, reverse=True)[:k]


class DriftMonitor:
    """Rolling window of live feature rows → PSI per feature vs the calibrated reference.

    PSI is only computed on a FULL window: thresholds were calibrated at that size,
    and smaller samples read noisier (the reason a fixed cut-off false-alarmed).
    """

    def __init__(self, reference: DriftReference, important: list[str]) -> None:
        self.ref, self.important = reference, set(important)
        self.thresholds = dict(zip(reference.names, reference.thresholds.tolist(), strict=True))
        self.rows: deque = deque(maxlen=reference.window)
        self.lock = threading.Lock()
        self.latest: dict = {"rows": 0, "window": reference.window, "features": {},
                             "thresholds": self.thresholds, "flagged": [],
                             "drift_flag": False}  # fmt: skip

    def add(self, matrix: np.ndarray) -> None:
        with self.lock:
            self.rows.extend(np.asarray(matrix, dtype=np.float32))

    def compute(self) -> dict:
        with self.lock:
            live = np.array(self.rows)
        if len(live) < self.ref.window:
            return self.latest | {"rows": int(len(live))}
        scores = {n: psi(self.ref.sample[:, i], live[:, i]) for i, n in enumerate(self.ref.names)}
        ratio = {n: scores[n] / self.thresholds[n] for n in scores}
        flagged = sorted((n for n in self.important if ratio.get(n, 0.0) > 1.0),
                         key=lambda n: -ratio[n])  # fmt: skip
        for name in scores:
            FEATURE_PSI.labels(name).set(scores[name])
            DRIFT_RATIO.labels(name).set(ratio[name])
        DRIFT_FLAG.set(1.0 if flagged else 0.0)
        self.latest = {
            "rows": int(len(live)),
            "window": self.ref.window,
            "features": dict(sorted(scores.items(), key=lambda kv: -ratio[kv[0]])),
            "thresholds": self.thresholds,
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
        raise SystemExit(f"no drift data yet: {report.get('rows', 0):,} / "
                         f"{report.get('window', WINDOW):,} live rows (needs a full window)")  # fmt: skip
    important, thresholds = set(report.get("important", [])), report["thresholds"]
    print(f"PSI over the last {report['rows']:,} scored events vs calibrated per-feature "
          f"thresholds (★ = top-{TOP_K} by gain)")  # fmt: skip
    print(f"    {'feature':<26} {'PSI':>8} {'threshold':>10} {'ratio':>7}")
    for name, value in list(report["features"].items())[: args.top]:
        mark = "★" if name in important else " "
        ratio = value / thresholds[name]
        alarm = "  ← DRIFT" if ratio > 1.0 and name in important else ""
        print(f"  {mark} {name:<26} {value:8.4f} {thresholds[name]:10.4f} {ratio:7.2f}{alarm}")

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

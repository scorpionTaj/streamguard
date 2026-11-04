"""Training & evaluation — milestone M1 (XGBoost) / M5 (autoencoder + ensemble).

Contract (PLAN.md §9):
- Temporal split 66.7/16.6/16.7 by event time — NEVER shuffle.
- XGBoost champion: objective=binary:logistic, eval_metric=aucpr,
  scale_pos_weight = neg/pos on train (≈ 27), max_depth 8, eta 0.05,
  early stopping 50 on val AUPRC.
- Autoencoder (M5): 85→64→32→8→32→64→85, trained on non-fraud train only;
  anomaly score = reconstruction MSE → val percentile.
- Ensemble: score = w·xgb + (1−w)·ae_percentile, w swept on val.
- Log to MLflow: params, metrics, PR curve, features.yaml hash, data hash.

The test split is reserved for the single final evaluation (M5) — not scored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from streamguard.config import REPO_ROOT
from streamguard.eval.metrics import precision_at_k, threshold_at_fpr
from streamguard.models.registry import MODEL_NAME, ensure_champion
from streamguard.offline.features import build_training_frame, feature_columns, temporal_split

log = logging.getLogger("streamguard.train")

FEATURES_YAML = REPO_ROOT / "configs" / "features.yaml"
TRANSACTION_CSV = REPO_ROOT / "data" / "train_transaction.csv"
EXPERIMENT = "streamguard"

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 8,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "tree_method": "hist",
    "seed": 42,
}
MAX_ROUNDS = 2000
EARLY_STOPPING = 50


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """PLAN.md §8 metric set: AUPRC (primary), ROC-AUC, recall @ FPR 1/5 %, precision@50."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    at_1, at_5 = threshold_at_fpr(y_true, y_score, 0.01), threshold_at_fpr(y_true, y_score, 0.05)
    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall_at_fpr_1pct": at_1.recall,
        "recall_at_fpr_5pct": at_5.recall,
        "threshold_at_fpr_1pct": at_1.threshold,
        "threshold_at_fpr_5pct": at_5.threshold,
        "score_p90": float(np.quantile(y_score, 0.90)),  # review-band floor (PLAN.md §9)
        "precision_at_50": precision_at_k(y_true, y_score, 50),
    }


def _pr_curve_png(y_true: np.ndarray, y_score: np.ndarray, auprc: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"XGBoost (AUPRC {auprc:.3f})")
    ax.axhline(y_true.mean(), ls="--", c="grey", label=f"base rate {y_true.mean():.3f}")
    ax.set(xlabel="recall", ylabel="precision", title="PR curve — validation split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_xgboost(limit: int = 0) -> dict[str, float]:
    import mlflow
    import xgboost as xgb

    features = build_training_frame(limit)
    columns = feature_columns(features)
    splits = temporal_split(features)
    train, val = splits["train"], splits["val"]
    x_train, y_train = train.select(columns).to_numpy(), train["is_fraud"].to_numpy()
    x_val, y_val = val.select(columns).to_numpy(), val["is_fraud"].to_numpy()
    log.info("train=%d val=%d features=%d", len(y_train), len(y_val), len(columns))

    params = XGB_PARAMS | {"scale_pos_weight": float((y_train == 0).sum() / y_train.sum())}
    dtrain = xgb.DMatrix(x_train, y_train, feature_names=columns)
    dval = xgb.DMatrix(x_val, y_val, feature_names=columns)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="xgboost-baseline") as run:
        booster = xgb.train(
            params, dtrain, MAX_ROUNDS, evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOPPING, verbose_eval=100,
        )  # fmt: skip
        p_val = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        metrics = {f"val_{k}": v for k, v in evaluate(y_val, p_val).items()}

        mlflow.log_params(params | {"max_rounds": MAX_ROUNDS, "early_stopping": EARLY_STOPPING,
                                    "best_iteration": booster.best_iteration,
                                    "n_features": len(columns), "n_train": len(y_train),
                                    "n_val": len(y_val), "limit": limit})  # fmt: skip
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "model_type": "xgboost",
            "features_yaml_sha256": sha256_file(FEATURES_YAML),
            "data_sha256": sha256_file(TRANSACTION_CSV),
        })  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            curve = Path(tmp) / "pr_curve_val.png"
            _pr_curve_png(y_val, p_val, metrics["val_auprc"], curve)
            (Path(tmp) / "feature_columns.json").write_text(json.dumps(columns))
            mlflow.log_artifacts(tmp)
        mlflow.log_artifact(str(FEATURES_YAML))
        info = mlflow.xgboost.log_model(booster, name="model", registered_model_name=MODEL_NAME)
        log.info("run %s: %s", run.info.run_id, json.dumps(metrics, indent=2))
    ensure_champion(info.registered_model_version)
    return metrics


def main() -> None:
    """Entry point for `make train`."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Train XGBoost baseline → MLflow")
    parser.add_argument("--limit", type=int, default=0, help="rows to use (0 = all; smoke runs)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    train_xgboost(args.limit)


if __name__ == "__main__":
    main()

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

Selection (early stopping, AE calibration, w, band thresholds) uses val only;
the temporal test split is evaluated once per run and only reported.
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
from streamguard.models.registry import MODEL_NAME, assign_new_version
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


def _pr_curves_png(y_true: np.ndarray, curves: dict[str, np.ndarray], title: str,
                   path: Path) -> None:  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(5, 4))
    for label, score in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax.plot(
            recall, precision, label=f"{label} (AUPRC {average_precision_score(y_true, score):.3f})"
        )
    ax.axhline(y_true.mean(), ls="--", c="grey", label=f"base rate {y_true.mean():.3f}")
    ax.set(xlabel="recall", ylabel="precision", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_hybrid(limit: int = 0, ae_epochs: int = 100, seed: int = 42) -> dict[str, float]:
    """XGBoost + autoencoder → w sweep → thresholds (all on val) → one test evaluation.

    Every choice (early stopping, AE calibration, w, band thresholds) uses the
    validation split only; test metrics are reported, never used to decide.
    """
    import mlflow
    import xgboost as xgb

    from streamguard.eval.drift import log_reference
    from streamguard.models.autoencoder import (
        Preprocessor,
        calibrated_percentile,
        reconstruction_error,
        train_autoencoder,
    )
    from streamguard.models.ensemble import HybridModel, pyfunc_model, sweep_weight

    features = build_training_frame(limit)
    columns = feature_columns(features)
    splits = temporal_split(features)
    x = {name: part.select(columns).to_numpy() for name, part in splits.items()}
    y = {name: part["is_fraud"].to_numpy() for name, part in splits.items()}
    log.info("train=%d val=%d test=%d features=%d",
             len(y["train"]), len(y["val"]), len(y["test"]), len(columns))  # fmt: skip

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="hybrid-xgb-ae") as run:
        # ── XGBoost lane ──────────────────────────────────────────────────────
        params = XGB_PARAMS | {
            "seed": seed,
            "scale_pos_weight": float((y["train"] == 0).sum() / y["train"].sum()),
        }
        dtrain = xgb.DMatrix(x["train"], y["train"], feature_names=columns)
        dval = xgb.DMatrix(x["val"], y["val"], feature_names=columns)
        booster = xgb.train(
            params, dtrain, MAX_ROUNDS, evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOPPING, verbose_eval=100,
        )  # fmt: skip
        booster = booster[: booster.best_iteration + 1]  # serve the early-stopped model

        # ── Autoencoder lane (non-fraud train only, calibrated on val-normal) ─
        normal = x["train"][y["train"] == 0]
        pre = Preprocessor.fit(normal)
        net, losses = train_autoencoder(pre.transform(normal), epochs=ae_epochs, seed=seed)
        val_errors = reconstruction_error(net, pre.transform(x["val"]))
        reference = np.sort(val_errors[y["val"] == 0])

        def lanes(split: str) -> tuple[np.ndarray, np.ndarray]:
            xgb_prob = np.asarray(booster.inplace_predict(x[split].astype(np.float32)), float)
            errors = (
                val_errors if split == "val" else reconstruction_error(net, pre.transform(x[split]))
            )
            return xgb_prob, calibrated_percentile(errors, reference)

        # ── Ensemble weight + thresholds on val; one final test evaluation ───
        xgb_val, ae_val = lanes("val")
        weight, sweep = sweep_weight(y["val"], xgb_val, ae_val)
        model = HybridModel(booster, net, pre, reference, weight, columns)
        metrics: dict[str, float] = {}
        for split in ("val", "test"):
            xgb_prob, ae_pct = (xgb_val, ae_val) if split == "val" else lanes(split)
            score = model.components(x[split])["score"]
            metrics |= {f"{split}_{k}": v for k, v in evaluate(y[split], score).items()}
            metrics |= {f"{split}_xgb_{k}": v for k, v in evaluate(y[split], xgb_prob).items()
                        if k in ("auprc", "roc_auc", "recall_at_fpr_5pct")}  # fmt: skip
            metrics |= {f"{split}_ae_{k}": v for k, v in evaluate(y[split], ae_pct).items()
                        if k in ("auprc", "roc_auc", "recall_at_fpr_5pct")}  # fmt: skip
            if split == "test":
                curves = {"hybrid": score, "xgboost": xgb_prob, "autoencoder": ae_pct}
        metrics |= {f"sweep_auprc_w{w:g}": v for w, v in sweep.items()}
        metrics["ensemble_weight"] = weight

        mlflow.log_params(params | {
            "max_rounds": MAX_ROUNDS, "early_stopping": EARLY_STOPPING,
            "best_iteration": booster.num_boosted_rounds() - 1, "ae_epochs": ae_epochs,
            "ae_hidden": "64-32-8-32-64", "n_features": len(columns), "limit": limit,
            "n_train": len(y["train"]), "n_val": len(y["val"]), "n_test": len(y["test"]),
        })  # fmt: skip
        mlflow.log_metrics(metrics)
        for epoch, loss in enumerate(losses):
            mlflow.log_metric("ae_train_loss", loss, step=epoch)
        mlflow.set_tags({
            "model_type": "hybrid-xgboost-autoencoder",
            "features_yaml_sha256": sha256_file(FEATURES_YAML),
            "data_sha256": sha256_file(TRANSACTION_CSV),
        })  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            _pr_curves_png(y["test"], curves, "PR curves — temporal test split",
                           Path(tmp) / "pr_curve_test.png")  # fmt: skip
            (Path(tmp) / "feature_columns.json").write_text(json.dumps(columns))
            mlflow.log_artifacts(tmp)
            model.save(Path(tmp) / "hybrid")
            mlflow.log_artifact(str(FEATURES_YAML))
            log_reference(run.info.run_id, x["train"], columns)  # PSI reference (eval/drift.py)
            info = mlflow.pyfunc.log_model(
                name="model", python_model=pyfunc_model(),
                artifacts={"hybrid": str(Path(tmp) / "hybrid")},
                registered_model_name=MODEL_NAME,
            )  # fmt: skip
        log.info("run %s: %s", run.info.run_id, json.dumps(metrics, indent=2))
    assign_new_version(info.registered_model_version)
    return metrics


def main() -> None:
    """Entry point for `make train`."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Train hybrid XGBoost + autoencoder → MLflow")
    parser.add_argument("--limit", type=int, default=0, help="rows to use (0 = all; smoke runs)")
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="XGBoost + AE seed (retrain variants)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    train_hybrid(args.limit, args.ae_epochs, args.seed)


if __name__ == "__main__":
    main()

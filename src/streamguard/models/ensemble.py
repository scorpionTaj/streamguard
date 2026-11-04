"""Hybrid model = XGBoost champion lane + autoencoder anomaly lane (PLAN.md §9).

score = w · xgb_prob + (1 − w) · ae_percentile, w swept on validation AUPRC.
Packaged as ONE MLflow pyfunc (booster + AE weights + preprocessing + calibration
+ w + feature order), so a registry version is everything the scorer needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from streamguard.models.autoencoder import (
    Preprocessor,
    build_network,
    calibrated_percentile,
    import_torch,
    reconstruction_error,
)

ENSEMBLE_WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)


def blend(xgb_prob: np.ndarray, ae_pct: np.ndarray, weight: float) -> np.ndarray:
    return weight * xgb_prob + (1.0 - weight) * ae_pct


def sweep_weight(y: np.ndarray, xgb_prob: np.ndarray, ae_pct: np.ndarray,
                 weights=ENSEMBLE_WEIGHTS) -> tuple[float, dict[float, float]]:  # fmt: skip
    """Weight maximizing validation AUPRC (ties → the larger, simpler XGBoost-heavy w)."""
    from sklearn.metrics import average_precision_score

    scores = {w: float(average_precision_score(y, blend(xgb_prob, ae_pct, w))) for w in weights}
    best = max(scores, key=lambda w: (scores[w], w))
    return best, scores


@dataclass
class HybridModel:
    booster: object  # xgboost.Booster
    net: object  # torch.nn.Module
    preprocessor: Preprocessor
    reference: np.ndarray  # sorted validation-normal reconstruction errors
    weight: float
    feature_names: list[str]

    def components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        xgb_prob = np.asarray(self.booster.inplace_predict(x), dtype=float)
        if self.weight >= 1.0:  # AE lane disabled by the sweep → skip its cost
            ae_pct = np.zeros_like(xgb_prob)
        else:
            errors = reconstruction_error(self.net, self.preprocessor.transform(x))
            ae_pct = calibrated_percentile(errors, self.reference)
        return {"xgb": xgb_prob, "ae": ae_pct, "score": blend(xgb_prob, ae_pct, self.weight)}

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.components(x)["score"]

    def save(self, path: Path) -> None:
        torch = import_torch()

        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path / "xgb.json")
        torch.save(self.net.state_dict(), path / "ae.pt")
        np.savez(
            path / "calibration.npz", reference=self.reference, **self.preprocessor.to_arrays()
        )
        meta = {"weight": self.weight, "feature_names": self.feature_names}
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> HybridModel:
        torch = import_torch()
        import xgboost as xgb

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        booster = xgb.Booster()
        booster.load_model(path / "xgb.json")
        arrays = np.load(path / "calibration.npz")
        pre = Preprocessor(median=arrays["median"], lo=arrays["lo"], hi=arrays["hi"])
        net = build_network(2 * len(meta["feature_names"]))
        net.load_state_dict(torch.load(path / "ae.pt", weights_only=True))
        return cls(booster, net.eval(), pre, arrays["reference"], meta["weight"],
                   meta["feature_names"])  # fmt: skip


def pyfunc_model():
    """MLflow wrapper; imported lazily so the module works without mlflow."""
    import mlflow.pyfunc

    class HybridPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context) -> None:
            self.model = HybridModel.load(context.artifacts["hybrid"])

        def predict(self, context, model_input, params=None):
            return self.model.score(model_input[self.model.feature_names].to_numpy(float))

    return HybridPyfunc()

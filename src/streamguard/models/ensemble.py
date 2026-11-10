"""Served model = XGBoost booster packaged with its feature order (PLAN.md §9).

The autoencoder lane was measured and cut *(post-v0.1.0)*: its own val AUPRC was
0.134 (0.064 on the 13 velocity features alone) and every blend scored below
XGBoost alone, so the weight sweep always picked w = 1.0. Its code and the torch
dependency are gone; the measurements are in docs/WRITEUP.md.

The module keeps its name because registered MLflow versions import `XgbModel`
from this path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class XgbModel:
    booster: object  # xgboost.Booster
    feature_names: list[str]

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.booster.inplace_predict(np.asarray(x, dtype=np.float32)), dtype=float
        )  # fmt: skip

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path / "xgb.json")
        (path / "meta.json").write_text(json.dumps({"feature_names": self.feature_names},
                                                   indent=2))  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> XgbModel:
        import xgboost as xgb

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        booster = xgb.Booster()
        booster.load_model(path / "xgb.json")
        return cls(booster, meta["feature_names"])


def pyfunc_model():
    """MLflow wrapper; imported lazily so the module works without mlflow."""
    import mlflow.pyfunc

    class XgbPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context) -> None:
            self.model = XgbModel.load(context.artifacts["xgb"])

        def predict(self, context, model_input, params=None):
            return self.model.score(model_input[self.model.feature_names].to_numpy(float))

    return XgbPyfunc()

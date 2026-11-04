"""Model tests: AE preprocessing/calibration (reference lane), packaging, SHAP (PLAN.md §9)."""

from __future__ import annotations

import numpy as np
import pytest

from streamguard.models.autoencoder import (
    Preprocessor,
    calibrated_percentile,
    reconstruction_error,
    train_autoencoder,
)
from streamguard.models.ensemble import XgbModel
from streamguard.models.explainer import top_drivers

xgb = pytest.importorskip("xgboost")


class TestPreprocessor:
    def test_scales_to_unit_range_and_masks_nan(self):
        x = np.array([[0.0, np.nan], [10.0, 5.0], [5.0, 5.0], [100.0, 5.0]])
        pre = Preprocessor.fit(x)
        z = pre.transform(np.array([[1e6, np.nan], [-1e6, 5.0]]))
        assert z.shape == (2, 4) and z.dtype == np.float32
        assert z[0, 0] == 1.0 and z[1, 0] == 0.0  # clipped to the 1st–99th percentile
        assert z[0, 3] == 1.0 and z[1, 3] == 0.0  # mask channel
        assert np.all((z >= 0) & (z <= 1))

    def test_all_nan_and_constant_columns_are_safe(self):
        x = np.array([[np.nan, 3.0], [np.nan, 3.0]])
        z = Preprocessor.fit(x).transform(x)
        assert np.isfinite(z).all()


class TestCalibration:
    def test_percentile_against_reference(self):
        reference = np.sort(np.array([0.1, 0.2, 0.3, 0.4]))
        pct = calibrated_percentile(np.array([0.05, 0.2, 0.35, 9.0]), reference)
        assert pct.tolist() == [0.0, 0.5, 0.75, 1.0]


class TestAeLane:
    """The measured-and-cut M5 lane (models/autoencoder.py) still trains and scores."""

    def test_autoencoder_loss_decreases(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 4)).astype(np.float32)
        pre = Preprocessor.fit(x)
        net, losses = train_autoencoder(pre.transform(x), epochs=3, log_every=0)
        assert losses[-1] < losses[0]
        errors = reconstruction_error(net, pre.transform(x[:5]))
        assert errors.shape == (5,) and np.all(errors >= 0)


@pytest.fixture(scope="module")
def tiny_model():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=400) > 1.0).astype(int)
    names = ["f0", "f1", "f2", "f3"]
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 3},
                        xgb.DMatrix(x, y, feature_names=names), 20)  # fmt: skip
    return XgbModel(booster, names), x


class TestXgbModel:
    def test_scores_are_probabilities(self, tiny_model):
        model, x = tiny_model
        scores = model.score(x[:5])
        assert scores.shape == (5,) and np.all((scores >= 0) & (scores <= 1))

    def test_save_load_round_trip(self, tiny_model, tmp_path):
        model, x = tiny_model
        model.save(tmp_path / "xgb")
        loaded = XgbModel.load(tmp_path / "xgb")
        np.testing.assert_allclose(loaded.score(x), model.score(x), atol=1e-6)
        assert loaded.feature_names == model.feature_names


class TestExplainer:
    def test_top_drivers_sum_to_margin_and_rank_by_magnitude(self, tiny_model):
        model, x = tiny_model
        drivers = top_drivers(model.booster, x[:3], model.feature_names, k=4)
        assert len(drivers) == 3 and all(len(d) == 4 for d in drivers)
        impacts = [abs(d.impact) for d in drivers[0]]
        assert impacts == sorted(impacts, reverse=True)
        # TreeSHAP is additive: contributions + bias = raw margin
        dm = xgb.DMatrix(x[:1], feature_names=model.feature_names)
        margin = model.booster.predict(dm, output_margin=True)[0]
        bias = model.booster.predict(dm, pred_contribs=True)[0, -1]
        assert sum(d.impact for d in drivers[0]) + bias == pytest.approx(margin, abs=1e-4)
        assert drivers[0][0].feature == "f0"  # the label depends on f0

    def test_empty_input(self, tiny_model):
        assert top_drivers(tiny_model[0].booster, np.empty((0, 4)), ["a"] * 4) == []

"""Hybrid model tests: AE preprocessing/calibration, ensemble sweep, packaging, SHAP (PLAN.md §9)."""

from __future__ import annotations

import numpy as np
import pytest

from streamguard.models.autoencoder import (
    Preprocessor,
    calibrated_percentile,
    reconstruction_error,
    train_autoencoder,
)
from streamguard.models.ensemble import HybridModel, blend, sweep_weight
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


class TestCalibrationAndBlend:
    def test_percentile_against_reference(self):
        reference = np.sort(np.array([0.1, 0.2, 0.3, 0.4]))
        pct = calibrated_percentile(np.array([0.05, 0.2, 0.35, 9.0]), reference)
        assert pct.tolist() == [0.0, 0.5, 0.75, 1.0]

    def test_sweep_picks_best_and_prefers_xgb_heavy_on_ties(self):
        y = np.array([0, 0, 1, 1])
        useless_ae = np.array([0.9, 0.9, 0.1, 0.1])
        best, scores = sweep_weight(y, np.array([0.1, 0.2, 0.8, 0.9]), useless_ae)
        assert best == 1.0 and scores[1.0] == pytest.approx(1.0)
        assert blend(np.array([1.0]), np.array([0.0]), 0.8).tolist() == [0.8]


@pytest.fixture(scope="module")
def tiny_hybrid():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=400) > 1.0).astype(int)
    names = ["f0", "f1", "f2", "f3"]
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 3},
                        xgb.DMatrix(x, y, feature_names=names), 20)  # fmt: skip
    pre = Preprocessor.fit(x[y == 0])
    net, losses = train_autoencoder(pre.transform(x[y == 0]), epochs=3, log_every=0)
    reference = np.sort(reconstruction_error(net, pre.transform(x[y == 0])))
    return HybridModel(booster, net, pre, reference, 0.8, names), x, losses


class TestHybridModel:
    def test_ae_trains_and_scores_blend(self, tiny_hybrid):
        model, x, losses = tiny_hybrid
        assert losses[-1] < losses[0]
        parts = model.components(x[:5])
        np.testing.assert_allclose(parts["score"], 0.8 * parts["xgb"] + 0.2 * parts["ae"])
        assert np.all((parts["ae"] >= 0) & (parts["ae"] <= 1))

    def test_save_load_round_trip(self, tiny_hybrid, tmp_path):
        model, x, _ = tiny_hybrid
        model.save(tmp_path / "hybrid")
        loaded = HybridModel.load(tmp_path / "hybrid")
        np.testing.assert_allclose(loaded.score(x), model.score(x), atol=1e-6)
        assert loaded.feature_names == model.feature_names and loaded.weight == 0.8


class TestExplainer:
    def test_top_drivers_sum_to_margin_and_rank_by_magnitude(self, tiny_hybrid):
        model, x, _ = tiny_hybrid
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

    def test_empty_input(self, tiny_hybrid):
        assert top_drivers(tiny_hybrid[0].booster, np.empty((0, 4)), ["a"] * 4) == []

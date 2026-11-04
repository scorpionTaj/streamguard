"""Scorer tests (serving/consumer.py, PLAN.md §3, §9) — no Kafka/Redis/MLflow needed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from streamguard.config import load_feature_spec
from streamguard.models.registry import ServingModel, Thresholds
from streamguard.offline.features import build_features, feature_columns, transactions_frame
from streamguard.schemas import Transaction
from streamguard.serving.consumer import fetch_snapshots, risk_band, score_events
from streamguard.streaming.features_job import velocity_columns

SPEC = load_feature_spec()
THRESHOLDS = Thresholds(high=0.8, medium=0.5, review=0.4)


class TestRiskBand:
    @pytest.mark.parametrize(
        ("score", "band"),
        [(0.95, "high"), (0.8, "high"), (0.6, "medium"), (0.45, "review"), (0.1, None)],
    )
    def test_bands(self, score, band):
        assert risk_band(score, THRESHOLDS) == band


class _CaptureBooster:
    """Stands in for xgboost.Booster: records the matrix, scores by row index."""

    def inplace_predict(self, matrix):
        self.matrix = matrix
        return np.linspace(0, 1, len(matrix))


def _txs() -> list[Transaction]:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        Transaction(
            transaction_id=i,
            event_ts=base + timedelta(seconds=60 * i),
            card_key="a" * 16,
            transaction_amt=10.0 * i,
            product_cd="W",
            card4="visa",
            d1=float(i),
            p_emaildomain="gmail.com",
            v={"V187": 0.5},
            is_fraud=0,
        )  # fmt: skip
        for i in range(1, 4)
    ]


class TestScoreEvents:
    def test_matrix_equals_training_features_when_snapshots_match(self):
        txs = _txs()
        frame = transactions_frame(txs, SPEC["v_subset"])
        offline = build_features(frame, SPEC)  # training path: velocity computed offline
        columns = feature_columns(offline)
        snapshots = {t.transaction_id: {n: offline[n][i] for n in velocity_columns(SPEC)}
                     for i, t in enumerate(txs)}  # fmt: skip
        booster = _CaptureBooster()
        model = ServingModel(booster, "1", columns, THRESHOLDS)

        scores = score_events(txs, snapshots, model, SPEC)

        expected = offline.select(columns).to_numpy().astype(np.float32)
        np.testing.assert_array_equal(booster.matrix, expected)
        assert scores.tolist() == [0.0, 0.5, 1.0]

    def test_missing_snapshot_gives_nan_velocity_only(self):
        txs = _txs()[:1]
        columns = feature_columns(build_features(transactions_frame(txs, SPEC["v_subset"]), SPEC))
        booster = _CaptureBooster()
        score_events(txs, {}, ServingModel(booster, "1", columns, THRESHOLDS), SPEC)
        row = dict(zip(columns, booster.matrix[0], strict=True))
        assert all(np.isnan(row[n]) for n in velocity_columns(SPEC))
        assert row["transaction_amt"] == 10.0


class _FakeRedis:
    """HGETALL-only fake; a key appears after `ready_after` polls."""

    def __init__(self, data, ready_after=0):
        self.data, self.ready_after, self.polls = data, ready_after, 0

    def pipeline(self, transaction=False):
        outer = self

        class _Pipe:
            def __enter__(self):
                self.keys = []
                return self

            def __exit__(self, *exc):
                return False

            def hgetall(self, key):
                self.keys.append(key)

            def execute(self):
                outer.polls += 1
                ready = outer.polls > outer.ready_after
                return [outer.data.get(k, {}) if ready else {} for k in self.keys]

        return _Pipe()


class TestFetchSnapshots:
    def test_waits_until_present(self):
        fake = _FakeRedis({"sg:tx:7": {b"txn_count_5m": b"2.0"}}, ready_after=2)
        found = fetch_snapshots(fake, "sg:tx:", [7], ["txn_count_5m"], wait_s=5)
        assert found == {7: {"txn_count_5m": 2.0}} and fake.polls == 3

    def test_gives_up_after_deadline(self):
        fake = _FakeRedis({})
        assert fetch_snapshots(fake, "sg:tx:", [1, 2], ["txn_count_5m"], wait_s=0.05) == {}

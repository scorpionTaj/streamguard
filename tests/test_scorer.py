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
THRESHOLDS = Thresholds(high=0.8, medium=0.5)


class TestRiskBand:
    @pytest.mark.parametrize(
        ("score", "band"),
        [
            (0.95, "high"),
            (0.8, "high"),
            (0.6, "medium"),
            (0.5, "medium"),
            (0.45, None),
            (0.1, None),
        ],
    )
    def test_bands(self, score, band):
        assert risk_band(score, THRESHOLDS) == band


class _CaptureModel:
    """Stands in for XgbModel: records the matrix, scores by row index."""

    def score(self, matrix):
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
        hybrid = _CaptureModel()
        model = ServingModel(hybrid, "1", columns, THRESHOLDS)

        scores, matrix = score_events(txs, snapshots, model, SPEC)

        expected = offline.select(columns).to_numpy()
        np.testing.assert_array_equal(hybrid.matrix, expected)
        np.testing.assert_array_equal(matrix, expected)
        assert scores.tolist() == [0.0, 0.5, 1.0]

    def test_missing_snapshot_gives_nan_velocity_only(self):
        txs = _txs()[:1]
        columns = feature_columns(build_features(transactions_frame(txs, SPEC["v_subset"]), SPEC))
        hybrid = _CaptureModel()
        score_events(txs, {}, ServingModel(hybrid, "1", columns, THRESHOLDS), SPEC)
        row = dict(zip(columns, hybrid.matrix[0], strict=True))
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
        fake = _FakeRedis({"sg:tx:7": {b"txn_count_5m": b"2.0", b"written_at": b"1.5"}},
                          ready_after=2)  # fmt: skip
        found, written = fetch_snapshots(fake, "sg:tx:", [7], ["txn_count_5m"], wait_s=5)
        assert found == {7: {"txn_count_5m": 2.0}} and fake.polls == 3
        assert written == {7: 1.5}

    def test_gives_up_after_deadline(self):
        fake = _FakeRedis({})
        assert fetch_snapshots(fake, "sg:tx:", [1, 2], ["txn_count_5m"], wait_s=0.05) == ({}, {})


class TestHotSwap:
    """Alias watcher + install, without Kafka/Redis (bare Scorer instance)."""

    @staticmethod
    def _serving(champ: str, chall: str | None, features=("a", "b")):
        from streamguard.serving.consumer import Serving

        model = lambda v, f: ServingModel(_CaptureModel(), v, list(f), THRESHOLDS)  # noqa: E731
        return Serving(model(champ, ("a", "b")), model(chall, features) if chall else None)

    def _scorer(self, serving, versions):
        from streamguard.serving.consumer import Scorer, ScorerState

        scorer = Scorer.__new__(Scorer)
        scorer.state, scorer._staged = ScorerState(), None
        scorer.alias_versions = lambda: versions[0]
        scorer.load_serving = lambda: self._serving(*versions[0])
        scorer.install(serving)
        return scorer

    def _watch_once(self, scorer, monkeypatch):
        import threading

        import streamguard.serving.consumer as consumer

        monkeypatch.setattr(consumer, "ALIAS_CHECK_S", 0.01)
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()
        scorer._watch_aliases(stop)

    def test_install_sets_versions_and_canary(self):
        scorer = self._scorer(self._serving("3", "4"), [("3", "4")])
        assert (scorer.state.model_version, scorer.state.challenger_version) == ("3", "4")
        assert scorer.state.canary is not None

    def test_mismatched_challenger_disables_canary(self):
        scorer = self._scorer(self._serving("3", "4", features=("x",)), [("3", "4")])
        assert scorer.challenger is None and scorer.state.canary is None

    def test_watcher_stages_reload_only_when_aliases_move(self, monkeypatch):
        versions = [("3", "4")]
        scorer = self._scorer(self._serving("3", "4"), versions)
        self._watch_once(scorer, monkeypatch)
        assert scorer._staged is None  # unchanged → nothing loaded
        versions[0] = ("4", None)  # `make promote`
        self._watch_once(scorer, monkeypatch)
        assert scorer._staged.versions == ("4", None)
        scorer.install(scorer._staged)
        assert scorer.state.model_version == "4" and scorer.state.canary is None

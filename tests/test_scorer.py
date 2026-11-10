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


class _Msg:
    def __init__(self, partition: int, offset: int, topic: str = "transactions.raw"):
        self._topic, self._partition, self._offset = topic, partition, offset

    def topic(self):
        return self._topic

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset


class _FakeProducer:
    """Records the transactional call sequence; `fail_on` makes one step raise."""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple] = []
        self.fail_on = fail_on

    def _step(self, name: str, *detail) -> None:
        self.calls.append((name, *detail))
        if name == self.fail_on:
            raise RuntimeError(f"{name} failed")

    def begin_transaction(self):
        self._step("begin")

    def produce(self, topic, key, value):
        self._step("produce", topic, key, value)

    def send_offsets_to_transaction(self, offsets, group_metadata):
        self._step("offsets", sorted((tp.partition, tp.offset) for tp in offsets), group_metadata)

    def commit_transaction(self):
        self._step("commit")

    def abort_transaction(self):
        self._step("abort")


class _FakeConsumer:
    def __init__(self):
        self.seeks: list[tuple[int, int]] = []

    def consumer_group_metadata(self):
        return "group-metadata"

    def seek(self, tp):
        self.seeks.append((tp.partition, tp.offset))


class TestExactlyOncePublish:
    """One batch = one transaction: outputs and consumer offsets commit together."""

    MESSAGES = [_Msg(0, 10), _Msg(1, 4), _Msg(0, 11), _Msg(0, 12)]
    OUTPUTS = [("transactions.scored", "k1", "v1"), ("transactions.alerts", "k1", "a1")]

    @staticmethod
    def _scorer(producer):
        from streamguard.serving.consumer import Scorer

        scorer = Scorer.__new__(Scorer)
        scorer.producer, scorer.consumer = producer, _FakeConsumer()
        return scorer

    def test_batch_bounds_per_partition(self):
        from streamguard.serving.consumer import batch_bounds

        first, resume = batch_bounds(self.MESSAGES)
        assert first == {("transactions.raw", 0): 10, ("transactions.raw", 1): 4}
        assert resume == {("transactions.raw", 0): 13, ("transactions.raw", 1): 5}

    def test_outputs_and_offsets_commit_in_one_transaction(self):
        producer = _FakeProducer()
        self._scorer(producer).publish(self.MESSAGES, self.OUTPUTS)
        assert producer.calls == [
            ("begin",),
            ("produce", "transactions.scored", "k1", "v1"),
            ("produce", "transactions.alerts", "k1", "a1"),
            ("offsets", [(0, 13), (1, 5)], "group-metadata"),  # the offset after each last message
            ("commit",),
        ]

    @pytest.mark.parametrize("step", ["produce", "offsets", "commit"])
    def test_failure_aborts_and_propagates(self, step):
        producer = _FakeProducer(fail_on=step)
        with pytest.raises(RuntimeError, match=f"{step} failed"):
            self._scorer(producer).publish(self.MESSAGES, self.OUTPUTS)
        names = [call[0] for call in producer.calls]
        assert names[-1] == "abort" and names.count("commit") == (step == "commit")

    def test_failed_abort_does_not_mask_the_cause(self):
        class _AbortFails(_FakeProducer):
            def abort_transaction(self):
                raise KeyError("fenced")

        with pytest.raises(RuntimeError, match="commit failed"):
            self._scorer(_AbortFails(fail_on="commit")).publish(self.MESSAGES, self.OUTPUTS)

    def test_rewind_returns_to_the_batch_start(self):
        scorer = self._scorer(_FakeProducer())
        scorer.rewind(self.MESSAGES)
        assert sorted(scorer.consumer.seeks) == [(0, 10), (1, 4)]

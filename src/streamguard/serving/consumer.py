"""Scorer consumer loop (PLAN.md §2, §3, §9).

One background thread, consumer group `scorer` on `transactions.raw`:
    poll a batch → for each event:
        1. wait for the event's own velocity snapshot sg:tx:<transaction_id>
           (written by the Spark job) — exact training-time features; on
           timeout the velocity columns are NaN and a miss is counted
        2. static features from the event via the offline twin (same code as training)
        3. champion XGBoost score (batched inplace_predict)
        4. publish ScoredTx → transactions.scored; band-passing → Alert → transactions.alerts
Offsets are stored only after publishing (at-least-once). Prometheus metrics are
module-level and exported by main.py /metrics.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter as TallyCounter
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from prometheus_client import Counter, Histogram

from streamguard.models.registry import ServingModel, Thresholds
from streamguard.offline.features import build_features, transactions_frame
from streamguard.schemas import Alert, RiskBand, ScoredTx, Transaction
from streamguard.streaming.features_job import velocity_columns

log = logging.getLogger("streamguard.scorer")

SCORED = Counter("streamguard_scored_total", "Transactions scored")
ALERTS = Counter("streamguard_alerts_total", "Alerts published", ["band"])
FEATURE_MISSES = Counter(
    "streamguard_feature_misses_total", "Events scored without a velocity snapshot"
)
FEATURE_WAIT = Histogram("streamguard_feature_wait_seconds", "Wait for Spark snapshots, per batch",
                         buckets=(0.005, 0.05, 0.25, 0.5, 1, 2, 3, 5, 10))  # fmt: skip
INFERENCE = Histogram("streamguard_inference_seconds", "Feature build + model, per batch",
                      buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1))  # fmt: skip
E2E = Histogram("streamguard_e2e_latency_seconds", "Producer send → scored/alert published",
                buckets=(0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 20))  # fmt: skip


def risk_band(score: float, thresholds: Thresholds) -> RiskBand | None:
    """Highest band whose floor the score reaches; None = no alert (PLAN.md §9)."""
    if score >= thresholds.high:
        return "high"
    if score >= thresholds.medium:
        return "medium"
    if score >= thresholds.review:
        return "review"
    return None


@dataclass
class ScorerState:
    """What the API serves: recent alerts + counters (shared with the consumer thread)."""

    alerts: deque = field(default_factory=lambda: deque(maxlen=2000))
    scored: int = 0
    by_band: TallyCounter = field(default_factory=TallyCounter)
    feature_misses: int = 0
    model_version: str = ""
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def fetch_snapshots(redis_client, prefix: str, ids: list[int], names: list[str],
                    wait_s: float) -> dict[int, dict[str, float]]:  # fmt: skip
    """Poll Redis until every event's snapshot exists or `wait_s` passes."""
    found: dict[int, dict[str, float]] = {}
    pending, deadline = list(dict.fromkeys(ids)), time.monotonic() + wait_s
    while pending:
        with redis_client.pipeline(transaction=False) as pipe:
            for tx_id in pending:
                pipe.hgetall(f"{prefix}{tx_id}")
            results = pipe.execute()
        for tx_id, raw in zip(pending, results, strict=True):
            if raw:
                found[tx_id] = {n: float(raw[n.encode()]) for n in names}
        pending = [i for i in pending if i not in found]
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    return found


def score_events(txs: list[Transaction], snapshots: dict[int, dict[str, float]],
                 model: ServingModel, spec: dict) -> np.ndarray:  # fmt: skip
    """Training-identical feature vectors (offline twin + Redis velocity) → champion scores."""
    names = velocity_columns(spec)
    velocity = {n: np.array([snapshots.get(t.transaction_id, {}).get(n, np.nan) for t in txs])
                for n in names}  # fmt: skip
    frame = transactions_frame(txs, spec.get("v_subset") or [])
    matrix = build_features(frame, spec, velocity).select(model.feature_names).to_numpy()
    return np.asarray(model.booster.inplace_predict(matrix.astype(np.float32)), dtype=float)


class Scorer:
    def __init__(self, model: ServingModel, spec: dict, pipeline: dict, state: ScorerState,
                 bootstrap: str, redis_url: str, wait_s: float = 10.0) -> None:  # fmt: skip
        import redis
        from confluent_kafka import Consumer, Producer

        self.model, self.spec, self.pipeline, self.state = model, spec, pipeline, state
        self.wait_s = wait_s
        self.velocity_names = velocity_columns(spec)
        self.redis = redis.Redis.from_url(redis_url)
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap,
            "group.id": "scorer",
            "auto.offset.reset": "earliest",
            "enable.auto.offset.store": False,
        })  # fmt: skip
        self.producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 5})
        self.consumer.subscribe([pipeline["topics"]["raw"]])
        state.model_version = model.version

    def run(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                messages = [m for m in self.consumer.consume(500, timeout=0.5) if not m.error()]
                if messages:
                    self.process(messages)
                self.producer.poll(0)
        except Exception as exc:  # surfaced on /health instead of a silently dead thread
            log.exception("scorer loop crashed")
            self.state.error = repr(exc)
            raise
        finally:
            self.producer.flush(10)
            self.consumer.close()

    def process(self, messages: list) -> None:
        txs = [Transaction.model_validate_json(m.value()) for m in messages]
        started = time.monotonic()
        snapshots = fetch_snapshots(self.redis, self.pipeline["redis"]["tx_key_prefix"],
                                    [t.transaction_id for t in txs], self.velocity_names,
                                    self.wait_s)  # fmt: skip
        FEATURE_WAIT.observe(time.monotonic() - started)
        with INFERENCE.time():
            scores = score_events(txs, snapshots, self.model, self.spec)

        topics, version = self.pipeline["topics"], self.model.version
        thresholds = self.model.thresholds
        floors = {"high": thresholds.high, "medium": thresholds.medium, "review": thresholds.review}
        new_alerts, misses = [], 0
        for msg, tx, score in zip(messages, txs, scores, strict=True):
            scored = ScoredTx(**tx.model_dump(), score=float(score), model_version=version,
                              scored_at=datetime.now(UTC))  # fmt: skip
            self.producer.produce(topics["scored"], key=tx.card_key, value=scored.model_dump_json())
            snapshot = snapshots.get(tx.transaction_id)
            misses += snapshot is None
            if band := risk_band(scored.score, thresholds):
                alert = Alert(**scored.model_dump(), risk_band=band, threshold=floors[band],
                              velocity_snapshot=snapshot or {})  # fmt: skip
                self.producer.produce(topics["alerts"], key=tx.card_key,
                                      value=alert.model_dump_json())  # fmt: skip
                new_alerts.append(alert)
                ALERTS.labels(band).inc()
            self.consumer.store_offsets(msg)
            E2E.observe(max(0.0, time.time() - msg.timestamp()[1] / 1000.0))
        self.producer.poll(0)

        SCORED.inc(len(txs))
        FEATURE_MISSES.inc(misses)
        with self.state.lock:
            self.state.scored += len(txs)
            self.state.feature_misses += misses
            self.state.alerts.extend(new_alerts)
            self.state.by_band.update(a.risk_band for a in new_alerts)

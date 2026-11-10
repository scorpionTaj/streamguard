"""Scorer consumer loop (PLAN.md §2, §3, §9).

One background thread, consumer group `scorer` on `transactions.raw`:
    poll a batch → for each event:
        1. wait for the event's own velocity snapshot sg:tx:<transaction_id>
           (written by the Spark job) — exact training-time features; on
           timeout the velocity columns are NaN and a miss is counted
        2. static features from the event via the offline twin (same code as training)
        3. champion score (XGBoost, batched)
        4. publish ScoredTx → transactions.scored; band-passing → Alert → transactions.alerts
           (with a challenger alias: score twice, record the canary delta — PLAN.md §10)
        5. SHAP top-5 for the batch's alerts on a background worker (never blocks publish);
           drivers land on the API's alert records (GET /alerts/{transaction_id})
Each batch is one Kafka transaction: its scored messages, its alerts and the consumer
offsets commit together or not at all (exactly-once on the output topics for
read-committed readers, the client default). A scorer killed mid-batch leaves an open
transaction that the next start aborts, then resumes from the last committed batch: no
event is lost and none is published twice. Prometheus metrics are module-level and
exported by main.py /metrics; a side thread recomputes feature PSI (eval/drift.py) and
the loop refreshes consumer lag every few seconds.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter as TallyCounter
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from streamguard.eval.drift import DriftMonitor
from streamguard.models.explainer import top_drivers
from streamguard.models.registry import ServingModel, Thresholds
from streamguard.offline.features import build_features, transactions_frame
from streamguard.schemas import Alert, RiskBand, ScoredTx, Transaction
from streamguard.serving.canary import CanaryWindow
from streamguard.streaming.features_job import velocity_columns

log = logging.getLogger("streamguard.scorer")

CONSUME_MAX = 500  # small polls keep `consumed` near arrival: a huge cap swallows the
#                    stream while a batch waits for Spark snapshots — the latency tail.
#                    500 still bounds that at ~200 tx/s (≈100 arrive per 0.5 s poll)
#                    and lets the scorer drain faster than the 1,000 tx/s load rate
SCORED = Counter("streamguard_scored_total", "Transactions scored")
ALERTS = Counter("streamguard_alerts_total", "Alerts published", ["band"])
FEATURE_MISSES = Counter(
    "streamguard_feature_misses_total", "Events scored without a velocity snapshot"
)
FEATURE_WAIT = Histogram("streamguard_feature_wait_seconds", "Wait for Spark snapshots, per batch",
                         buckets=(0.005, 0.05, 0.25, 0.5, 1, 2, 3, 5, 10))  # fmt: skip
INFERENCE = Histogram("streamguard_inference_seconds", "Feature build + model, per batch",
                      buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1))  # fmt: skip
EXPLAIN = Histogram("streamguard_explain_seconds", "Async SHAP per alert batch",
                    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1))  # fmt: skip
E2E = Histogram("streamguard_e2e_latency_seconds", "Producer send → scored/alert published",
                buckets=(0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 20))  # fmt: skip
CONSUMER_LAG = Gauge(
    "streamguard_consumer_lag_messages", "transactions.raw messages not yet scored"
)

TRANSACTIONAL_ID = "streamguard-scorer"  # one scorer; a restart fences its predecessor
LAG_EVERY_S = 5.0
DRIFT_EVERY_S = 30.0
ALIAS_CHECK_S = 30.0
MODEL_SWAPS = Counter("streamguard_model_swaps_total", "Live champion/challenger reloads")


def risk_band(score: float, thresholds: Thresholds) -> RiskBand | None:
    """Highest band whose floor the score reaches; None = no alert (PLAN.md §9)."""
    if score >= thresholds.high:
        return "high"
    if score >= thresholds.medium:
        return "medium"
    return None


@dataclass
class ScorerState:
    """What the API serves: recent alerts + counters (shared with the consumer thread)."""

    alerts: deque = field(default_factory=lambda: deque(maxlen=2000))
    scored: int = 0
    by_band: TallyCounter = field(default_factory=TallyCounter)
    feature_misses: int = 0
    model_version: str = ""
    challenger_version: str | None = None
    error: str | None = None
    canary: CanaryWindow | None = None
    drift: DriftMonitor | None = None
    # per-event latency spans for `make bench`: (tx_id, created, consumed, snapshot_written,
    # features_ready, scored, published) — epoch seconds
    spans: deque = field(default_factory=lambda: deque(maxlen=100_000))
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class Serving:
    """Everything that changes when registry aliases move (hot-swapped as one unit)."""

    champion: ServingModel
    challenger: ServingModel | None = None
    drift: DriftMonitor | None = None

    @property
    def versions(self) -> tuple[str, str | None]:
        return self.champion.version, self.challenger.version if self.challenger else None


def batch_bounds(messages: list) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], int]]:
    """Per (topic, partition): the batch's first offset, and the offset after its last.

    The second is what the group resumes from once the batch is published; the first is
    where to rewind when it has to be polled again.
    """
    first: dict[tuple[str, int], int] = {}
    resume: dict[tuple[str, int], int] = {}
    for m in messages:
        key = (m.topic(), m.partition())
        first[key] = min(first.get(key, m.offset()), m.offset())
        resume[key] = max(resume.get(key, 0), m.offset() + 1)
    return first, resume


def fetch_snapshots(redis_client, prefix: str, ids: list[int], names: list[str],
                    wait_s: float) -> tuple[dict[int, dict[str, float]], dict[int, float]]:  # fmt: skip
    """Poll Redis until every event's snapshot exists or `wait_s` passes.

    Returns (velocity features per tx, Spark write time per tx — for latency spans).
    """
    found: dict[int, dict[str, float]] = {}
    written: dict[int, float] = {}
    pending, deadline = list(dict.fromkeys(ids)), time.monotonic() + wait_s
    while pending:
        with redis_client.pipeline(transaction=False) as pipe:
            for tx_id in pending:
                pipe.hgetall(f"{prefix}{tx_id}")
            results = pipe.execute()
        for tx_id, raw in zip(pending, results, strict=True):
            if raw:
                found[tx_id] = {n: float(raw[n.encode()]) for n in names}
                written[tx_id] = float(raw.get(b"written_at", "nan"))
        pending = [i for i in pending if i not in found]
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    return found, written


def score_events(txs: list[Transaction], snapshots: dict[int, dict[str, float]],
                 model: ServingModel, spec: dict) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Training-identical feature vectors (offline twin + Redis velocity) → (scores, matrix)."""
    names = velocity_columns(spec)
    velocity = {n: np.array([snapshots.get(t.transaction_id, {}).get(n, np.nan) for t in txs])
                for n in names}  # fmt: skip
    frame = transactions_frame(txs, spec.get("v_subset") or [])
    matrix = build_features(frame, spec, velocity).select(model.feature_names).to_numpy()
    return np.asarray(model.model.score(matrix), dtype=float), matrix


class Scorer:
    def __init__(self, serving: Serving, spec: dict, pipeline: dict, state: ScorerState,
                 bootstrap: str, redis_url: str, wait_s: float = 10.0,
                 load_serving: Callable[[], Serving] | None = None,
                 alias_versions: Callable[[], tuple[str | None, str | None]] | None = None,
                 ) -> None:  # fmt: skip
        import redis
        from confluent_kafka import Consumer, Producer

        self.spec, self.pipeline, self.state = spec, pipeline, state
        self.load_serving, self.alias_versions = load_serving, alias_versions
        self._staged: Serving | None = None  # loaded by the watcher, installed between batches
        self.wait_s = wait_s
        self.velocity_names = velocity_columns(spec)
        self.redis = redis.Redis.from_url(redis_url)
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap,
            "group.id": "scorer",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,  # offsets move only inside the publish transaction
        })  # fmt: skip
        self.producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 5,
                                  "transactional.id": TRANSACTIONAL_ID})  # fmt: skip
        self.producer.init_transactions(30)  # also aborts a killed predecessor's open batch
        self.explainer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shap")
        self.consumer.subscribe([pipeline["topics"]["raw"]])
        self._next_lag = 0.0
        self.install(serving)

    def install(self, serving: Serving) -> None:
        """Make `serving` live (consumer thread, between batches); resets the canary window."""
        challenger = serving.challenger
        if challenger is not None and challenger.feature_names != serving.champion.feature_names:
            log.warning("challenger v%s has a different feature set — canary disabled",
                        challenger.version)  # fmt: skip
            challenger = None
        self.serving = Serving(serving.champion, challenger, serving.drift)
        self.model, self.challenger = serving.champion, challenger
        with self.state.lock:
            self.state.model_version = serving.champion.version
            self.state.challenger_version = challenger.version if challenger else None
            self.state.canary = CanaryWindow() if challenger else None
            self.state.drift = serving.drift
        log.info("serving champion v%s%s", serving.champion.version,
                 f", canary vs challenger v{challenger.version}" if challenger else "")  # fmt: skip

    def _watch_aliases(self, stop: threading.Event) -> None:
        """Poll the registry; when an alias moves, load the new models off the hot path."""
        while not stop.wait(ALIAS_CHECK_S):
            try:
                wanted = self.alias_versions()
                if wanted == self.serving.versions or wanted[0] is None:
                    continue
                log.info("aliases moved %s → %s, loading", self.serving.versions, wanted)
                self._staged = self.load_serving()
            except Exception:  # registry hiccup: keep serving what we have
                log.exception("alias check / model reload failed")

    def run(self, stop: threading.Event) -> None:
        import redis.exceptions

        threading.Thread(target=self._drift_loop, args=(stop,), name="drift", daemon=True).start()
        if self.load_serving is not None and self.alias_versions is not None:
            threading.Thread(target=self._watch_aliases, args=(stop,), name="aliases",
                             daemon=True).start()  # fmt: skip
        redis_fails = 0
        try:
            while not stop.is_set():
                if (staged := self._staged) is not None:
                    self._staged = None
                    self.install(staged)
                    MODEL_SWAPS.inc()
                messages = [m for m in self.consumer.consume(CONSUME_MAX, timeout=0.5)
                            if not m.error()]  # fmt: skip
                if messages:
                    try:
                        self.process(messages)
                    except redis.exceptions.ConnectionError:
                        # a Redis restart must not kill the loop: the batch was neither
                        # published nor committed (snapshots are fetched before any
                        # publish), so back off, rewind and poll it again
                        redis_fails += 1
                        delay = min(2**redis_fails, 30)
                        log.warning("redis unavailable (fail %d) — retrying batch in %d s",
                                    redis_fails, delay)  # fmt: skip
                        time.sleep(delay)
                        self.rewind(messages)
                        continue
                    redis_fails = 0
                    self.state.error = None
                self.producer.poll(0)
                if time.monotonic() >= self._next_lag:
                    self._update_lag()
        except Exception as exc:  # surfaced on /health instead of a silently dead thread
            log.exception("scorer loop crashed")
            self.state.error = repr(exc)
            raise
        finally:
            self.explainer.shutdown(wait=False, cancel_futures=True)
            self.producer.flush(10)
            self.consumer.close()

    def rewind(self, messages: list) -> None:
        """Put the consumer back at the start of a batch that was not published."""
        from confluent_kafka import TopicPartition

        first, _ = batch_bounds(messages)
        for (topic, partition), offset in first.items():
            self.consumer.seek(TopicPartition(topic, partition, offset))

    def publish(self, messages: list, outputs: list[tuple[str, str, str]]) -> None:
        """Write `outputs` (topic, key, value) and move the group past `messages`, atomically.

        If anything fails the transaction is aborted: nothing becomes visible, the offsets
        stay where they were, and the error propagates.
        """
        from confluent_kafka import TopicPartition

        _, resume = batch_bounds(messages)
        self.producer.begin_transaction()
        try:
            for topic, key, value in outputs:
                self.producer.produce(topic, key=key, value=value)
            self.producer.send_offsets_to_transaction(
                [TopicPartition(t, p, offset) for (t, p), offset in resume.items()],
                self.consumer.consumer_group_metadata(),
            )
            self.producer.commit_transaction()
        except Exception:
            try:
                self.producer.abort_transaction()
            except Exception:  # e.g. fenced by a newer scorer: the original error matters
                log.exception("could not abort the publish transaction")
            raise

    def _update_lag(self) -> None:
        """Messages on transactions.raw past the scorer's position (consumer thread only)."""
        self._next_lag = time.monotonic() + LAG_EVERY_S
        try:
            assigned = self.consumer.assignment()
            if not assigned:
                return
            # position() is invalid (< 0) until a partition yields its first message after a
            # restart — an idle, caught-up scorer would read as "whole topic behind". Fall
            # back to the committed offset, and skip partitions with neither.
            committed = {(tp.topic, tp.partition): tp.offset
                         for tp in self.consumer.committed(assigned, timeout=1.0)}  # fmt: skip
            lag = 0
            for tp in self.consumer.position(assigned):
                offset = (
                    tp.offset if tp.offset >= 0 else committed.get((tp.topic, tp.partition), -1)
                )
                if offset < 0:
                    continue
                _, high = self.consumer.get_watermark_offsets(tp, timeout=1.0, cached=False)
                lag += max(0, high - offset)
            CONSUMER_LAG.set(lag)
        except Exception:  # monitoring must never take the scorer down
            log.debug("lag update failed", exc_info=True)

    def _drift_loop(self, stop: threading.Event) -> None:
        while not stop.wait(DRIFT_EVERY_S):
            try:
                if (drift := self.state.drift) is not None:
                    drift.compute()
            except Exception:
                log.exception("drift computation failed")

    def process(self, messages: list) -> None:
        consumed_at = time.time()
        txs = [Transaction.model_validate_json(m.value()) for m in messages]
        started = time.monotonic()
        snapshots, written_at = fetch_snapshots(
            self.redis, self.pipeline["redis"]["tx_key_prefix"],
            [t.transaction_id for t in txs], self.velocity_names, self.wait_s,
        )  # fmt: skip
        FEATURE_WAIT.observe(time.monotonic() - started)
        features_at = time.time()
        with INFERENCE.time():
            scores, matrix = score_events(txs, snapshots, self.model, self.spec)
        scored_at = time.time()
        if self.challenger is not None:  # canary: score twice, champion alone drives alerts
            challenger_scores = np.asarray(self.challenger.model.score(matrix), dtype=float)
            self.state.canary.record(
                scores, challenger_scores,
                [risk_band(s, self.model.thresholds) is not None for s in scores],
                [risk_band(s, self.challenger.thresholds) is not None for s in challenger_scores],
            )  # fmt: skip
        if self.state.drift is not None:
            self.state.drift.add(matrix)

        topics, version = self.pipeline["topics"], self.model.version
        thresholds = self.model.thresholds
        floors = {"high": thresholds.high, "medium": thresholds.medium}
        outputs: list[tuple[str, str, str]] = []
        new_alerts, alert_rows, misses = [], [], 0
        for row, (tx, score) in enumerate(zip(txs, scores, strict=True)):
            scored = ScoredTx(**tx.model_dump(), score=float(score), model_version=version,
                              scored_at=datetime.now(UTC))  # fmt: skip
            outputs.append((topics["scored"], tx.card_key, scored.model_dump_json()))
            snapshot = snapshots.get(tx.transaction_id)
            misses += snapshot is None
            if band := risk_band(scored.score, thresholds):
                alert = Alert(**scored.model_dump(), risk_band=band, threshold=floors[band],
                              velocity_snapshot=snapshot or {})  # fmt: skip
                outputs.append((topics["alerts"], tx.card_key, alert.model_dump_json()))
                new_alerts.append(alert)
                alert_rows.append(row)
        self.publish(messages, outputs)
        published_at = time.time()  # the commit: when readers can first see the batch

        spans = []
        for msg, tx in zip(messages, txs, strict=True):
            created_at = msg.timestamp()[1] / 1000.0
            E2E.observe(max(0.0, published_at - created_at))
            spans.append((tx.transaction_id, created_at, consumed_at,
                          written_at.get(tx.transaction_id, float("nan")),
                          features_at, scored_at, published_at))  # fmt: skip
        for alert in new_alerts:
            ALERTS.labels(alert.risk_band).inc()

        SCORED.inc(len(txs))
        FEATURE_MISSES.inc(misses)
        with self.state.lock:
            self.state.scored += len(txs)
            self.state.feature_misses += misses
            self.state.alerts.extend(new_alerts)
            self.state.by_band.update(a.risk_band for a in new_alerts)
            self.state.spans.extend(spans)
        if new_alerts:  # SHAP off the publish path (PLAN.md §3)
            # pass the model that scored them: a hot swap may land before the job runs
            self.explainer.submit(self.explain, self.model, new_alerts, matrix[alert_rows])

    def explain(self, model: ServingModel, alerts: list[Alert], rows: np.ndarray) -> None:
        try:
            with EXPLAIN.time():
                drivers = top_drivers(model.model.booster, rows, model.feature_names)
        except Exception:
            log.exception("SHAP explanation failed")
            return
        with self.state.lock:
            for alert, top in zip(alerts, drivers, strict=True):
                alert.top_drivers = top

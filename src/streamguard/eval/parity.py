"""Online/offline feature parity (`make parity`, PLAN.md §8).

Ground truth is the stream itself: read every event on `transactions.raw`,
recompute velocity offline (same contract payloads, replay order), and compare
with the Spark job's Redis hash for up to 1k sampled cards. Each hash is the
snapshot after the card's latest event, so the offline value for that same
transaction_id must match per feature within 1e-6. Waits for Spark to catch up.

Exit code 1 on any mismatch or if fewer than `--min-cards` could be compared —
CI fails when the streaming and training feature definitions diverge.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

import numpy as np
import polars as pl

from streamguard.config import (
    load_feature_spec,
    load_pipeline_config,
    redis_card_key,
    velocity_window_seconds,
)
from streamguard.offline.features import transactions_frame
from streamguard.schemas import Transaction
from streamguard.streaming.features_job import velocity_columns
from streamguard.velocity import velocity_features

TOLERANCE = 1e-6


def read_topic(bootstrap: str, topic: str, idle_timeout: float = 10.0) -> list[Transaction]:
    """Every event currently on `topic`, from the earliest offset (throwaway consumer group)."""
    from confluent_kafka import Consumer, KafkaError, TopicPartition

    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": f"sg-parity-{uuid.uuid4().hex[:8]}",
        "enable.auto.commit": False,
        "enable.partition.eof": True,
    })  # fmt: skip
    partitions = consumer.list_topics(topic, timeout=10).topics[topic].partitions
    consumer.assign([TopicPartition(topic, p, 0) for p in partitions])  # offset 0 = earliest
    events, at_eof, last_msg = [], set(), time.monotonic()
    while len(at_eof) < len(partitions) and time.monotonic() - last_msg < idle_timeout:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        last_msg = time.monotonic()
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                at_eof.add(msg.partition())
                continue
            raise RuntimeError(msg.error())
        events.append(Transaction.model_validate_json(msg.value()))
    consumer.close()
    return events


def offline_latest(events: list[Transaction], spec: dict) -> pl.DataFrame:
    """Offline velocity for each card's latest event, in the stream's replay order."""
    frame = transactions_frame(events, []).sort("event_ts", "transaction_id")
    if frame["transaction_id"].is_duplicated().any():
        raise SystemExit(
            "duplicate transaction_ids on the topic — run `make reset` and replay once"
        )
    cfg = spec["velocity"]
    feats = velocity_features(
        frame["card_key"].to_numpy(),
        frame["event_ts_s"].to_numpy(),
        frame["transaction_amt"].to_numpy(),
        velocity_window_seconds(spec),
        cfg["metrics"],
        float(cfg.get("sentinel_no_history", -1.0)),
    )
    return (
        frame.select("card_key", "transaction_id")
        .with_columns(**feats)
        .unique("card_key", keep="last", maintain_order=True)
    )


def compare(
    expected: pl.DataFrame, redis_client, prefix: str, names: list[str], wait_s: float
) -> tuple[int, float, list[str]]:
    """(cards compared, max |Δ|, problems). Polls until Redis holds each card's latest tx."""
    pending = {row["card_key"]: row for row in expected.iter_rows(named=True)}
    max_delta, problems, deadline = 0.0, [], time.monotonic() + wait_s
    compared = 0
    while pending and time.monotonic() < deadline:
        for card, row in list(pending.items()):
            online = redis_client.hgetall(redis_card_key(prefix, card))
            if not online or int(online[b"transaction_id"]) != row["transaction_id"]:
                continue  # Spark hasn't written this card's latest event yet
            del pending[card]
            compared += 1
            for name in names:
                delta = abs(float(online[name.encode()]) - row[name])
                max_delta = max(max_delta, delta)
                if delta >= TOLERANCE:
                    problems.append(f"{card} tx={row['transaction_id']} {name}: "
                                    f"online={online[name.encode()].decode()} offline={row[name]}")  # fmt: skip
        if pending:
            time.sleep(1.0)
    if pending:
        problems.append(f"{len(pending)} cards never reached their latest event in Redis "
                        f"within {wait_s:.0f}s (is `make features` running?)")  # fmt: skip
    return compared, max_delta, problems


def main() -> None:
    import redis
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Online (Redis) vs offline feature parity")
    parser.add_argument("--sample", type=int, default=1000, help="cards to compare")
    parser.add_argument("--min-cards", type=int, default=1000)
    parser.add_argument("--wait", type=float, default=120.0, help="seconds to wait for Spark")
    args = parser.parse_args()

    spec, pipeline = load_feature_spec(), load_pipeline_config()
    events = read_topic(os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"), pipeline["topics"]["raw"])
    expected = offline_latest(events, spec)
    if expected.height > args.sample:
        rng = np.random.default_rng(42)
        expected = expected[np.sort(rng.choice(expected.height, args.sample, replace=False))]
    print(f"{len(events):,} events on topic → comparing {expected.height:,} cards")

    client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    names = velocity_columns(spec)
    compared, max_delta, problems = compare(
        expected, client, pipeline["redis"]["key_prefix"], names, args.wait
    )
    print(f"compared {compared:,} cards × {len(names)} features: max |Δ| = {max_delta:.3g}")
    for line in problems[:20]:
        print("  ✗", line)
    if problems or compared < args.min_cards:
        if compared < args.min_cards:
            print(f"  ✗ only {compared} cards compared (< {args.min_cards}); replay more events")
        sys.exit(1)
    print(f"parity OK (max |Δ| < {TOLERANCE:g})")


if __name__ == "__main__":
    main()

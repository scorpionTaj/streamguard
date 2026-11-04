"""Measure stream completeness & duplicates for a replay window — resilience drills.

Two modes:
    --mark out.json      record the current per-partition high watermarks of the raw topic
    --measure out.json   consume raw from those offsets → the window's event ids; scan
                         transactions.scored and transactions.alerts (from the beginning)
                         for those ids and report completeness / duplicates.

Run `--mark` BEFORE the drill's replay, `--measure` after the scorer has caught up
(check consumer lag first). Writes the report next to the mark file.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from streamguard.config import load_pipeline_config

RAW, SCORED, ALERTS = ("transactions.raw", "transactions.scored", "transactions.alerts")


def consumer():
    from confluent_kafka import Consumer

    c = Consumer({"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
                  "group.id": "sg-resilience-probe",
                  "enable.auto.commit": False,
                  "enable.partition.eof": True,
                  "auto.offset.reset": "earliest"})  # fmt: skip
    return c


def watermarks(c, topic: str) -> dict[int, int]:
    from confluent_kafka import TopicPartition

    meta = c.list_topics(topic, timeout=10).topics[topic]
    return {p: c.get_watermark_offsets(TopicPartition(topic, p), timeout=10)[1]
            for p in meta.partitions}  # fmt: skip


def read_raw_window(c, start: dict[int, int], end: dict[int, int]) -> set[int]:
    """Event ids on the raw topic in [start, end) per partition."""
    from confluent_kafka import KafkaError, TopicPartition

    assignment = [TopicPartition(RAW, p, start.get(p, 0)) for p in end]
    c.assign(assignment)
    ids: set[int] = set()
    eofs: set[int] = set()
    while len(eofs) < len(end):
        msg = c.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                eofs.add(msg.partition())
            continue
        if msg.offset() >= start.get(msg.partition(), 0):
            ids.add(int(json.loads(msg.value())["transaction_id"]))
    c.unassign()
    return ids


def scan_topic(c, topic: str, ids: set[int], since: str = "") -> tuple[Counter, int]:
    """Per-id message counts on `topic` restricted to the window ids; total messages scanned.

    `since` (ISO timestamp prefix) restricts to messages produced after that moment —
    needed because `make replay` re-streams the same rows, so ids alone are ambiguous.
    """
    from confluent_kafka import KafkaError, TopicPartition

    end = watermarks(c, topic)
    c.assign([TopicPartition(topic, p, 0) for p in end])
    counts: Counter = Counter()
    total = 0
    eofs: set[int] = set()
    while len(eofs) < len(end):
        msg = c.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                eofs.add(msg.partition())
            continue
        total += 1
        value = json.loads(msg.value())
        if since and str(value.get("scored_at", "")) < since:
            continue
        tx_id = value.get("transaction_id")
        if tx_id in ids:
            counts[tx_id] += 1
    c.unassign()
    return counts, total


def mark(path: str, *_: object) -> None:
    c = consumer()
    try:
        marks = watermarks(c, load_pipeline_config()["topics"]["raw"])
    finally:
        c.close()
    with open(path, "w") as f:
        json.dump({"topic": RAW, "offsets": marks}, f)
    print(f"marked raw offsets {marks} → {path}")


def measure(path: str, since: str = "") -> None:
    with open(path) as f:
        marks = {int(k): v for k, v in json.load(f)["offsets"].items()}
    c = consumer()
    try:
        end = watermarks(c, RAW)
        ids = read_raw_window(c, marks, end)
        scored, scored_total = scan_topic(c, SCORED, ids, since)
        alerts, alerts_total = scan_topic(c, ALERTS, ids, since)
    finally:
        c.close()

    distinct = len(scored)
    scored_window_total = sum(scored.values())
    alert_window_total = sum(alerts.values())
    report = {
        "window_events": len(ids),
        "scored_messages": scored_window_total,
        "scored_distinct": distinct,
        "scored_duplicates": scored_window_total - distinct,
        "missing_scores": len(ids) - distinct,
        "alert_messages": alert_window_total,
        "alert_distinct": len(alerts),
        "alert_duplicates": alert_window_total - len(alerts),
        "scored_topic_total_msgs": scored_total,
        "alerts_topic_total_msgs": alerts_total,
    }
    print(f"window events        : {report['window_events']:,}")
    print(
        f"scored (distinct)    : {distinct:,} of {scored_window_total:,} in-window messages "
        f"→ {report['scored_duplicates']} duplicates, {report['missing_scores']} missing"
    )
    print(
        f"alerts (distinct tx) : {len(alerts):,} of {alert_window_total:,} in-window messages "
        f"→ {report['alert_duplicates']} duplicate alerts"
    )
    out = path.replace(".json", "-report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"→ {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resilience drill measurement")
    parser.add_argument("mode", choices=["mark", "measure"])
    parser.add_argument("--state", default="artifacts/resilience.json")
    parser.add_argument(
        "--since", default="", help="ISO time: only scored/alerts messages at/after it count"
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
    (mark if args.mode == "mark" else measure)(args.state, args.since)


if __name__ == "__main__":
    main()

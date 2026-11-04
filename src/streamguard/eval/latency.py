"""End-to-end latency benchmark — `make bench` (PLAN.md §3, §13).

Replays the next N dataset rows (continuing wherever `transactions.raw` ends, so
Spark's velocity state stays a clean prefix replay) at the configured pace
(≤ 200 tx/s), waits until the scorer has published all of them, then reads the
scorer's per-event spans (GET /latency) and reports p50/p95/p99 per hop:

    broker → Spark → Redis   snapshot_written − created     (Spark micro-batch + write)
    broker → scorer          consumed − created
    snapshot wait            features_ready − consumed      (scorer waits for Spark)
    inference                scored − features_ready        (feature build + model, batched)
    publish                  published − scored
    end-to-end               published − created            ← KPI: p95 < 5 s

`created` is the Kafka CreateTime set by the producer. Spark runs in a container,
so its timestamp relies on host/VM clock sync (OrbStack/Docker keep ms-level sync).
Writes artifacts/bench.json and prints a markdown table for the README.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime

import numpy as np

from streamguard.config import REPO_ROOT, load_pipeline_config

KPI_P95_S = 5.0
HOPS = {
    "broker → Spark → Redis": ("created", "snapshot_written"),
    "broker → scorer": ("created", "consumed"),
    "snapshot wait": ("consumed", "features_ready"),
    "inference": ("features_ready", "scored"),
    "publish": ("scored", "published"),
    "end-to-end": ("created", "published"),
}


def topic_size(bootstrap: str, topic: str) -> int:
    """Messages currently on `topic` (sum of high watermarks; 0 if it doesn't exist)."""
    from confluent_kafka import Consumer, TopicPartition

    consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": "sg-bench-probe"})
    try:
        meta = consumer.list_topics(topic, timeout=10).topics.get(topic)
        if meta is None or meta.error is not None:
            return 0
        return sum(consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=10)[1]
                   for p in meta.partitions)  # fmt: skip
    finally:
        consumer.close()


def hop_stats(spans: list[list[float]], fields: list[str]) -> dict[str, dict[str, float]]:
    data = np.array(spans, dtype=float)
    col = {name: data[:, i] for i, name in enumerate(fields)}
    stats = {}
    for hop, (start, end) in HOPS.items():
        values = col[end] - col[start]
        values = values[np.isfinite(values)]
        stats[hop] = {
            "n": int(values.size),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "mean": float(values.mean()),
        }
    return stats


def markdown(stats: dict, meta: dict) -> str:
    lines = [
        f"Bench: {meta['events']:,} events @ {meta['throughput_tx_s']:.0f} tx/s "
        f"(champion v{meta['model_version']}, {meta['timestamp']})",
        "",
        "| hop | p50 (s) | p95 (s) | p99 (s) |",
        "|---|---|---|---|",
    ]
    for hop, s in stats.items():
        bold = "**" if hop == "end-to-end" else ""
        lines.append(f"| {bold}{hop}{bold} | {bold}{s['p50']:.2f}{bold} | "
                     f"{bold}{s['p95']:.2f}{bold} | {bold}{s['p99']:.2f}{bold} |")  # fmt: skip
    verdict = "PASS" if stats["end-to-end"]["p95"] < KPI_P95_S else "FAIL"
    lines += ["", f"KPI p95 end-to-end < {KPI_P95_S:.0f} s: **{verdict}**"]
    return "\n".join(lines)


def main() -> None:
    import httpx
    from dotenv import load_dotenv

    from streamguard.producer.replay import load_replay_frame, run

    load_dotenv()
    parser = argparse.ArgumentParser(description="End-to-end latency benchmark")
    parser.add_argument("--events", type=int, default=12_000, help="≈ 60 s at 200 tx/s")
    parser.add_argument("--api", default=os.getenv("SCORER_API", "http://localhost:8000"))
    parser.add_argument("--timeout", type=float, default=120.0, help="wait for scoring (s)")
    args = parser.parse_args()

    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    health = httpx.get(f"{args.api}/health", timeout=5).json()
    if health.get("status") != "ok":
        raise SystemExit(f"scorer not healthy: {health}")
    offset = topic_size(bootstrap, load_pipeline_config()["topics"]["raw"])
    ids = set(load_replay_frame([], args.events, offset=offset)["TransactionID"].to_list())
    print(f"replaying rows {offset:,}–{offset + len(ids) - 1:,} (continuing the stream)")

    started = time.time()
    run(limit=args.events, offset=offset, bootstrap=bootstrap,
        speed=float(os.getenv("TIME_COMPRESSION", 8800)),
        max_rate=float(os.getenv("REPLAY_RATE_TX_PER_SEC", 200)))  # fmt: skip
    produce_s = time.time() - started

    deadline, spans = time.time() + args.timeout, []
    while time.time() < deadline:
        body = httpx.get(f"{args.api}/latency", params={"limit": 100_000}, timeout=30).json()
        spans = [s for s in body["spans"] if s[0] in ids]
        if len(spans) >= len(ids):
            break
        time.sleep(2)
    if len(spans) < len(ids):
        raise SystemExit(f"only {len(spans)}/{len(ids)} bench events scored within {args.timeout}s")

    stats = hop_stats(spans, list(body["fields"]))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=REPO_ROOT).stdout.strip()  # fmt: skip
    meta = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "events": len(spans),
        "throughput_tx_s": len(ids) / produce_s,
        "model_version": health.get("model_version"),
        "git_commit": commit,
        "kpi_p95_s": KPI_P95_S,
        "kpi_pass": stats["end-to-end"]["p95"] < KPI_P95_S,
    }
    out = REPO_ROOT / "artifacts" / "bench.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "hops": stats}, indent=2))
    print(markdown(stats, meta))
    print(f"\n→ {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

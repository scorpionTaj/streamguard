"""Spark Structured Streaming feature job — milestone M2 (PLAN.md §2, §6).

Contract (implement in M2, run inside docker service `spark-features`):

1. Read `transactions.raw` from Redpanda (Kafka source) as JSON.
2. Parse payloads with the same field names as `schemas.Transaction`.
3. Stateful velocity aggregation per `card_key` over the windows in
   `configs/features.yaml` (5m/1h/24h/7d): txn_count, amt_sum, amt_mean,
   uniq_merchants, time_since_last_txn — keyed on EVENT TIME
   (`event_ts`), watermark 10 minutes (configs/pipeline.yaml).
4. Join the velocity snapshot with the static features of the current event.
5. Write one Redis hash per card (HSET sg:card:<card_key> …, TTL 7 days)
   via foreachBatch with a pipelined redis client; latest-writer-wins per card.
6. Checkpoint to /opt/app/checkpoints/features (mounted volume) for exactly-once
   state recovery.

Environment (see docker-compose.yml service `spark-features`):
    KAFKA_BOOTSTRAP, REDIS_URL, FEATURES_CONFIG, CHECKPOINT_DIR

This module is imported ONLY inside the Spark container; pyspark is an
optional dependency (`uv sync --extra stream`) and must not be imported at
module import time on the host.
"""

from __future__ import annotations


def main() -> None:
    """Entry point wired to `docker compose up spark-features`. — M2"""
    raise NotImplementedError("M2: Spark Structured Streaming velocity job → Redis")


if __name__ == "__main__":
    main()

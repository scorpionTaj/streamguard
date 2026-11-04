"""Spark Structured Streaming feature job (PLAN.md §2, §6) — runs in docker service
`spark-features` (`make features`).

1. Read `transactions.raw` from Redpanda (Kafka source), parse the JSON payload.
2. Stateful per-card velocity via `applyInPandasWithState`: state keeps the card's
   events inside the longest window (+ the last event for time_since_last_txn);
   each micro-batch runs the shared numpy kernel (streamguard.velocity) over
   state + new events, ordered by (event_ts, transaction_id) = replay order.
3. foreachBatch, pipelined: one Redis hash per card (HSET sg:card:<card_key>,
   TTL 7d, latest event per card wins) + one per event (sg:tx:<transaction_id>,
   short TTL) — the scorer waits for the event's own snapshot, so serving sees
   exactly the training-time features.
4. Checkpoint to CHECKPOINT_DIR for state recovery.

State has no timeout on purpose: dropping an idle card would reset its
time_since_last_txn to the sentinel, diverging from the offline twin.

Environment: KAFKA_BOOTSTRAP, REDIS_URL, FEATURES_CONFIG, PIPELINE_CONFIG, CHECKPOINT_DIR.
pyspark is only imported inside main() — this module is never run on the host.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator

import numpy as np
import pandas as pd

from streamguard.config import (
    load_feature_spec,
    load_pipeline_config,
    redis_card_key,
    velocity_window_seconds,
)
from streamguard.velocity import velocity_features

# Velocity-state columns carried between micro-batches (one row per card).
STATE_FIELDS = ("event_ts_s", "amt", "transaction_id")


def velocity_columns(spec: dict) -> list[str]:
    """Output feature names in kernel order (matches offline build_features)."""
    metrics = spec["velocity"]["metrics"]
    windowed = [m for m in metrics if m != "time_since_last_txn"]
    names = [f"{m}_{w}" for w in spec["velocity"]["windows"] for m in windowed]
    return names + (["time_since_last_txn"] if "time_since_last_txn" in metrics else [])


def update_card(
    card_key: str,
    history: tuple[list[float], list[float], list[int]] | None,
    new_events: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, tuple[list[float], list[float], list[int]]]:
    """Pure state transition for one card: (state, new events) → (feature rows, new state).

    Kept free of Spark types so it is unit-testable on the host.
    """
    windows = velocity_window_seconds(spec)
    cfg = spec["velocity"]
    new_events = new_events.sort_values(["event_ts_s", "transaction_id"], kind="stable")
    past = pd.DataFrame(dict(zip(STATE_FIELDS, history or ([], [], []), strict=True)))
    events = pd.concat([past, new_events[list(STATE_FIELDS)]], ignore_index=True)

    feats = velocity_features(
        np.full(len(events), card_key),
        events["event_ts_s"].to_numpy(float),
        events["amt"].to_numpy(float),
        windows,
        cfg["metrics"],
        float(cfg.get("sentinel_no_history", -1.0)),
    )
    fresh = slice(len(past), len(events))
    out = pd.DataFrame({name: values[fresh] for name, values in feats.items()})
    out.insert(0, "event_ts_s", events["event_ts_s"].to_numpy()[fresh])
    out.insert(0, "transaction_id", events["transaction_id"].to_numpy()[fresh])
    out.insert(0, "card_key", card_key)

    # Keep what a future event (t ≥ newest) can still see: t_j > newest − longest window,
    # which always includes the newest event itself (needed for time_since_last_txn).
    horizon = events["event_ts_s"].iloc[-1] - max(windows.values())
    kept = events[events["event_ts_s"] > horizon]
    state = (
        kept["event_ts_s"].astype(float).tolist(),
        kept["amt"].astype(float).tolist(),
        kept["transaction_id"].astype(int).tolist(),
    )
    return out, state


def latest_per_card(batch: pd.DataFrame) -> pd.DataFrame:
    """Latest-writer-wins: the last event (replay order) of each card in a micro-batch."""
    ordered = batch.sort_values(["card_key", "event_ts_s", "transaction_id"], kind="stable")
    return ordered.groupby("card_key", sort=False).tail(1)


def redis_mapping(row: pd.Series, feature_names: list[str]) -> dict[str, str]:
    """Hash fields for one snapshot. repr() round-trips floats exactly (parity needs 1e-6)."""
    mapping = {"transaction_id": str(int(row["transaction_id"])),
               "event_ts_s": repr(float(row["event_ts_s"]))}  # fmt: skip
    mapping.update({name: repr(float(row[name])) for name in feature_names})
    return mapping


def main() -> None:
    """Entry point wired to `docker compose up spark-features`."""
    import redis
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.streaming.state import GroupStateTimeout
    from pyspark.sql.types import (
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    spec = load_feature_spec(os.environ.get("FEATURES_CONFIG", "configs/features.yaml"))
    pipeline = load_pipeline_config(os.environ.get("PIPELINE_CONFIG", "configs/pipeline.yaml"))
    job_cfg, redis_cfg = pipeline["features_job"], pipeline["redis"]
    feature_names = velocity_columns(spec)
    ttl_seconds = int(redis_cfg["ttl_days"]) * 86400

    spark = (
        SparkSession.builder.appName("streamguard-features")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    payload = StructType([
        StructField("transaction_id", LongType()),
        StructField("event_ts", StringType()),
        StructField("card_key", StringType()),
        StructField("transaction_amt", DoubleType()),
    ])  # fmt: skip
    events = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", os.environ["KAFKA_BOOTSTRAP"])
        # a pattern (not `subscribe`) tolerates the topic not existing yet, e.g. after `make reset`
        .option("subscribePattern", re.escape(pipeline["topics"]["raw"]))
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", job_cfg["max_offsets_per_trigger"])
        .option("failOnDataLoss", "false")
        .load()
        .select(F.from_json(F.col("value").cast("string"), payload).alias("tx"))
        .select("tx.*")
        .withColumn("event_time", F.to_timestamp("event_ts"))
        .withColumn("event_ts_s", F.col("event_time").cast("double"))
        .withColumnRenamed("transaction_amt", "amt")
        .withWatermark("event_time", job_cfg["watermark"])
    )

    output_schema = StructType(
        [StructField("card_key", StringType()), StructField("transaction_id", LongType()),
         StructField("event_ts_s", DoubleType())]
        + [StructField(name, DoubleType()) for name in feature_names]
    )  # fmt: skip
    # One JSON string, not ArrayType columns: array-typed group state fails to pickle
    # between JVM and Python ("couldn't introspect javabean"). json floats round-trip exactly.
    state_schema = StructType([StructField("history_json", StringType())])

    def per_card(key: tuple, pdfs: Iterator[pd.DataFrame], state) -> Iterator[pd.DataFrame]:
        history = json.loads(state.get[0]) if state.exists else None
        new_events = pd.concat(list(pdfs), ignore_index=True)
        out, new_state = update_card(key[0], history, new_events, spec)
        state.update((json.dumps(new_state),))
        yield out[output_schema.fieldNames()]

    features = events.groupBy("card_key").applyInPandasWithState(
        per_card, output_schema, state_schema, "append", GroupStateTimeout.NoTimeout
    )

    redis_url = os.environ["REDIS_URL"]

    def write_batch(batch_df, batch_id: int) -> None:
        rows = batch_df.toPandas()
        if rows.empty:
            return
        client = redis.Redis.from_url(redis_url)
        with client.pipeline(transaction=False) as pipe:
            for _, row in rows.iterrows():
                key = f"{redis_cfg['tx_key_prefix']}{int(row['transaction_id'])}"
                pipe.hset(key, mapping=redis_mapping(row, feature_names))
                pipe.expire(key, int(redis_cfg["tx_ttl_seconds"]))
            for _, row in latest_per_card(rows).iterrows():
                key = redis_card_key(redis_cfg["key_prefix"], row["card_key"])
                pipe.hset(key, mapping=redis_mapping(row, feature_names))
                pipe.expire(key, ttl_seconds)
            pipe.execute()
        print(f"batch {batch_id}: {len(rows)} events → {rows['card_key'].nunique()} cards")

    query = (
        features.writeStream.foreachBatch(write_batch)
        .option("checkpointLocation", os.environ.get("CHECKPOINT_DIR", job_cfg["checkpoint_dir"]))
        .trigger(processingTime=job_cfg["trigger"])
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()

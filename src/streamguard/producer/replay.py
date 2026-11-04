"""Replay producer — streams the IEEE-CIS dataset into `transactions.raw` in
`TransactionDT` order with a compressed synthetic wall clock (PLAN.md §7).

Usage: REPLAY_LIMIT=1000 make replay   ·   python -m streamguard.producer.replay --limit 1000 --speed 0
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from streamguard.config import REPO_ROOT, load_feature_spec, load_pipeline_config
from streamguard.schemas import Transaction, derive_card_key

log = logging.getLogger("streamguard.replay")

# All event_ts values are derived from this reference; the dataset's TransactionDT
# is a relative timedelta in seconds (not a wall-clock timestamp).
REF_DATETIME = datetime(2025, 1, 1, tzinfo=UTC)

DEFAULT_TIME_COMPRESSION = 8800.0  # ≈6 months of simulated time in ~49 min at 200 tx/s
DEFAULT_MAX_RATE_TX_PER_SEC = 200.0

DEFAULT_TRANSACTION_CSV = REPO_ROOT / "data" / "train_transaction.csv"
DEFAULT_IDENTITY_CSV = REPO_ROOT / "data" / "train_identity.csv"

_M_FLAGS = [f"M{i}" for i in range(1, 10)]
_C_FEATURES = [f"C{i}" for i in range(1, 15)]
_D_FEATURES = [f"D{i}" for i in range(1, 16)]
_ID_COLS = ["card2", "card3", "card5", "addr1", "addr2"]
_FLOAT_COLS = ["TransactionDT", "TransactionAmt", *_ID_COLS, "dist1", "dist2",
               *_C_FEATURES, *_D_FEATURES]  # fmt: skip
_STR_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", *_M_FLAGS]


def to_event_ts(transaction_dt_seconds: float, ref: datetime = REF_DATETIME) -> datetime:
    """Map a dataset-relative TransactionDT (seconds) onto the synthetic wall clock.

    Monotonic by construction: TransactionDT is non-decreasing in the CSV order.
    """
    return ref + timedelta(seconds=transaction_dt_seconds)


def pace_delay(
    dt_delta_seconds: float,
    time_compression: float = DEFAULT_TIME_COMPRESSION,
    max_rate_tx_per_sec: float = DEFAULT_MAX_RATE_TX_PER_SEC,
) -> float:
    """Seconds to sleep before sending the next event.

    sleep = (simulated gap) / time_compression, floored at 1/max_rate so the
    producer never exceeds the configured throughput cap (rate limit wins over
    compression for dense bursts).
    """
    if dt_delta_seconds < 0:
        raise ValueError("dt_delta_seconds must be non-negative (events must be ordered)")
    if time_compression <= 0:
        raise ValueError("time_compression must be positive")
    if max_rate_tx_per_sec <= 0:
        raise ValueError("max_rate_tx_per_sec must be positive")
    simulated = dt_delta_seconds / time_compression
    return max(simulated, 1.0 / max_rate_tx_per_sec)


def _num(value: float | None) -> float | None:
    """CSV null/NaN → None; everything else → float."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _flag(value: str | None) -> bool | None:
    """IEEE-CIS match flags are 'T'/'F'; M4 holds categories (M0/M1/M2) → not a bool."""
    return {"T": True, "F": False}.get(value) if value is not None else None


def row_to_transaction(row: dict, v_subset: list[str]) -> Transaction:
    """One joined IEEE-CIS row (raw CSV column names) → the `transactions.raw` contract."""
    v = {name: float(row[name]) for name in v_subset if _num(row.get(name)) is not None}
    return Transaction(
        transaction_id=int(row["TransactionID"]),
        event_ts=to_event_ts(row["TransactionDT"]),
        card_key=derive_card_key(row["card1"], row["card2"], row["card5"], row["addr1"]),
        transaction_amt=float(row["TransactionAmt"]),
        product_cd=row["ProductCD"],
        card1=row.get("card1"),
        **{c: _num(row.get(c)) for c in _ID_COLS},
        card4=row.get("card4"),
        card6=row.get("card6"),
        dist1=_num(row.get("dist1")),
        dist2=_num(row.get("dist2")),
        p_emaildomain=row.get("P_emaildomain"),
        r_emaildomain=row.get("R_emaildomain"),
        **{m.lower(): _flag(row.get(m)) for m in _M_FLAGS},
        **{c.lower(): _num(row.get(c)) or 0.0 for c in _C_FEATURES},
        **{d.lower(): _num(row.get(d)) for d in _D_FEATURES},
        v=v,
        device_type=row.get("DeviceType"),
        device_info=row.get("DeviceInfo"),
        is_fraud=int(row["isFraud"]),
    )


def load_replay_frame(
    v_subset: list[str],
    limit: int = 0,
    transaction_csv: Path = DEFAULT_TRANSACTION_CSV,
    identity_csv: Path = DEFAULT_IDENTITY_CSV,
    offset: int = 0,
) -> pl.DataFrame:
    """Contract columns of the transaction table, identity-joined, in TransactionDT order."""
    columns = ["TransactionID", "isFraud", "card1", *_FLOAT_COLS, *_STR_COLS, *v_subset]
    overrides = {c: pl.Float64 for c in _FLOAT_COLS + v_subset} | {c: pl.Utf8 for c in _STR_COLS}
    tx = pl.scan_csv(transaction_csv, schema_overrides=overrides).select(columns)
    if identity_csv.exists():
        identity = pl.scan_csv(identity_csv, schema_overrides={"DeviceInfo": pl.Utf8}).select(
            "TransactionID", "DeviceType", "DeviceInfo"
        )
        tx = tx.join(identity, on="TransactionID", how="left")
    else:
        log.warning("%s missing — device_type/device_info will be null", identity_csv)
        tx = tx.with_columns(DeviceType=pl.lit(None, pl.Utf8), DeviceInfo=pl.lit(None, pl.Utf8))
    tx = tx.sort("TransactionDT", "TransactionID", maintain_order=True)
    if offset or limit > 0:
        tx = tx.slice(offset, limit if limit > 0 else None)
    return tx.collect()


def iter_transactions(frame: pl.DataFrame, v_subset: list[str]) -> Iterator[Transaction]:
    for row in frame.iter_rows(named=True):
        yield row_to_transaction(row, v_subset)


def ensure_topics(bootstrap: str, pipeline: dict) -> None:
    """Create raw/scored/alerts topics (idempotent) with the configured partitions."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    kafka = pipeline["kafka"]
    new = [
        NewTopic(
            name, num_partitions=kafka["partitions"], replication_factor=kafka["replication_factor"]
        )  # fmt: skip
        for name in pipeline["topics"].values()
        if name not in existing
    ]
    for name, future in admin.create_topics(new).items() if new else ():
        future.result()
        log.info("created topic %s", name)


def run(
    limit: int = 0,
    speed: float = DEFAULT_TIME_COMPRESSION,
    max_rate: float = DEFAULT_MAX_RATE_TX_PER_SEC,
    bootstrap: str = "localhost:9092",
    transaction_csv: Path = DEFAULT_TRANSACTION_CSV,
    offset: int = 0,
) -> int:
    """Replay `limit` rows (0 = all 590,540) into `transactions.raw`. `speed=0` = max rate.

    `offset` skips that many rows (in replay order) to continue an earlier replay.

    Returns the number of events delivered.
    """
    from confluent_kafka import Producer

    pipeline = load_pipeline_config()
    v_subset = load_feature_spec().get("v_subset") or []
    ensure_topics(bootstrap, pipeline)

    frame = load_replay_frame(v_subset, limit, transaction_csv, offset=offset)
    log.info("replaying %d rows (speed=%s, max_rate=%s tx/s)", frame.height, speed, max_rate)

    cfg = pipeline["kafka"]["producer"]
    producer = Producer({
        "bootstrap.servers": bootstrap,
        "linger.ms": cfg["linger_ms"],
        "compression.type": cfg["compression"],
        "batch.size": cfg["batch_size"],
        "enable.idempotence": True,
    })  # fmt: skip
    topic = pipeline["topics"]["raw"]
    delivered = failed = 0

    def on_delivery(err, _msg) -> None:
        nonlocal delivered, failed
        if err is None:
            delivered += 1
        else:
            failed += 1
            log.error("delivery failed: %s", err)

    # Absolute schedule (not sleep-per-event) so per-event overhead doesn't erode the rate.
    next_send = time.perf_counter()
    prev_dt: float | None = None
    for tx, dt in zip(iter_transactions(frame, v_subset), frame["TransactionDT"], strict=True):
        if speed > 0 and prev_dt is not None:
            next_send += pace_delay(dt - prev_dt, speed, max_rate)
            if (wait := next_send - time.perf_counter()) > 0:
                time.sleep(wait)
        prev_dt = dt
        producer.produce(topic, key=tx.card_key, value=tx.model_dump_json(),
                         on_delivery=on_delivery)  # fmt: skip
        producer.poll(0)
    producer.flush(30)
    log.info("delivered %d / %d events to %s (%d failed)", delivered, frame.height, topic, failed)
    if failed or delivered != frame.height:
        raise RuntimeError(f"{frame.height - delivered} events not delivered")
    return delivered


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Replay IEEE-CIS into transactions.raw")
    parser.add_argument("--limit", type=int, default=int(os.getenv("REPLAY_LIMIT", "0")))
    parser.add_argument("--speed", type=float,
                        default=float(os.getenv("TIME_COMPRESSION", DEFAULT_TIME_COMPRESSION)),
                        help="time compression factor; 0 = no pacing (benchmark max rate)")  # fmt: skip
    parser.add_argument("--max-rate", type=float,
                        default=float(os.getenv("REPLAY_RATE_TX_PER_SEC", DEFAULT_MAX_RATE_TX_PER_SEC)))  # fmt: skip
    parser.add_argument("--source", type=Path, default=DEFAULT_TRANSACTION_CSV)
    parser.add_argument("--offset", type=int, default=int(os.getenv("REPLAY_OFFSET", "0")),
                        help="skip this many rows (continue an earlier replay)")  # fmt: skip
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(args.limit, args.speed, args.max_rate,
        os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"), args.source, args.offset)  # fmt: skip


if __name__ == "__main__":
    main()

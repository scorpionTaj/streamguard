"""Replay producer — streams the IEEE-CIS dataset into `transactions.raw` in
`TransactionDT` order with a compressed synthetic wall clock (PLAN.md §7).

Implemented: pure clock/pacing helpers (tested).
Milestone M0: `run()` — read CSV (polars), build Transaction events,
produce with confluent-kafka keyed by card_key, pace with pace_delay().
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# All event_ts values are derived from this reference; the dataset's TransactionDT
# is a relative timedelta in seconds (not a wall-clock timestamp).
REF_DATETIME = datetime(2025, 1, 1, tzinfo=UTC)

DEFAULT_TIME_COMPRESSION = 8800.0  # ≈6 months of simulated time in ~49 min at 200 tx/s
DEFAULT_MAX_RATE_TX_PER_SEC = 200.0


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


def run(limit: int = 0, source_csv: str = "data/train_transaction.csv") -> None:
    """Replay `limit` rows (0 = all 590,540) into `transactions.raw`. — M0"""
    raise NotImplementedError("M0: replay loop (polars read → Transaction → confluent-kafka)")


if __name__ == "__main__":
    run()

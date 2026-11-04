"""Per-card trailing-window velocity kernel (PLAN.md §6) — numpy only.

Shared by the offline polars twin (offline/features.py) and the Spark streaming
job (streaming/features_job.py, runs inside the Spark container), so both paths
execute the same arithmetic; the parity test then checks the streaming plumbing
(ordering, state, Redis) rather than two re-implementations.

Semantics: for event i of card k at event time t, window W covers card k's events
j ≤ i (replay order) with t_j > t − W — the snapshot *after* processing event i.
`time_since_last_txn` is window-independent (one column), sentinel when no history.
"""

from __future__ import annotations

import numpy as np

METRICS = ("txn_count", "amt_sum", "amt_mean", "time_since_last_txn")


def velocity_features(
    card_key: np.ndarray,
    event_ts_s: np.ndarray,
    amount: np.ndarray,
    windows: dict[str, int],
    metrics: list[str],
    sentinel: float,
) -> dict[str, np.ndarray]:
    """Trailing-window aggregates per card, inputs in replay order (non-decreasing time).

    Sums run on integer thousandths (amounts carry ≤ 3 decimals) so the result is
    exact and independent of summation order — a prerequisite for 1e-6 parity.
    """
    n = card_key.size
    _, codes = np.unique(card_key, return_inverse=True)
    order = np.lexsort((np.arange(n), codes))  # by card, then replay order
    ts = event_ts_s[order] - event_ts_s.min()
    mills = np.round(amount[order] * 1000).astype(np.int64)
    gap = float(ts.max()) + max(windows.values(), default=0) + 1.0
    # Composite sort key: separates cards by more than any window → one searchsorted.
    key = codes[order] * gap + ts
    idx = np.arange(n)
    cum = np.concatenate([[0], np.cumsum(mills)])

    out: dict[str, np.ndarray] = {}
    for name, seconds in windows.items():
        start = np.searchsorted(key, key - seconds, side="right")
        count = (idx - start + 1).astype(float)
        total = (cum[idx + 1] - cum[start]) / 1000.0
        computed = {"txn_count": count, "amt_sum": total, "amt_mean": total / count}
        for metric in metrics:
            if metric in computed:
                out[f"{metric}_{name}"] = computed[metric]
    if "time_since_last_txn" in metrics:
        same_card = np.concatenate([[False], codes[order][1:] == codes[order][:-1]])
        since = np.where(same_card, np.diff(ts, prepend=0.0), sentinel)
        out["time_since_last_txn"] = since
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown velocity metrics in features.yaml: {sorted(unknown)}")

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = idx  # back to replay order
    return {name: values[inverse] for name, values in out.items()}

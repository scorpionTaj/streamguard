"""Offline (polars) feature twin (PLAN.md §6, §8).

Features are built from `schemas.Transaction` events — the exact payloads the
replay producer publishes — so the online (Spark) and offline paths share one
input contract. Every feature listed in configs/features.yaml is built here; an
unknown name fails loudly instead of being silently skipped.

Velocity semantics (the online job must match, parity test checks 1e-6):
for event i of card k at event time t, window W covers card k's events j ≤ i
(replay order) with t_j > t − W — i.e. the snapshot *after* processing event i.
`time_since_last_txn` is window-independent (one column), sentinel when no history.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

import numpy as np
import polars as pl

from streamguard.config import load_feature_spec, velocity_window_seconds
from streamguard.schemas import Transaction

# Known mailbox providers in IEEE-CIS email domains; everything else → "other".
_EMAIL_GROUPS: tuple[str, ...] = (
    "gmail",
    "yahoo",
    "hotmail",
    "outlook",
    "aol",
    "protonmail",
    "icloud",
    "comcast",
    "verizon",
    "me.com",
    "web.de",
    "mail.com",
)

NO_EMAIL = "unknown"
OTHER_EMAIL = "other"

# Fixed vocabularies → stable column sets / integer codes regardless of the data slice.
EMAIL_GROUP_CODES: tuple[str, ...] = (NO_EMAIL, OTHER_EMAIL, *_EMAIL_GROUPS)
PRODUCT_CODES = ("W", "C", "R", "H", "S")
CARD4_VALUES = ("visa", "mastercard", "american express", "discover")
CARD6_VALUES = ("debit", "credit")
DEVICE_TYPES = ("desktop", "mobile")

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def email_domain_group(domain: str | None) -> str:
    """Map a raw email domain (e.g. 'gmail.com', 'mail.yahoo.fr') to its group.

    >>> email_domain_group("mail.yahoo.fr")
    'yahoo'
    >>> email_domain_group("somecorp.cz")
    'other'
    """
    if not domain or not domain.strip():
        return NO_EMAIL
    cleaned = domain.strip().lower()
    for group in _EMAIL_GROUPS:
        if group in cleaned:
            return group
    return OTHER_EMAIL


# ── Contract events → frame ──────────────────────────────────────────────────


def transactions_frame(txs: Iterable[Transaction], v_subset: list[str]) -> pl.DataFrame:
    """Flatten contract events into one row each; `v` becomes columns (missing → null)."""
    rows = []
    for tx in txs:
        row = tx.model_dump(exclude={"v"})
        row["event_ts_s"] = (tx.event_ts - _EPOCH).total_seconds()
        row.update({name: tx.v.get(name) for name in v_subset})
        rows.append(row)
    return pl.DataFrame(rows, schema_overrides={name: pl.Float64 for name in v_subset})


def load_transactions(limit: int = 0, v_subset: list[str] | None = None) -> pl.DataFrame:
    """IEEE-CIS CSV → contract events (same mapping as the replay producer) → frame."""
    from streamguard.producer.replay import iter_transactions, load_replay_frame

    if v_subset is None:
        v_subset = load_feature_spec().get("v_subset") or []
    raw = load_replay_frame(v_subset, limit)
    return transactions_frame(iter_transactions(raw, v_subset), v_subset)


# ── Velocity (per-card trailing windows) ─────────────────────────────────────


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
    unknown = set(metrics) - {"txn_count", "amt_sum", "amt_mean", "time_since_last_txn"}
    if unknown:
        raise ValueError(f"unknown velocity metrics in features.yaml: {sorted(unknown)}")

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = idx  # back to replay order
    return {name: values[inverse] for name, values in out.items()}


# ── Static features (name in features.yaml → expressions) ────────────────────


def _onehot(column: str, values: tuple[str, ...], prefix: str) -> list[pl.Expr]:
    return [(pl.col(column) == v).fill_null(False).cast(pl.Float64).alias(
        f"{prefix}_{v.replace(' ', '_')}") for v in values]  # fmt: skip


def _email_code(column: str, alias: str) -> list[pl.Expr]:
    codes = {g: float(i) for i, g in enumerate(EMAIL_GROUP_CODES)}
    return [
        pl.col(column)
        .map_elements(lambda d: codes[email_domain_group(d)], return_dtype=pl.Float64)
        .fill_null(codes[NO_EMAIL])
        .alias(alias)
    ]


def _cyclical(unit_seconds: float, period: float, name: str) -> list[pl.Expr]:
    phase = (pl.col("event_ts_s") / unit_seconds) % period * (2 * math.pi / period)
    return [phase.sin().alias(f"{name}_sin"), phase.cos().alias(f"{name}_cos")]


_STATIC: dict[str, Callable[[], list[pl.Expr]]] = {
    "transaction_amt": lambda: [pl.col("transaction_amt")],
    "log_transaction_amt": lambda: [pl.col("transaction_amt").log1p().alias("log_transaction_amt")],
    "amt_over_card_mean_24h": lambda: [
        (pl.col("transaction_amt") / pl.col("amt_mean_24h")).alias("amt_over_card_mean_24h")
    ],
    "product_cd": lambda: _onehot("product_cd", PRODUCT_CODES, "product_cd"),
    "card4_onehot": lambda: _onehot("card4", CARD4_VALUES, "card4"),
    "card6_onehot": lambda: _onehot("card6", CARD6_VALUES, "card6"),
    "dist1": lambda: [pl.col("dist1")],
    "dist2": lambda: [pl.col("dist2")],
    "log_dist1": lambda: [pl.col("dist1").log1p().alias("log_dist1")],
    "p_emaildomain_group": lambda: _email_code("p_emaildomain", "p_emaildomain_group"),
    "r_emaildomain_group": lambda: _email_code("r_emaildomain", "r_emaildomain_group"),
    "device_type_onehot": lambda: _onehot("device_type", DEVICE_TYPES, "device_type"),
    "hour_sin": lambda: _cyclical(3600, 24, "hour")[:1],
    "hour_cos": lambda: _cyclical(3600, 24, "hour")[1:],
    # 2025-01-01 (REF_DATETIME) is a Wednesday; phase only needs to be consistent.
    "dow_sin": lambda: _cyclical(86400, 7, "dow")[:1],
    "dow_cos": lambda: _cyclical(86400, 7, "dow")[1:],
}


def build_features(frame: pl.DataFrame, spec: dict | None = None) -> pl.DataFrame:
    """Contract frame → [transaction_id, event_ts, is_fraud, *feature_columns(spec)]."""
    spec = spec or load_feature_spec()
    velocity_cfg = spec["velocity"]
    velocity = velocity_features(
        frame[spec["key"]].to_numpy(),
        frame["event_ts_s"].to_numpy(),
        frame["transaction_amt"].to_numpy(),
        velocity_window_seconds(spec),
        velocity_cfg["metrics"],
        float(velocity_cfg.get("sentinel_no_history", -1.0)),
    )
    unknown = [name for name in spec["static"] if name not in _STATIC]
    if unknown:
        raise ValueError(f"no offline implementation for static features: {unknown}")

    flags = [pl.col(m).cast(pl.Float64) for m in spec["m_flags"]]
    counts = [pl.col(c).cast(pl.Float64) for c in spec["c_features"]]
    passthrough = [
        pl.col(c).cast(pl.Float64) for c in [*spec.get("raw_ids", []), *spec.get("d_features", [])]
    ]
    v_cols = [pl.col(v).cast(pl.Float64) for v in spec.get("v_subset") or []]
    static = [e for name in spec["static"] for e in _STATIC[name]()]
    return frame.with_columns(**velocity).select(
        "transaction_id", "event_ts", "is_fraud",
        *static, *flags, *counts, *passthrough, *v_cols, *velocity.keys(),
    )  # fmt: skip


def feature_columns(features: pl.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ("transaction_id", "event_ts", "is_fraud")]


# ── Temporal split & V-feature selection ─────────────────────────────────────

SPLIT_QUANTILES = (2 / 3, 5 / 6)  # train | val | test by event time (PLAN.md §5)


def temporal_split(frame: pl.DataFrame, time_col: str = "event_ts") -> dict[str, pl.DataFrame]:
    """Strict time-ordered split at the 66.7 % / 83.3 % event-time quantiles. Never shuffled."""
    physical = frame[time_col].to_physical()  # datetime → int µs; numeric unchanged
    q_val, q_test = (physical.quantile(q, interpolation="lower") for q in SPLIT_QUANTILES)
    t = pl.col(time_col).to_physical()
    return {
        "train": frame.filter(t < q_val),
        "val": frame.filter((t >= q_val) & (t < q_test)),
        "test": frame.filter(t >= q_test),
    }


def discrete_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """MI (nats) between a quantile-binned feature (NaN = its own bin) and a binary label."""
    finite = ~np.isnan(x)
    codes = np.full(x.size, bins, dtype=np.int64)  # NaN bin
    if finite.any():
        edges = np.unique(np.quantile(x[finite], np.linspace(0, 1, bins + 1))[1:-1])
        codes[finite] = np.searchsorted(edges, x[finite], side="right")
    joint = np.zeros((bins + 1, 2))
    np.add.at(joint, (codes, y.astype(np.int64)), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def select_v_subset(v_train: pl.DataFrame, y_train: np.ndarray, k: int = 30) -> list[str]:
    """Top-k V columns by mutual information with the label — call on the TRAIN split only."""
    scores = {
        c: discrete_mutual_information(v_train[c].cast(pl.Float64).to_numpy(), y_train)
        for c in v_train.columns
    }
    return sorted(scores, key=lambda c: (-scores[c], c))[:k]


def build_training_frame(limit: int = 0) -> pl.DataFrame:
    """Full IEEE-CIS → feature frame per configs/features.yaml (velocity over full history)."""
    return build_features(load_transactions(limit))

"""Champion vs challenger canary (PLAN.md §10).

The scorer scores every event with both registry aliases; only the champion
drives alerts. Per event we record the score delta (champion − challenger) and
whether both models make the same alert decision. Promotion is allowed when the
p95 |delta| over the canary window is below TOLERANCE (registry.promote checks it).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from prometheus_client import Counter, Histogram

TOLERANCE = 0.02  # p95 |champion − challenger| (PLAN.md §13)
MIN_EVENTS = 1000  # don't judge a canary on a handful of events

CANARY_DELTA = Histogram("streamguard_canary_delta", "|champion − challenger| score per event",
                         buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0))  # fmt: skip
CANARY_DECISIONS = Counter("streamguard_canary_decisions_total",
                           "Alert decisions compared", ["agree"])  # fmt: skip


@dataclass
class CanaryWindow:
    """Recent per-event comparisons (bounded), shared between scorer thread and API."""

    deltas: deque = field(default_factory=lambda: deque(maxlen=50_000))
    agreements: deque = field(default_factory=lambda: deque(maxlen=50_000))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, champion: np.ndarray, challenger: np.ndarray,
               champion_alerts: np.ndarray, challenger_alerts: np.ndarray) -> None:  # fmt: skip
        delta = np.asarray(champion, float) - np.asarray(challenger, float)
        agree = np.asarray(champion_alerts, bool) == np.asarray(challenger_alerts, bool)
        for value in np.abs(delta):
            CANARY_DELTA.observe(value)
        CANARY_DECISIONS.labels("true").inc(int(agree.sum()))
        CANARY_DECISIONS.labels("false").inc(int((~agree).sum()))
        with self.lock:
            self.deltas.extend(delta.tolist())
            self.agreements.extend(agree.tolist())

    def summary(self) -> dict:
        with self.lock:
            deltas, agreements = np.array(self.deltas), np.array(self.agreements, bool)
        return summarize(deltas, agreements)


def summarize(deltas: np.ndarray, agreements: np.ndarray) -> dict:
    """Canary verdict inputs: size, p95/mean |delta|, alert-decision agreement."""
    if deltas.size == 0:
        return {"events": 0, "p95_abs_delta": None, "mean_abs_delta": None,
                "alert_agreement": None, "within_tolerance": False}  # fmt: skip
    p95 = float(np.quantile(np.abs(deltas), 0.95))
    return {
        "events": int(deltas.size),
        "p95_abs_delta": p95,
        "mean_abs_delta": float(np.abs(deltas).mean()),
        "mean_delta": float(deltas.mean()),
        "alert_agreement": float(agreements.mean()),
        "within_tolerance": bool(deltas.size >= MIN_EVENTS and p95 < TOLERANCE),
    }

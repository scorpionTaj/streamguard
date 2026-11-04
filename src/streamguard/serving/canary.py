"""Champion vs challenger canary — milestone M6 (PLAN.md §10).

Contract: score every event with both registry aliases; champion drives alerts.
Log per-event delta (champion − challenger) to Prometheus
(streamguard_canary_delta) and sample deltas to MLflow; when p95 |delta| < 0.02
over the canary window, `registry.promote()` is allowed to move the alias.
"""

from __future__ import annotations


def score_both(event_features, champion, challenger) -> dict[str, float]:
    """Return {'champion': s1, 'challenger': s2, 'delta': s1 - s2}. — M6"""
    raise NotImplementedError("M6: dual scoring + delta logging")

"""Scorer consumer loop — milestone M3 (PLAN.md §2, §3).

Contract: one thread/task, consumer group `scorer` on `transactions.raw`:
    poll batch → for each event:
        1. HGETALL sg:card:<card_key> (Redis, ≤ 5 ms; empty → cold-start zeros)
        2. build feature vector (order fixed by configs/features.yaml)
        3. champion model score (MLflow pyfunc, ≤ 10 ms)
        4. publish ScoredTx → transactions.scored; if above band threshold →
           Alert → transactions.alerts (SHAP attached async, M5)
Latency spans recorded per hop for eval/latency.py; Prometheus counters/gauges
exported via main.py /metrics.
"""

from __future__ import annotations


def run() -> None:
    """Blocking scorer loop; started by `make api` lifespan. — M3"""
    raise NotImplementedError("M3: kafka consumer → redis enrich → score → publish")

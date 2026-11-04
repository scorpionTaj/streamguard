"""FastAPI scorer service — milestone M3 (PLAN.md §2–3, §8).

Runs on the host (`make api`, :8000). Components:

- `consumer_loop()` (serving/consumer.py): consume `transactions.raw`, enrich
  each event with the card's Redis velocity snapshot (HGETALL ≤ 5 ms), score
  with the MLflow champion (≤ 10 ms), publish ScoredTx / Alert to Kafka.
- Endpoints:
    GET /health        liveness
    GET /alerts        alerts since ?since=… (Streamlit polls every 2 s)
    GET /stats         counters: tx scored, alerts, bands
    GET /metrics       Prometheus (prometheus-client)
- Canary (M6): serving/canary.py scores with challenger too, logs delta.
- Risk bands: high / medium / review per configs/pipeline.yaml thresholds (M5).
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="StreamGuard scorer", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "consumer": "not_started_M3"}


@app.get("/alerts")
def alerts(since: str | None = None) -> dict:
    """Alert feed for the dashboard (poll every 2 s). — M3"""
    return {"alerts": [], "since": since}


@app.get("/stats")
def stats() -> dict:
    return {"scored": 0, "alerts": 0, "by_band": {}}


@app.get("/metrics")
def metrics() -> dict:
    """Prometheus exposition — wire prometheus_client in M3."""
    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)  # type: ignore[return-value]

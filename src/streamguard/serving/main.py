"""FastAPI scorer service (PLAN.md §2–3, §8) — `make api`, :8000 on the host.

The lifespan loads the MLflow `champion` alias and starts the consumer loop
(serving/consumer.py) in a background thread.
    GET /health        liveness + model version + consumer status
    GET /alerts        alerts newer than ?since=<ISO datetime> (Streamlit polls every 2 s)
    GET /stats         counters: tx scored, alerts by band, feature misses
    GET /metrics       Prometheus exposition
Canary (M6): serving/canary.py scores with the challenger too.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from streamguard.config import load_feature_spec, load_pipeline_config
from streamguard.serving.consumer import Scorer, ScorerState

log = logging.getLogger("streamguard.api")
state = ScorerState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    from streamguard.models.registry import load_alias

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    model = load_alias()  # fail fast: no champion → no scorer
    scorer = Scorer(
        model, load_feature_spec(), load_pipeline_config(), state,
        os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )  # fmt: skip
    stop = threading.Event()
    thread = threading.Thread(target=scorer.run, args=(stop,), name="scorer", daemon=True)
    thread.start()
    log.info("scoring with %s v%s", "champion", model.version)
    yield
    stop.set()
    thread.join(timeout=15)


app = FastAPI(title="StreamGuard scorer", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "error" if state.error else "ok",
        "model_version": state.model_version,
        "consumer_error": state.error,
    }


@app.get("/alerts")
def alerts(since: datetime | None = None, limit: int = Query(200, ge=1, le=2000)) -> dict:
    """Alerts scored after `since` (oldest first, at most `limit` newest)."""
    with state.lock:
        recent = [a for a in state.alerts if since is None or a.scored_at > since]
    return {"alerts": [a.model_dump(mode="json") for a in recent[-limit:]]}


@app.get("/stats")
def stats() -> dict:
    with state.lock:
        return {
            "scored": state.scored,
            "alerts": sum(state.by_band.values()),
            "by_band": dict(state.by_band),
            "feature_misses": state.feature_misses,
            "model_version": state.model_version,
        }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

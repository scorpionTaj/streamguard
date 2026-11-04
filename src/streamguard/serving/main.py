"""FastAPI scorer service (PLAN.md §2–3, §8) — `make api`, :8000 on the host.

The lifespan loads the MLflow `champion` alias and starts the consumer loop
(serving/consumer.py) in a background thread.
    GET /health        liveness + model version + consumer status
    GET /alerts        alerts newer than ?since=<ISO datetime> (Streamlit polls every 2 s)
    GET /alerts/{id}   one alert incl. async SHAP top drivers
    GET /stats         counters: tx scored, alerts by band, feature misses
    GET /canary        champion vs challenger score deltas (promotion gate)
    GET /drift         PSI per feature vs the training reference
    GET /latency       per-event spans for `make bench`
    GET /metrics       Prometheus exposition
With a `challenger` alias the scorer also scores every event with it (canary).
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from streamguard.config import load_feature_spec, load_pipeline_config
from streamguard.serving.consumer import Scorer, ScorerState

log = logging.getLogger("streamguard.api")
state = ScorerState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    from streamguard.eval.drift import DriftMonitor, load_reference, top_features
    from streamguard.models.registry import CHALLENGER, load_alias, load_optional

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    model = load_alias()  # fail fast: no champion → no scorer
    challenger = load_optional(CHALLENGER)
    reference = load_reference(model.run_id)
    drift = None
    if reference is not None:
        drift = DriftMonitor(reference, top_features(model.model.booster))
    else:
        log.warning("champion v%s has no drift reference — PSI monitoring off", model.version)
    scorer = Scorer(
        model, load_feature_spec(), load_pipeline_config(), state,
        os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        challenger=challenger, drift=drift,
    )  # fmt: skip
    stop = threading.Event()
    thread = threading.Thread(target=scorer.run, args=(stop,), name="scorer", daemon=True)
    thread.start()
    log.info("scoring with champion v%s%s", model.version,
             f", canary vs challenger v{challenger.version}" if challenger else "")  # fmt: skip
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


@app.get("/alerts/{transaction_id}")
def alert(transaction_id: int) -> dict:
    """One alert, including SHAP top drivers once the async explainer has filled them."""
    with state.lock:
        found = next(
            (a for a in reversed(state.alerts) if a.transaction_id == transaction_id), None
        )
        if found is None:
            raise HTTPException(status_code=404, detail="alert not in the recent window")
        return found.model_dump(mode="json")


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


@app.get("/canary")
def canary() -> dict:
    """Champion vs challenger over the recent window (registry.promote reads this)."""
    base = {"champion_version": state.model_version, "challenger_version": state.challenger_version}
    if state.canary is None:
        return base | {"events": 0, "within_tolerance": False, "detail": "no challenger alias"}
    return base | state.canary.summary()


@app.get("/drift")
def drift() -> dict:
    """Latest PSI per feature vs the champion's training reference (eval/drift.py)."""
    if state.drift is None:
        return {"rows": 0, "features": {}, "detail": "no drift reference for this champion"}
    return state.drift.compute()


@app.get("/latency")
def latency(limit: int = Query(20_000, ge=1, le=100_000)) -> dict:
    """Recent per-event spans for `make bench` (epoch seconds)."""
    fields = ("transaction_id", "created", "consumed", "snapshot_written",
              "features_ready", "scored", "published")  # fmt: skip
    with state.lock:
        spans = list(state.spans)[-limit:]
    return {"fields": fields, "spans": spans}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

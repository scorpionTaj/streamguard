# StreamGuard

**Real-time fraud detection with an online feature store — Kafka → Spark Structured Streaming → Redis → hybrid scoring → live alert dashboard.**

> Status: 📋 **planned** — full spec, contracts and milestones in [PLAN.md](./PLAN.md). Implementation starts at milestone M0 (`make setup` → `make up` → `make replay`).

## Why this project

- Fills the biggest gap in the current portfolio: **streaming** (everything so far is batch).
- Proves **Kafka + Spark Structured Streaming** for real, not just as skills-list items.
- Reuses proven strengths: anomaly detection, MLflow/FastAPI/Docker MLOps discipline.
- Uses **publicly labeled data** (Kaggle credit-card fraud / IEEE-CIS), so metrics are real and defensible in interviews.

## Concept

1. A producer replays labeled transaction data into Kafka, simulating live traffic (configurable rate, bursts, drift).
2. Spark Structured Streaming computes rolling features (per-card/merchant velocity, amount z-scores, time-since-last-txn) and keeps them warm in **Redis** — the online feature store.
3. Hybrid scoring: supervised **XGBoost/LightGBM** + unsupervised **autoencoder/IsolationForest** for novel-pattern detection.
4. Scored transactions become alerts on a live dashboard (WebSocket), each with **SHAP top-driver explanations**.
5. Full MLOps loop: MLflow registry, model promotion, canary scoring, drift monitor.

## Stack

| Layer | Choice |
|---|---|
| Transport | Kafka (Redpanda as lighter alternative) |
| Stream processing | Spark Structured Streaming (PySpark) |
| Online features | Redis |
| Models | XGBoost/LightGBM + autoencoder (PyTorch) |
| Serving | FastAPI |
| Dashboard | Streamlit or Next.js (WebSocket) |
| MLOps | MLflow, Docker Compose, GitHub Actions, Prometheus/Grafana |

## MVP milestones

1. **Producer + topic:** labeled transactions → Kafka with realistic timing.
2. **Streaming features:** rolling windows → Redis, with offline parity check.
3. **Hybrid scorer:** supervised + unsupervised, ensemble threshold policy.
4. **Alert loop:** FastAPI scoring service, WebSocket dashboard, SHAP explanations per alert.
5. **Ops:** MLflow registry, canary rollout, drift monitor, CI.

## KPIs (targets)

- **AUPRC ≥ 0.80** on the labeled test split (imbalanced data → PR-AUC, not accuracy)
- **p95 end-to-end detection latency < 5s** at 200 tx/s
- **Online/offline feature parity δ < 1e-6** on sampled joins
- Measured **false-positive rate at fixed recall** (the operational trade-off interviewers ask about)

## CV bullet (target)

"Built streaming fraud detection (Kafka → Spark Structured Streaming → Redis online features) scoring 200 tx/s at <5s p95; hybrid XGBoost + autoencoder with AUPRC 0.8x, live WebSocket alert dashboard with SHAP explanations and canary model rollout."

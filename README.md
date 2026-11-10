# StreamGuard

**Real-time fraud detection with an online feature store — Kafka → Spark Structured Streaming → Redis → XGBoost scoring → live alert dashboard.**

> **v0.1.0** — runs end to end on one laptop (Docker Compose + host processes).
> [Measurements in detail](./docs/MEASUREMENTS.md) · [one-page write-up](./docs/WRITEUP.md) · [spec and milestone log](./PLAN.md)

![StreamGuard demo: live alert feed, SHAP drill-down, Grafana](docs/demo.gif)

## Results

The data is IEEE-CIS (590,540 real, labeled e-commerce transactions), evaluated on a strict temporal split with no shuffling. Each number comes from the command in the last column.

| KPI | Target | Result | Proof |
| --- | --- | --- | --- |
| **AUPRC, temporal test** | ≥ 0.48 | **0.489** (val 0.550) | `make train` → MLflow |
| p95 end-to-end latency @ ~180 tx/s | < 5 s | **1.47 s** (p50 1.01 s, p99 1.57 s) | `make bench` |
| Same KPI at 2.5× the rate (~500 tx/s) | < 5 s | **1.89 s**, no dropped events | `make bench` |
| Online/offline feature parity | max \|Δ\| < 1e-6 | **0** (1 k cards × 13 features, 60 k events) | `make parity` |
| Live vs offline score parity | — | **0** (5 k events) | M3 check |
| Recall @ FPR 5 % · precision@50 | report | 0.590 · 1.00 (test) | `make train` |
| Canary gate | promote only if p95 \|Δscore\| < 0.02 | challenger at 0.118 → **refused** | `make promote` |
| Drift monitor (calibrated PSI) | no false alarms in-distribution | **0/46** train/val windows flagged; injected shifts caught 6–8/8 | `make drift` |
| Scorer killed mid-stream (`kill -9`, once and three times in a 12 k replay) | no lost events | **0 missing, 0 duplicates** | `make resilience-measure` |

**About the AUPRC target.** Published XGBoost PR-AUC figures (0.78–0.88) come from other card-fraud datasets. On IEEE-CIS with a temporal split, XGBoost using all 431 raw columns reaches only 0.510 on test. StreamGuard's 111 streaming-safe features reach 96 % of that ceiling; the target was re-anchored to it in [PLAN.md §8](./PLAN.md#8-evaluation-protocol).

## Architecture

![architecture](docs/architecture.svg)

1. **Replay producer:** streams IEEE-CIS rows in event-time order on a compressed clock (≈ 6 months in ≈ 49 min, capped at 200 tx/s), keyed by a hashed card so per-card order holds within a partition.
2. **Spark Structured Streaming:** keeps exact trailing-window velocity per card (5 m / 1 h / 24 h / 7 d counts, sums, means) and writes a latest-per-card and a per-event snapshot to Redis.
3. **FastAPI scorer:** waits for each event's own snapshot, builds features with the offline twin's code, scores with the MLflow `champion` (and the `challenger` as a canary), and publishes banded alerts with SHAP drivers.
4. **Streamlit dashboard:** live alert feed with drill-down, plus model health (canary verdict, drift per feature, live precision by band).
5. **Prometheus + Grafana:** throughput, latency percentiles, consumer lag, snapshot hit-rate, drift ratios, canary delta.

## Design decisions

- **One contract, one velocity function.** The producer, the offline twin and the Spark job share one schema and one numpy velocity function ([velocity.py](src/streamguard/velocity.py)), so the parity check tests the streaming plumbing, not two copies of a formula. Sums run in integer thousandths, so parity is exactly 0.
- **Serving equals training.** Training features include the current event, so Spark writes a short-lived per-event snapshot and the scorer waits for it. Live scores matched offline predictions for every event.
- **Everything decided on validation.** Early stopping and alert thresholds come from validation; the test split is evaluated once per run and only reported.
- **Exactly-once publishing.** Each batch is one Kafka transaction: its scores, its alerts and the consumer offsets commit together or not at all. A killed scorer leaves nothing half-published and repeats nothing. It costs about 0.15 s at p95.
- **Canary-gated promotion.** New versions register as `challenger` and score every event alongside the champion. `make promote` moves the alias only inside tolerance; `make rollback` is one alias move. A running scorer hot-swaps models without a restart (measured: 16 s for a promote, 26 s for a rollback).

## Quickstart

Prerequisites: Docker (≥ 6 GB RAM allocated), [uv](https://docs.astral.sh/uv/), a free Kaggle account, and on macOS `brew install libomp` (needed by XGBoost).

```bash
make setup      # install dependencies; then copy .env.example → .env with your Kaggle token
make data       # IEEE-CIS train files → data/ (accept the competition rules on Kaggle once)
make up         # Redpanda, console :8085, Redis, Prometheus :9090, Grafana :3000
```

Then run each of these in its own terminal:

```bash
make mlflow                      # MLflow tracking + registry on :5000
make train                       # XGBoost → MLflow; first version becomes champion (~10 min)
make features                    # Spark Structured Streaming job (container)
make api                         # scorer on :8000
make dashboard                   # Streamlit on :8501
REPLAY_LIMIT=20000 make replay   # stream transactions
```

Checks: `make parity`, `make bench`, `make drift`, `make test`. Operations: `make promote` / `make rollback`, `make watch-drift` (3 consecutive drift flags → retrain a challenger; the canary gate still decides promotion), `make resilience-mark` / `make resilience-measure`. `make reset` gives a fresh stream; stop `make api` before running it.

## Limitations

- **Single node.** Spark runs in local mode with one Redis and no horizontal-scaling story, by design for a laptop project.
- **One scorer.** Exactly-once relies on a single transactional scorer, and holds for read-committed readers (the Kafka client default). After a kill, the replacement waits up to 45 s for the dead member's session to expire before it resumes.
- **Drift sensitivity.** Thresholds are calibrated per feature to stop false alarms, so a slow shift on a volatile feature must exceed its normal range before it fires.
- **Cold start.** The first window after a Spark restart is slow (p99 ≈ 4.4 s) while the state store recovers.

Two ideas were measured and cut: an autoencoder anomaly lane and a low-confidence "review" alert band. The numbers for both, and for everything above, are in [docs/MEASUREMENTS.md](./docs/MEASUREMENTS.md).

## Repository

Most of the logic lives in `src/streamguard/`. `tests/` holds 110 tests covering contracts, feature semantics, micro-batch-split invariance, scoring, hot swaps, the canary, drift and the dashboard's data shaping. CI runs lint and the suite on every push. The layout is described in [PLAN.md §14](./PLAN.md).

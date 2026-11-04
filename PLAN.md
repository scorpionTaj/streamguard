# StreamGuard — Project Plan

**Real-time fraud detection with an online feature store**
Kafka → Spark Structured Streaming → Redis → hybrid scoring → live Streamlit alerts.

> Status: ✅ **v0.1.0** — M0–M7 done. Results: §13. Write-up: [docs/WRITEUP.md](./docs/WRITEUP.md).
> Companion doc: [README.md](./README.md) (concept). This document is the single source of truth for scope, contracts, KPIs and milestones.

---

## 1. Goals & Non-Goals

**Goals**

- Portfolio-grade, locally-runnable streaming ML system (one laptop, Docker Compose).
- Real, defensible metrics: labeled data, temporal evaluation, measured latency.
- Full MLOps loop: registry, canary scoring, drift monitoring, benchmarked SLOs.

**Non-Goals**

- No multi-node cluster (Spark local mode inside one container is enough at 200 tx/s).
- No real payment integration, PII, or production security hardening.
- No horizontal scaling story (documented as future work instead).

---

## 2. Architecture

```
                     ┌──────────────────────────────────────────────────────────┐
                     │                        DOCKER COMPOSE                    │
                     │                                                          │
 IEEE-CIS csv ──► Replay ──► Redpanda ──► Spark SS features job ──► Redis 7 ──┐   │
 (590,540 tx)    producer   :9092       (1s micro-batch,          :6379      │   │
                 pace 200/s  3 partitions  watermark 10min)     card feature │   │
                             keyed by card_hash  hashes, TTL 7d  │           │   │
                                                                 ▼           │   │
                                                        ┌──────────────┐     │   │
                                                        │ FastAPI scorer│◄───┘   │
                                                        │ (host, uv)    │        │
                                                        │ MLflow models │        │
                                                        └──┬─────┬─────┘        │
                                          transactions.scored│     │alerts        │
                                                             ▼     ▼              │
                                                       Streamlit dashboard        │
                                                       (host, poll /alerts 2s)    │
                                                                          │       │
                                        Prometheus :9090 ◄── /metrics ────┘       │
                                        Grafana :3000                             │
                     └──────────────────────────────────────────────────────────┘
```

| # | Component | Tech | Runs where | Responsibility |
|---|-----------|------|-----------|----------------|
| 1 | Replay producer | Python, `confluent-kafka` | host | Stream IEEE-CIS rows in `TransactionDT` order at 200 tx/s with a synthetic wall clock (§7) |
| 2 | Broker | **Redpanda** v24 (Kafka API, no JVM/ZK) | container | Topics: `transactions.raw`, `transactions.scored`, `transactions.alerts` — 3 partitions, keyed by `card_key` hash |
| 3 | Feature job | **PySpark 4.0** Structured Streaming | container | Stateful windowed velocity aggregates per `card_key`; write to Redis; checkpointing |
| 4 | Online store | **Redis 7** | container | Hash per card: velocity features, TTL 7d; target read ≤ 5 ms |
| 5 | Scorer | **FastAPI** consumer loop | host | Consume raw → enrich from Redis → MLflow model → publish scored + alerts; `/metrics` for Prometheus |
| 6 | Dashboard | **Streamlit** | host | Alert feed (`st.fragment(run_every=2s)`), live counters, SHAP drill-down |
| 7 | Model registry | **MLflow** | host (`make mlflow`) | Train/run tracking, model registry, champion/challenger aliases |
| 8 | Observability | Prometheus + Grafana | containers | Lag, throughput, p95 latency, Redis hit-rate, drift PSI |
| 9 | Offline features / training | polars + XGBoost + PyTorch | host | Batch twin of online features; model training/eval |

**Why Redpanda:** single binary, no JVM/ZooKeeper, Kafka-API compatible (swap to `apache/kafka` KRaft by editing one image line), built-in console UI for demos.

---

## 3. Latency Budget (end-to-end KPI: p95 < 5 s @ 200 tx/s)

| Hop | Budget | Mechanism |
|---|---|---|
| Producer → broker | ≤ 50 ms | batched produce, `linger.ms=5` |
| Broker → Spark micro-batch | ≤ 1.2 s | `trigger="1 second"`, `maxOffsetsPerTrigger=400` |
| Spark agg → Redis write | ≤ 200 ms | `foreachBatch` pipeline with batched `hset` |
| Redis read (scorer) | ≤ 5 ms | single `HGETALL` per card |
| Model inference | ≤ 10 ms | XGBoost CPU, single row |
| Alert → dashboard | ≤ 100 ms | 2 s polling worst case — count separately |
| **Measured total (broker→alert published)** | **< 5 s p95** | `make bench` reports p50/p95/p99 per hop |

Micro-batching dominates the budget: typical end-to-end ≈ 1.5–3 s. SHAP explanation is **async** and never blocks the alert publish.

---

## 4. Data Contract (Kafka topics)

Defined in `src/streamguard/schemas.py` (pydantic v2, also frozen as JSON Schema exports).

**`transactions.raw`** — `Transaction`:
| field | type | notes |
|---|---|---|
| `transaction_id` | int | IEEE-CIS `TransactionID` |
| `event_ts` | datetime | synthetic wall clock derived from `TransactionDT` (§7) |
| `card_key` | str | `sha256(card1|card2|card5|addr1)[:16]` — stable pseudo- PAN |
| `transaction_amt` | float | |
| `product_cd` | str | W/C/R/H/S |
| `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2` | int/float \| null | anonymized dataset IDs (not PANs) — added at M1 |
| `card4`, `card6` | str \| null | network / type |
| `dist1`, `dist2` | float \| null | |
| `p_emaildomain`, `r_emaildomain` | str \| null | |
| `m1`…`m9` | bool \| null | match flags |
| `c1`…`c14` | float | count features |
| `d1`…`d15` | float \| null | timedelta features (days) — added at M1 |
| `v` | dict[str, float] | the 30 V-features frozen in `configs/features.yaml` (§6) |
| `device_type`, `device_info` | str \| null | identity table join (may be missing) |
| `is_fraud` | int | ground truth — replayed labeled, enables live precision measurement |

**`transactions.scored`** — `ScoredTx` = Transaction + `score` (float), `model_version` (str), `scored_at` (datetime).

**`transactions.alerts`** — `Alert` = ScoredTx + `risk_band` (`high|medium|review`), `top_drivers` (list of `{feature, impact}` from SHAP, filled async), `threshold` (float), `velocity_snapshot` (dict — the Redis features at scoring time).

Keying: `key = card_key` → per-card ordering within a partition (required for correct velocity state).

---

## 5. Dataset

| | |
|---|---|
| **Primary** | [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) (Vesta, real e-commerce transactions) |
| Size | `train_transaction.csv` 590,540 × 394 + `train_identity.csv` 144,233 × 33 (~1.2 GB) |
| Fraud rate | ~3.5 % (card-not-present) |
| Label | `isFraud` ∈ {0,1}; test split is unlabeled → use train only, own temporal split |
| Download | `make data` → `kaggle competitions download -c ieee-fraud-detection` (needs free Kaggle account + `~/.kaggle/kaggle.json`, see `.env.example`) |
| **Fallback** | [Sparkov synthetic](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (real timestamps 2019–2020, natural card numbers) — documented in `scripts/download_data.py --source sparkov` if Kaggle competition API is unavailable |

**Why IEEE-CIS:** real production data (stronger interview story than the synthetic Sparkov), published PR-AUC band to anchor our KPI, 3.5 % fraud rate gives meaningful PR curves. `TransactionDT` is a *timedelta* in seconds spanning ≈ 6 months — perfect for a compressed replay clock and a strict temporal split.

**Temporal split (frozen, no shuffling ever):**

| split | `TransactionDT` quantile | rows ≈ | purpose |
|---|---|---|---|
| train | 0 – 66.7 % | ~393 k | fit |
| val | 66.7 – 83.3 % | ~98 k | early stopping, threshold & ensemble-weight sweep |
| test | 83.3 – 100 % | ~98 k | reported metrics (one final evaluation, no peeking) |

---

## 6. Feature Spec (`configs/features.yaml` = single source of truth)

Both the Spark online job and the polars offline twin **read this file**; the parity test (§8) asserts they agree. Do not hardcode features in code.

```yaml
key: card_key                      # velocity grouping key
static:
  - transaction_amt
  - log_transaction_amt
  - amt_over_card_mean_24h         # derived after velocity pass
  - product_cd                     # one-hot: W C R H S
  - card4_onehot, card6_onehot
  - dist1, dist2, log_dist1
  - p_emaildomain_group            # mapped to ~12 groups (gmail, proton, corporate, …)
  - r_emaildomain_group
  - device_type_onehot
  - hour_sin, hour_cos, dow_sin, dow_cos   # cyclical, from event_ts
m_flags: [m1, m2, m3, m5, m6, m7, m8, m9]   # m4 is categorical in the data → excluded
c_features: [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]
raw_ids: [card1, card2, card3, card5, addr1, addr2]   # added at M1, passthrough
d_features: [d1, …, d15]                              # added at M1, passthrough
v_subset: [V187, V258, …]          # FROZEN AT M1: top-30 V-features by mutual information on train only
velocity:
  windows: [5m, 1h, 24h, 7d]
  metrics:
    - txn_count                    # count in window
    - amt_sum
    - amt_mean
    - time_since_last_txn          # seconds, ∞→ -1 sentinel (window-independent, one column)
    # uniq_merchants dropped at M1: IEEE-CIS has no merchant field
```

111 features total after one-hot (M1).

**Velocity semantics (M2, shared kernel `src/streamguard/velocity.py`):** for event *i* of card *k* at event time *t*, window *W* covers card *k*'s events *j ≤ i* (replay order = `(event_ts, transaction_id)`) with *t_j > t − W* — the snapshot **after** processing event *i*. Sums are exact (integer thousandths). The Spark job runs the same numpy kernel inside `applyInPandasWithState` (state = card events within 7d + last event, no timeout), and writes the card's latest snapshot to Redis `sg:card:<card_key>` = `{transaction_id, event_ts_s, <velocity features>}`. Static features are derived from the event itself by the scorer, not stored.

**Serving parity (M3):** Spark also writes a per-event snapshot `sg:tx:<transaction_id>` (TTL 600 s). The scorer waits for the event's own snapshot (≤ 10 s, else NaN velocity + `streamguard_feature_misses_total`) and builds the vector with the offline twin's code → live scores equal offline predictions (M3 run: 5 k events, max |Δscore| = 0, 0 misses). The snapshot wait is part of the measured end-to-end latency. Null policy: `v`/`dist`/`device` nulls → XGBoost native NaN handling; AE input uses train-median imputation + missing-mask channel.

---

## 7. Replay Clock & Pacing

- `TransactionDT` (seconds, relative) → `event_ts = REF_DATETIME + TransactionDT` with `REF_DATETIME = 2025-01-01T00:00:00Z`.
- **Time compression** `TIME_COMPRESSION=8800` (default): replaying all 590 k tx at 200 tx/s takes ~49 min real time ≈ 6 months of simulated time.
- Producer paces by `ΔTransactionDT / TIME_COMPRESSION`, capped at `REPLAY_RATE_TX_PER_SEC=200` (env-tunable; `--speed 0` = max rate for benchmark runs).
- Event time (not processing time) drives all Spark windows → window correctness is independent of compression factor.

---

## 8. Evaluation Protocol

- **Metrics** (`src/streamguard/eval/metrics.py`): AUPRC (primary), ROC-AUC (secondary), recall @ FPR 1 % / 5 %, alert precision@50, PR curve artifact logged to MLflow on every run.
- **Benchmark anchor (re-anchored at M1):** the published 0.78–0.88 XGBoost PR-AUC band ([EAI 2024](https://publications.eai.eu/index.php/ismla/article/view/12078), [Masethe et al. 2026](https://www.preprints.org/manuscript/202608.1727)) comes from *other* card-fraud datasets and does not transfer to IEEE-CIS under a strict temporal split. Measured on our split, same XGBoost params: **all 431 raw columns → val AUPRC 0.562** (the practical ceiling); original 90-feature contract → 0.496; + D1–D15 and raw card/addr IDs → 0.547. **Re-anchored again at M7 on the test split:** the same all-columns reference only reaches **test AUPRC 0.510** (the last 17 % of the timeline is harder for any model), so the M1 target ≥ 0.55 — derived from validation — is unreachable on test. Final KPI: **test AUPRC ≥ 0.48, i.e. ≥ 95 % of the all-columns test ceiling** (stretch: match it, 0.51).
- **Parity test** (`src/streamguard/eval/parity.py`): read every event on `transactions.raw`, recompute velocity offline, compare with the Redis hash of 1 k sampled cards (each hash = snapshot after the card's latest event); per-feature `max|Δ| < 1e-6`, else exit 1. Needs a single replay since `make reset`. M2 result: 60 k events at max rate, 152 micro-batches, max |Δ| = 0.
- **Latency bench** (`src/streamguard/eval/latency.py`): instrumented producer→alert run (`make bench`), spans per hop written to `artifacts/bench.json`, rendered into README table.
- **Drift** (`src/streamguard/eval/drift.py`): hourly PSI per feature vs training reference (10 bins, ε-smoothed); PSI > 0.2 → Grafana panel red + MLflow tag. Training-set reference histogram stored as a model artifact.

---

## 9. Model Spec

**Champion — XGBoost**
- `objective=binary:logistic`, `eval_metric=aucpr`, `scale_pos_weight ≈ 27` (1/0.035), `max_depth 8`, `eta 0.05`, `subsample 0.8`, `colsample_bytree 0.7`, early stopping 50 rounds on val AUPRC, 500–2000 trees.
- Native NaN handling for V/dist/device nulls.

**Challenger lane — Autoencoder (novel-pattern detector)**
- PyTorch, input 85 → 64 → 32 → **8** → 32 → 64 → 85 (sigmoid head), trained on **non-fraud train only**, 100 epochs Adam 1e-3, batch 512.
- Anomaly score = reconstruction MSE → converted to percentile against val-normal calibration.

**Ensemble:** `score = w · xgb_prob + (1-w) · ae_percentile`, `w` swept {0.5…1.0} on val for max AUPRC (expect w≈0.8–0.9; AE is the "unknown unknowns" lane, reported separately too).

**M5 result (run `b94ee73d…`, registered v3 = champion):** AE alone val AUPRC 0.134 / test 0.064 (base rate ≈ 0.037), so it carries some signal but every blend is worse than XGBoost alone: sweep 0.466 (w=0.5) → 0.543 (0.8) → 0.549 (0.9) → **0.550 (w=1.0, selected)**. The AE lane stays packaged, calibrated and reported, and is skipped at serving while w=1. **Temporal test (one evaluation, never used for selection):** AUPRC **0.489**, ROC-AUC 0.884, recall@FPR 5 % 0.590, precision@50 1.0. Reference: XGBoost on all 431 raw columns gets 0.561 val / **0.510 test**, so the test period is harder for any model and ours reaches 96 % of that ceiling.

**Packaging:** one MLflow pyfunc per version (booster + AE weights + preprocessing + calibration + w + feature order), loaded by the scorer via the alias. macOS note: torch and XGBoost ship separate libomp runtimes; `import_torch()` loads XGBoost first (else segfault) and pins torch to one thread (else deadlock).

**Threshold policy:** operating point = **max recall s.t. FPR ≤ 5 %** on val; report recall + alert precision@50 at that point. Risk bands: high ≥ `thr_high` (FPR ≤ 1 %), medium ≥ 5 %-thr, review 90th-percentile score. Thresholds are logged as val metrics on each training run, and the scorer reads them from the served version's run, so model and operating point move together.

**Explainability:** per-alert SHAP top-5 |impact| features via XGBoost's native TreeSHAP (`pred_contribs`, the TreeExplainer algorithm), on the XGBoost lane in log-odds, computed on a background worker after publish (~13 ms per alert batch) and attached to the API's alert record (`GET /alerts/{id}`, dashboard drill-down).

**Registry discipline:** every train run → MLflow (params, metrics, PR curve, feature spec hash, data hash). Promotion = move alias `champion` → challenger after M6 canary shows score-delta distribution within tolerance. Rollback = alias revert (one command, `src/streamguard/models/registry.py`).

---

## 10. MLOps Loop

1. **Canary:** scorer loads both `champion` and `challenger` aliases; every event scored twice; alerts use champion only; per-event score delta + agreement logged to Prometheus (`streamguard_canary_delta`) and sampled to MLflow.
2. **Drift:** §8 PSI job → Prometheus gauge + Grafana panel; PSI > 0.2 on any top-20-importance feature = retrain flag in README runbook.
3. **Retraining trigger (manual, documented):** drift flag or canary degradation → rerun `make train` with same code, new window → registry promotion flow.
4. **Grafana boards (provisioned):** consume lag (s), tx/s, alert rate, p50/p95/p99 latency, Redis hit-rate, PSI heatmap, canary delta.

**M6 implementation & results:**
- `make train` registers each new version as **challenger** (the first ever becomes champion). The scorer scores every event with both aliases (`GET /canary`, `streamguard_canary_*`); `make promote` moves the alias only if p95 |Δ| < 0.02 over ≥ 1 k events, recording the canary stats + `previous_champion` as version tags; `make rollback` is one alias move. Restart `make api` to serve a new champion.
- Canary run (12 k events, v3 champion vs v4 = seed-7 retrain with w = 0.95): p95 |Δ| **0.118**, mean Δ −0.034 (the 5 % AE lane shifts scores up), alert agreement 97.3 % → **promotion refused** by the gate (correct). Forced promote → rollback drill restored v3 in two commands.
- Drift: 10 k-row train sample logged per run (`drift_reference.npz`); scorer recomputes PSI (NaN = own bin) every 30 s over the last 20 k events; flag = any top-20-gain feature > 0.2. Live: top-20 max PSI 0.135 (`c10`) → no flag. Caveat seen live: `dow_*` PSI ≫ 0.2 in short windows (a 12 k-event window spans ≈ 1 simulated week, so weekday mix can't match training) — not a top-20 feature; real slow drift visible on `d10` (0.34).
- `make bench`: 12 k events continuing the stream @ 177 tx/s → end-to-end p50 **2.54 s**, p95 **4.17 s** (KPI < 5 s ✅), p99 6.85 s; Spark micro-batch + snapshot wait dominate, inference/publish ≈ 10 ms.

---

## 11. Stack & Environment (macOS Apple Silicon, 16 GB)

| layer | choice | notes |
|---|---|---|
| Python | 3.12, `uv` | `uv sync --all-extras` installs everything |
| Broker | Redpanda v24 + console | containers, ~1 GB |
| Streaming | PySpark 4.0 (`apache/spark:4.0.0` image) | container, mem-limit 2 GB; fallback `apache/spark:3.5.6` if 4.0 image issues |
| Online store | Redis 7.4-alpine, AOF on | container, < 200 MB |
| Offline ML | polars, XGBoost 2.x, SHAP ≥ 0.48 (numpy-2 safe), scikit-learn 1.5, PyTorch 2.x (CPU/MPS) | host |
| Serving/UI | FastAPI, uvicorn, Streamlit ≥ 1.38 (`st.fragment`) | host |
| Registry | MLflow 2.x | host, `make mlflow` → :5000 |
| Observability | Prometheus v2.54, Grafana 11 | containers |
| Total Docker RAM | ≈ 3.5–4 GB | Docker Desktop allocation ≥ 6 GB recommended |

Heavy deps are **optional extras** (`[train]`, `[stream]`) so the default `uv sync` stays light; `make setup` installs all.

---

## 12. Milestones (day-level, ≈ 4 weeks part-time)

| # | Name | Days | Deliverable — **definition of done** |
|---|------|------|--------------------------------------|
| M0 | Scaffold & smoke | 1 | `make up` green; `make replay` streams 1 k tx into Redpanda; console shows messages |
| M1 | Data & offline baseline | 3 | `make data` + `notebooks/01_eda.py` done; `v_subset` frozen into `features.yaml`; `make train` → XGBoost-only val AUPRC logged (≥ 0.50; originally 0.75, re-anchored — see §8) |
| M2 | Streaming features | 4 | features job consumes raw topic, Redis hashes populate; `make parity` green (δ < 1e-6) |
| M3 | Scorer & registry | 3 | FastAPI consumer scores end-to-end; alerts topic populated; model pulled from MLflow alias; `/metrics` scraped |
| M4 | Dashboard | 2 | Streamlit alert feed + counters + velocity snapshot drill-down; demo-able |
| M5 | Hybrid & explanations | 3 | AE trained, ensemble `w` swept, SHAP top-5 on alerts, threshold policy implemented + tested |
| M6 | MLOps hardening | 3 | canary scoring live, PSI drift job + Grafana, `make bench` report filled (KPI table §13) |
| M7 | Portfolio polish | 2 | README results table, architecture diagram image, demo GIF, 1-page write-up; final commit tag `v0.1.0` |

**Rule:** one milestone fully done before the next starts; every milestone ends with a commit + green tests.

---

## 13. KPI Table (each KPI ↔ the artifact that proves it)

| KPI | Target | Result (v0.1.0) | Measured by |
|---|---|---|---|
| AUPRC (temporal test) | ≥ 0.48 (≥ 95 % of all-columns ceiling 0.510) | **0.489** ✅ (96 %) · val 0.550 | `make train` → MLflow run `b94ee73d…` (v3) |
| p95 end-to-end latency @ ~200 tx/s | < 5 s | **4.17 s** ✅ (p50 2.54, p99 6.85 @ 177 tx/s) | `make bench` → `artifacts/bench.json` |
| Online/offline feature parity | max Δ < 1e-6 | **0** ✅ (1 k cards × 13 features, 60 k events) | `make parity` |
| Live vs offline score parity | (added M3) | **0** ✅ (5 k events) | M3 check |
| Recall @ FPR 5 % | report | **0.590** test · 0.653 val | `make train` |
| Alert precision@50 | report | **1.0** test · 1.0 val | `make train` |
| Drift (PSI, top-20 feature) | < 0.2 steady-state | **0.135** max ✅ (no retrain flag) | `make drift`, Grafana |
| Canary agreement | score-delta p95 < 0.02 to promote | v4 challenger: **0.118** → promotion refused (gate works) | `GET /canary`, Grafana |

---

## 14. Repo Layout

```
streamguard/
├── PLAN.md, README.md
├── pyproject.toml            # uv; extras: [train], [stream]
├── Makefile                  # setup data up down replay train parity bench drift api dashboard mlflow test lint demo
├── docker-compose.yml        # redpanda, console, redis, spark, prometheus, grafana
├── .env.example              # kaggle keys, endpoints, replay pacing
├── configs/
│   ├── pipeline.yaml         # topics, partitions, windows, watermark, trigger, pacing
│   ├── features.yaml         # §6 feature contract (SSOT)
│   └── prometheus.yml
├── src/streamguard/
│   ├── schemas.py            # §4 pydantic contracts  (implemented)
│   ├── config.py             # yaml loading + window parsing (implemented)
│   ├── producer/replay.py    # replay clock + pacing helpers (implemented) / loop (M0)
│   ├── streaming/features_job.py   # Spark SS job (M2)
│   ├── offline/features.py   # polars offline twin (M1–M2)
│   ├── models/               # train.py, autoencoder.py, explainer.py, registry.py (M1/M3/M5)
│   ├── serving/              # main.py (API), consumer.py (scorer loop), canary.py (M3/M6)
│   └── eval/                 # metrics.py (implemented), parity.py, latency.py, drift.py (M2/M6)
├── dashboard/app.py          # Streamlit (M4)
├── notebooks/01_eda.py       # EDA + v_subset selection (M1)
├── tests/                    # contract tests, green from day one
└── data/                     # gitignored; kaggle download target
```

---

## 15. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Kaggle competition download requires account/accept-rules | `.env.example` documents setup; `--source sparkov` fallback in `scripts/download_data.py` |
| Spark 4.0 image / arm64 quirks | fallback `apache/spark:3.5.6` line in compose; PySpark only needed in container, never on host |
| Laptop RAM pressure | container mem-limits; MLflow/API/dashboard are host processes, not containers |
| Per-card ordering broken → wrong velocity | producer keys by `card_key`; parity test would catch drift |
| V-feature selection overfit | select on **train split only**; freeze into `features.yaml` (hash logged with model) |
| Scope creep (grafana dashboards, AE tuning) | timeboxed milestones; M7 polish is non-negotiable — portfolio quality over feature count |

---

## 16. References

- [IEEE-CIS Fraud Detection — Kaggle data page](https://www.kaggle.com/c/ieee-fraud-detection/data) (schema, `TransactionDT` semantics)
- [IEEE-CIS top-5 % solution walkthrough](https://towardsdatascience.com/ieee-cis-fraud-detection-top-5-solution-5488fc66e95f)
- [Sparkov / Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (fallback)
- [EAI 2024 — cost-sensitive XGBoost, PR-AUC 0.78→0.88](https://publications.eai.eu/index.php/ismla/article/view/12078)
- [Masethe et al. 2026 — explainable XGBoost, PR-AUC 0.823](https://www.preprints.org/manuscript/202608.1727)
- [Redpanda quickstart (Docker, Kafka-API)](https://docs.redpanda.com/streaming/current/get-started/quick-start)
- [Local dev with Redpanda in Docker Compose](https://www.tinybird.co/blog/custom-ci-for-kafka-and-tinybird)
- [Databricks — real-time fraud detection with Spark](https://www.databricks.com/blog/how-build-real-time-fraud-detection-using-spark-real-time-mode-and-lakebase)
- [Redis — real-time fraud detection: latency, features & scale](https://redis.io/blog/real-time-fraud-detection)
- [AWS reference architecture — real-time fraud detection powered by Redis](https://docs.aws.amazon.com/reference-architecture-diagrams/latest/realtime-fraud-detection-redis/realtime-fraud-detection-redis.html)
- [Spark Structured Streaming programming guide](https://spark.apache.org/docs/latest/streaming/index.html)

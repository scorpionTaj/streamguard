# StreamGuard

**Real-time fraud detection with an online feature store — Kafka → Spark Structured Streaming → Redis → hybrid scoring → live alert dashboard.**

> Status: ✅ **v0.1.0** — runs end to end on one laptop (Docker Compose + host processes). Full spec and milestone log in [PLAN.md](./PLAN.md); one-page write-up in [docs/WRITEUP.md](./docs/WRITEUP.md).

![StreamGuard demo: live alert feed, SHAP drill-down, Grafana](docs/demo.gif)

## Results

All numbers come from the commands in the last column. The data is IEEE-CIS (590,540 real, labeled e-commerce transactions), evaluated on a strict temporal split with no shuffling.

| KPI                                | Target                                       | Result                                                      | Proof                                      |
| ---------------------------------- | -------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------ |
| **AUPRC, temporal test**     | ≥ 0.48 (≥ 95 % of the all-columns ceiling) | **0.489** ✅ (val 0.550)                              | `make train` → MLflow                   |
| p95 end-to-end latency @ ~200 tx/s | < 5 s                                        | **4.17 s** ✅ (p50 2.54 s, p99 6.85 s)                | `make bench` → `artifacts/bench.json` |
| Online/offline feature parity      | max \|Δ\| < 1e-6                            | **0** ✅ (1 k cards × 13 features, 60 k events)      | `make parity`                            |
| Live vs offline score parity       | —                                           | **0** ✅ (5 k events)                                 | M3 check                                   |
| Recall @ FPR 5 % · precision@50   | report                                       | 0.590 · 1.00 (test)                                        | `make train`                             |
| Canary gate                        | promote only if p95 \|Δscore\| < 0.02       | challenger at 0.118 → **refused** ✅                 | `make promote`                           |
| Drift monitor (calibrated PSI)     | no false alarms in-distribution; catches real shifts | **0/46** train/val windows flagged ✅ (fixed 0.2 rule: 18 % of train windows); flagged 2/8 test windows on a real V218 regime change; injected shifts caught 6–8/8 | `make drift`, Grafana                    |

**About the AUPRC target.** Published XGBoost PR-AUC figures (0.78–0.88) come from other card-fraud datasets. On IEEE-CIS with a temporal split, XGBoost using **all 431 raw columns** reaches only 0.561 on validation and **0.510 on test**, because the last 17 % of the timeline is harder for any model. StreamGuard's 111 streaming-safe features reach 96 % of that test ceiling. The target was re-anchored to the measured ceiling, and every step is recorded in [PLAN.md §8](./PLAN.md#8-evaluation-protocol).

**Latency per hop** (`make bench`, 12 k events @ 177 tx/s):

| hop                          | p50 (s)        | p95 (s)        | p99 (s)        |
| ---------------------------- | -------------- | -------------- | -------------- |
| broker → Spark → Redis     | 1.41           | 2.90           | 5.55           |
| snapshot wait (scorer)       | 1.44           | 1.98           | 2.52           |
| inference (features + model) | 0.01           | 0.01           | 0.01           |
| publish                      | 0.01           | 0.01           | 0.01           |
| **end-to-end**         | **2.54** | **4.17** | **6.85** |

Spark's 1-second micro-batch accounts for most of the latency, while the model itself is negligible.

## Architecture

![architecture](docs/architecture.svg)

1. **Replay producer:** streams IEEE-CIS rows in `TransactionDT` order on a compressed synthetic clock (≈ 6 months in ≈ 49 min, capped at 200 tx/s), keyed by a hashed `card_key` so per-card order holds within each partition.
2. **Spark Structured Streaming:** keeps exact trailing-window velocity (5 m / 1 h / 24 h / 7 d counts, sums and means, plus time since the previous transaction) per card with `applyInPandasWithState`. It writes the latest snapshot per card (`sg:card:*`) and a per-event snapshot (`sg:tx:*`) to Redis.
3. **FastAPI scorer:** consumes the same topic and **waits for each event's own snapshot**. It builds the feature vector with the *offline* twin's code, scores with the MLflow `champion` (and the `challenger`, as a canary), and publishes scored transactions and banded alerts. SHAP top-5 drivers are attached asynchronously.
4. **Streamlit dashboard:** two tabs. **Live alerts** has the filtered feed (band, ground truth, minimum score, card/transaction search) and a per-alert drill-down with the velocity snapshot and SHAP drivers. **Model health** shows champion/challenger versions with the canary verdict, calibrated drift ratios per feature, and alert precision by band. Labels are replayed with the stream, so precision is measured live.
5. **Prometheus + Grafana:** throughput, alerts by band, latency p50/p95/p99, consumer lag, snapshot hit-rate, PSI ÷ calibrated threshold per feature, retrain flag, canary delta.

## Design decisions worth a look

- **One contract, one velocity function.** The producer, the offline twin and the Spark job all build from the `Transaction` schema and `configs/features.yaml`. The velocity math is a single numpy function ([velocity.py](src/streamguard/velocity.py)) that Spark runs inside its state function, so the parity check tests the *streaming plumbing* (ordering, state, Redis) rather than two copies of the formulas. Sums run in integer thousandths of a currency unit, so parity is exactly 0.
- **Serving equals training, exactly.** Training features include the current event. Redis only holds a card's *latest* snapshot, which could be from before this event or after it. So Spark also writes a short-lived per-event snapshot, and the scorer waits for it. Live scores matched the offline predictions for every event.
- **Everything decided on validation.** Early stopping, autoencoder calibration, the ensemble weight and the band thresholds all come from validation. The test split is evaluated once per run and only reported.
- **A model version is self-contained.** Each MLflow pyfunc packages the booster, the autoencoder, its preprocessing and calibration, the weight `w` and the feature order. Thresholds come from the version's own run, so a model and its operating point move together.
- **Canary-gated promotion.** New versions register as `challenger`, and the scorer scores every event twice. `make promote` moves the alias only when the 95th-percentile score delta is below 0.02. `make rollback` is a single alias move. A running scorer polls the aliases every 30 s and hot-swaps models between batches, with no restart (measured: 16 s for a promote, 26 s for a rollback).

## Quickstart

Prerequisites: Docker (≥ 6 GB RAM allocated), [uv](https://docs.astral.sh/uv/), a free Kaggle account, and on macOS `brew install libomp` (needed by XGBoost).

```bash
make setup                      # uv sync --all-extras; then copy .env.example → .env with your Kaggle token
```

```bash
make data                       # IEEE-CIS train files → data/ (accept the competition rules on Kaggle once)
```

```bash
make up                         # Redpanda, console :8085, Redis, Prometheus :9090, Grafana :3000
```

Then run each of these in its own terminal:

```bash
make mlflow                     # MLflow tracking + registry on :5000
```

```bash
make train                      # hybrid model → MLflow; first version becomes champion (~10 min)
```

```bash
make features                   # Spark Structured Streaming job (container)
```

```bash
make api                        # scorer on :8000
```

```bash
make dashboard                  # Streamlit on :8501
```

```bash
REPLAY_LIMIT=20000 make replay  # stream transactions
```

Checks and operations: `make parity`, `make bench`, `make drift`, `make promote` / `make rollback`, `make test`. `make reset` gives a fresh stream; stop `make api` before running it.

## Limitations & next steps

- **Drift thresholds are calibrated per feature, which costs some sensitivity.** A fixed PSI > 0.2 rule false-alarmed on 18 % of 20 k-event windows *inside* the training period, and it fired live during the demo. Several top features (V201, V257, V187, c10, …) routinely swing to PSI ≈ 0.47 from one week to the next. Each feature's threshold is now the worst PSI any training window of the live window's size shows (with 0.2 as a floor). In-distribution windows no longer flag, but a slow shift on those volatile features has to exceed their normal range before it fires.
- **Two PSI blind spots were found and fixed, at a measured cost.** (1) Quantile bins collapsed for features whose present values are dominated by one value (93 % of V187 = 1.0 gave a single bin), so discrete features (≤ 50 distinct values) now get one bin per value. (2) For mostly-missing features the missing bin swamps value changes, so a second, present-values-only PSI with its own calibrated threshold runs alongside. Moving 20 % of V187's present values from 1 to 2 went from 0/8 to 8/8 test windows detected. On the un-injected temporal test split, 2/8 windows now flag on V218: its present values jump from a mean of ~1.4 to ~17 in early-to-mid June, and model AUPRC in those windows is 0.45 against 0.51–0.54 in clean windows. That looks like real drift, but windows 2 and 5 score similarly low without a flag, so the link is suggestive, not proven.
- **The autoencoder lane adds nothing yet.** Its own val AUPRC is 0.134 (base rate 0.037), and every blend scores below XGBoost alone, so the sweep picks `w = 1.0`. It stays packaged and reported as the "unknown unknowns" lane.
- **Tail latency:** p99 is 6.85 s, driven by the micro-batch timing. Spark's real-time trigger or a smaller trigger interval would shorten it.
- **Review band:** on a 336-alert live sample, precision was 72 % for high, 13 % for medium and **0 %** for review (82 alerts). If that holds over longer runs, the review floor (the 90th-percentile score) generates mostly noise and should be raised or dropped.
- **Single node only:** Spark runs in local mode with one Redis and no horizontal-scaling story, which is deliberate for a laptop project (see PLAN.md non-goals).
- **macOS:** torch and XGBoost ship separate OpenMP runtimes. The code loads XGBoost first and pins torch to one thread; the alternatives were a segfault and a deadlock, both reproduced.

## Repository

The spec, contracts, milestone log and KPI table are in [PLAN.md](./PLAN.md) (§14 covers the layout). Most of the logic lives in `src/streamguard/`, and `tests/` holds 94 tests covering contracts, feature semantics, micro-batch-split invariance, scoring, hot swaps, the canary, drift and the dashboard's data shaping. CI (GitHub Actions) runs lint and the full suite on every push.

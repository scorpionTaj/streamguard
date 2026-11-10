# StreamGuard — measurements in detail

The README gives the headline numbers. This page holds the detail behind them: how each was
measured, what was changed after v0.1.0, and what was tried and cut. The spec and milestone
log are in [PLAN.md](../PLAN.md); the narrative is in [WRITEUP.md](./WRITEUP.md).

## Latency

Per hop (`make bench`, 12 k events @ 181 tx/s, steady state, exactly-once publishing):

| hop                          | p50 (s)  | p95 (s)  | p99 (s)  |
| ---------------------------- | -------- | -------- | -------- |
| broker → Spark → Redis       | 0.63     | 0.86     | 0.91     |
| snapshot wait (scorer)       | 0.47     | 0.57     | 0.79     |
| inference (features + model) | 0.01     | 0.01     | 0.01     |
| publish (transaction commit) | 0.01     | 0.01     | 0.02     |
| **end-to-end**               | **1.01** | **1.47** | **1.57** |

`published` is now the transaction commit, the moment a reader can first see the batch.
Before exactly-once publishing the same bench read 0.94 / 1.32 / 1.56 s, with `published`
taken when each message was handed to the producer: the commit itself costs about 10 ms
per batch, and the end-to-end figure rises by about 0.15 s at p95.

The first bench after a cold (re)start is slower while Spark recovers its state store from the
checkpoint (p99 ≈ 4.4 s); the running service then holds the table above.

*(post-v0.1.0)* Three changes cut end-to-end p99 from 6.85 s to 1.56 s: the micro-batch trigger
1 s → 500 ms, Spark `local[2]` → `local[4]` (3 Kafka partitions no longer run in two task
waves), and the scorer polls Kafka in ≤ 100-message batches so it never swallows the stream
while waiting for snapshots. The model itself is negligible (~10 ms). The tail is now bounded
by Spark's 500 ms micro-batch plus per-batch scheduling.

### Load runs above the design rate

With exactly-once publishing, two runs reached 393 and 506 tx/s on a busier machine:
p95 2.18 s and 1.89 s (p99 2.51 s and 2.05 s), no dropped events. The higher-rate run
below was measured before that change and has not been repeated.

*(post-v0.1.0)* Replayed at ~774 tx/s (the synthetic-clock pacing compresses to the 1 ms
scheduler floor; the ceiling is Python's sleep granularity, not the pipeline). Steady state
over two 12 k-event runs: **p50 2.13 s, p95 2.98 s, p99 3.20 s** — the same < 5 s p95 KPI
holds at 3.9× the rate, with no dropped events. Two capacity knobs were raised for the load
rate and are inert at 200 tx/s (`max_offsets_per_trigger` 400 → 2000, scorer poll cap
100 → 500); the producer now pre-serializes payloads off the pacing path, and the bench reads
its true produce rate from Kafka CreateTime spans.

## Drift monitor

- **Thresholds are calibrated per feature, which costs some sensitivity.** A fixed PSI > 0.2
  rule false-alarmed on 18 % of 20 k-event windows *inside* the training period, and it fired
  live during the demo. Several top features (V201, V257, V187, c10, …) routinely swing to
  PSI ≈ 0.47 from one week to the next. Each feature's threshold is now the worst PSI any
  training window of the live window's size shows (with 0.2 as a floor). In-distribution
  windows no longer flag, but a slow shift on those volatile features has to exceed their
  normal range before it fires.
- **Two PSI blind spots were found and fixed, at a measured cost.** (1) Quantile bins
  collapsed for features whose present values are dominated by one value (93 % of V187 = 1.0
  gave a single bin), so discrete features (≤ 50 distinct values) now get one bin per value.
  (2) For mostly-missing features the missing bin swamps value changes, so a second,
  present-values-only PSI with its own calibrated threshold runs alongside. Moving 20 % of
  V187's present values from 1 to 2 went from 0/8 to 8/8 test windows detected. On the
  un-injected temporal test split, 2/8 windows now flag on V218: its present values jump from
  a mean of ~1.4 to ~17 in early-to-mid June, and model AUPRC in those windows is 0.45 against
  0.51–0.54 in clean windows. That looks like real drift, but windows 2 and 5 score similarly
  low without a flag, so the link is suggestive, not proven.

## Measured and cut

- **The autoencoder lane** *(post-v0.1.0)*. Its own val AUPRC was 0.134 (base rate 0.037) on
  all 111 features and **0.064** on the 13 velocity features alone — the "behavioral anomaly"
  hypothesis doesn't hold on this dataset — and every blend scored below XGBoost alone, so the
  sweep always picked `w = 1.0`. The lane's code and the torch dependency have been removed;
  model versions registered before the cut (≤ v4) need pre-0.1.1 code to load.
- **The review band** *(post-v0.1.0)*. On a 336-alert live sample, precision was 72 % for
  high, 13 % for medium and **0 %** for review (82 alerts). Offline scoring explains why: the
  review floor (the 90th-percentile validation score) sits at the FPR ≈ 7.5 % operating point,
  and the region between it and the 5 %-FPR medium floor holds 2.6 % of all traffic at 8–9 %
  precision — noise both live and offline. Alerts now start at the medium floor, and every
  band is an FPR-bounded validation operating point.

## Resilience drills

*(post-v0.1.0)* Three kill-mid-stream drills (~200 tx/s, 12 k events each, measured with
[scripts/measure_resilience.py](../scripts/measure_resilience.py)):

- **Scorer `kill -9`:** each batch is one Kafka transaction (scores, alerts and consumer
  offsets together), so the restarted scorer resumes from the last committed batch.
  Measured twice on 12 k-event replays: one kill, and three kills ten seconds apart —
  **12 000/12 000 scored, 0 missing, 0 duplicates** both times. Before this the scorer
  committed offsets on a 5 s timer and the same drill left 900 duplicates. After a kill
  the replacement waits for the dead member's session to expire (up to 45 s) before it
  is assigned partitions.
- **Redis restart** *(measured before exactly-once publishing; not repeated)*: *(both components originally crashed — fixed)* the Spark writer now
  retries with exponential backoff and the scorer re-polls the unpublished batch (snapshots
  are fetched before any publish, so the batch is never half-done). After the fix both
  survive: **0 missing**, 700 events published in degraded mode (no velocity snapshot) during
  the outage, Spark stalled then caught up.
- **Spark kill** *(measured before exactly-once publishing; not repeated)*: checkpoint resume replays the gap and re-writes the missing snapshots —
  **0 duplicates, 0 missing**, self-healed.

Events are never lost; the cost of a Redis outage is briefly degraded features, visible on
the dashboard and in Grafana.

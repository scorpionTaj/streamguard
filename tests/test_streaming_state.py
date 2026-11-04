"""Streaming state transition tests (streaming/features_job.py, PLAN.md §6).

The Spark job's per-card logic is a pure function; these tests prove that feeding
events in arbitrary micro-batch splits gives exactly the single-pass offline result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamguard.config import load_feature_spec, velocity_window_seconds
from streamguard.streaming.features_job import (
    latest_per_card,
    redis_mapping,
    update_card,
    velocity_columns,
)
from streamguard.velocity import velocity_features

SPEC = load_feature_spec()


def _events(ts, amts, start_id=1):
    return pd.DataFrame({"event_ts_s": np.asarray(ts, float), "amt": amts,
                         "transaction_id": np.arange(start_id, start_id + len(ts))})  # fmt: skip


def _offline(ts, amts):
    cfg = SPEC["velocity"]
    return velocity_features(np.full(len(ts), "c"), np.asarray(ts, float), np.asarray(amts),
                             velocity_window_seconds(SPEC), cfg["metrics"],
                             cfg["sentinel_no_history"])  # fmt: skip


class TestUpdateCard:
    @pytest.mark.parametrize("splits", [[60], [1, 59], [7, 13, 40], [1] * 60])
    def test_micro_batch_split_matches_single_pass(self, splits):
        rng = np.random.default_rng(1)
        # gaps up to 3 days so the 7d window both fills and prunes
        ts = np.cumsum(rng.integers(0, 3 * 86400, 60)).astype(float) + 1.7e9
        amts = np.round(rng.uniform(0.5, 900, 60), 3)
        events = _events(ts, amts)

        state, rows, start = None, [], 0
        for size in splits:
            out, state = update_card("c", state, events.iloc[start : start + size], SPEC)
            rows.append(out)
            start += size
        online = pd.concat(rows, ignore_index=True)

        offline = _offline(ts, amts)
        for name in velocity_columns(SPEC):
            np.testing.assert_allclose(online[name].to_numpy(), offline[name], atol=1e-9)
        assert online["transaction_id"].tolist() == list(range(1, 61))

    def test_state_pruned_to_longest_window_keeps_last(self):
        events = _events([0, 10, 10 * 86400], [1.0, 2.0, 3.0])
        _, (ts, amt, ids) = update_card("c", None, events, SPEC)
        assert ids == [3] and ts == [10 * 86400.0] and amt == [3.0]

    def test_unordered_batch_sorted_by_time_then_id(self):
        events = _events([50, 50, 10], [1.0, 2.0, 3.0]).iloc[[1, 2, 0]]
        out, _ = update_card("c", None, events, SPEC)
        assert out["transaction_id"].tolist() == [3, 1, 2]
        assert out["txn_count_5m"].tolist() == [1.0, 2.0, 3.0]


class TestRedisWrite:
    def test_latest_event_per_card_wins(self):
        batch = pd.DataFrame({"card_key": ["a", "b", "a"], "event_ts_s": [5.0, 1.0, 5.0],
                              "transaction_id": [9, 2, 10]})  # fmt: skip
        latest = latest_per_card(batch).set_index("card_key")["transaction_id"]
        assert latest.to_dict() == {"a": 10, "b": 2}

    def test_mapping_round_trips_floats_exactly(self):
        row = pd.Series({"transaction_id": 7, "event_ts_s": 1.7e9, "amt_mean_1h": 1 / 3})
        mapping = redis_mapping(row, ["amt_mean_1h"])
        assert float(mapping["amt_mean_1h"]) == 1 / 3
        assert mapping["transaction_id"] == "7"

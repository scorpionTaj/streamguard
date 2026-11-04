"""Offline feature twin tests (offline/features.py, PLAN.md §5, §6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from streamguard.offline.features import (
    build_features,
    discrete_mutual_information,
    feature_columns,
    select_v_subset,
    temporal_split,
    transactions_frame,
    velocity_features,
)
from streamguard.schemas import Transaction

WINDOWS = {"5m": 300, "1h": 3600}
METRICS = ["txn_count", "amt_sum", "amt_mean", "time_since_last_txn"]


def velocity(cards, ts, amounts):
    return velocity_features(
        np.array(cards), np.array(ts, dtype=float), np.array(amounts), WINDOWS, METRICS, -1.0
    )


class TestVelocity:
    def test_trailing_window_includes_current_event(self):
        out = velocity(["a", "a", "a"], [0, 100, 400], [10.0, 20.0, 30.0])
        # 5m window at t=400 covers (100, 400] → excludes t=100 (boundary is open on the left)
        assert out["txn_count_5m"].tolist() == [1, 2, 1]
        assert out["amt_sum_5m"].tolist() == [10.0, 30.0, 30.0]
        assert out["txn_count_1h"].tolist() == [1, 2, 3]
        assert out["amt_mean_1h"].tolist() == pytest.approx([10.0, 15.0, 20.0])

    def test_cards_are_independent_and_order_restored(self):
        out = velocity(["a", "b", "a", "b"], [0, 10, 20, 30], [1.0, 2.0, 3.0, 4.0])
        assert out["txn_count_1h"].tolist() == [1, 1, 2, 2]
        assert out["amt_sum_1h"].tolist() == [1.0, 2.0, 4.0, 6.0]
        assert out["time_since_last_txn"].tolist() == [-1.0, -1.0, 20.0, 20.0]

    def test_same_timestamp_counts_in_replay_order(self):
        # streaming sees ties one at a time → the first event must not count the second
        out = velocity(["a", "a"], [50, 50], [5.0, 7.0])
        assert out["txn_count_5m"].tolist() == [1, 2]
        assert out["time_since_last_txn"].tolist() == [-1.0, 0.0]

    def test_sums_exact_at_three_decimals(self):
        amounts = [57.084, 0.001, 31937.391] * 1000
        out = velocity(["a"] * 3000, list(range(3000)), amounts)
        assert out["amt_sum_1h"][-1] == pytest.approx(sum(amounts), abs=1e-9)

    def test_unknown_metric_rejected(self):
        with pytest.raises(ValueError):
            velocity_features(np.array(["a"]), np.array([0.0]), np.array([1.0]), WINDOWS,
                              ["uniq_merchants"], -1.0)  # fmt: skip


def _tx(i: int, card: str, seconds: int, amt: float, **kw) -> Transaction:
    return Transaction(
        transaction_id=i, event_ts=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds),
        card_key=card * 16, transaction_amt=amt, product_cd="W", is_fraud=0, **kw,
    )  # fmt: skip


class TestBuildFeatures:
    def test_columns_follow_repo_spec(self):
        frame = transactions_frame(
            [_tx(1, "a", 0, 10.0, card4="visa", p_emaildomain="gmail.com", v={"V187": 1.0}),
             _tx(2, "a", 60, 30.0, m1=True, device_type="mobile")],
            ["V187"],
        )  # fmt: skip
        spec = {
            "key": "card_key",
            "static": [
                "transaction_amt",
                "amt_over_card_mean_24h",
                "card4_onehot",
                "p_emaildomain_group",
            ],
            "m_flags": ["m1"],
            "c_features": ["c1"],
            "v_subset": ["V187"],
            "velocity": {"windows": ["24h"], "metrics": ["txn_count", "amt_mean"]},
        }
        out = build_features(frame, spec)
        assert feature_columns(out) == [
            "transaction_amt", "amt_over_card_mean_24h", "card4_visa", "card4_mastercard",
            "card4_american_express", "card4_discover", "p_emaildomain_group", "m1", "c1", "V187",
            "txn_count_24h", "amt_mean_24h",
        ]  # fmt: skip
        assert out["amt_over_card_mean_24h"].to_list() == pytest.approx([1.0, 1.5])
        assert out["card4_visa"].to_list() == [1.0, 0.0]
        assert out["m1"].to_list() == [None, 1.0]
        assert out["V187"].to_list() == [1.0, None]

    def test_unknown_static_feature_fails_loudly(self):
        frame = transactions_frame([_tx(1, "a", 0, 1.0)], [])
        spec = {"key": "card_key", "static": ["no_such_feature"], "m_flags": [], "c_features": [],
                "velocity": {"windows": ["1h"], "metrics": ["txn_count"]}}  # fmt: skip
        with pytest.raises(ValueError, match="no_such_feature"):
            build_features(frame, spec)


class TestSplitAndSelection:
    def test_temporal_split_is_ordered_and_complete(self):
        frame = pl.DataFrame({"t": np.arange(600, dtype=float)})
        parts = temporal_split(frame, "t")
        assert sum(p.height for p in parts.values()) == 600
        assert parts["train"]["t"].max() < parts["val"]["t"].min()
        assert parts["val"]["t"].max() < parts["test"]["t"].min()
        assert parts["train"].height == pytest.approx(400, abs=2)

    def test_mi_ranks_informative_feature_first(self):
        rng = np.random.default_rng(0)
        y = (rng.random(5000) < 0.1).astype(int)
        informative = y + rng.normal(0, 0.3, 5000)
        noise = rng.normal(0, 1, 5000)
        mostly_null = np.where(rng.random(5000) < 0.9, np.nan, noise)
        frame = pl.DataFrame({"Vnoise": noise, "Vgood": informative, "Vnull": mostly_null})
        assert select_v_subset(frame, y, k=2)[0] == "Vgood"
        assert discrete_mutual_information(noise, y) < 0.01

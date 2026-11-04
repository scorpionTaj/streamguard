"""Replay clock & pacing tests (PLAN.md §7)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from streamguard.producer.replay import (
    DEFAULT_MAX_RATE_TX_PER_SEC,
    DEFAULT_TIME_COMPRESSION,
    iter_transactions,
    load_replay_frame,
    pace_delay,
    to_event_ts,
)
from streamguard.schemas import derive_card_key


class TestEventClock:
    def test_maps_relative_to_wall_clock(self):
        assert to_event_ts(0) == datetime(2025, 1, 1, tzinfo=UTC)
        assert to_event_ts(3600) == datetime(2025, 1, 1, 1, 0, tzinfo=UTC)

    def test_monotonic(self):
        earlier = to_event_ts(100.0)
        later = to_event_ts(250.0)
        assert later > earlier and later.tzinfo is UTC


class TestPacing:
    def test_rate_cap_wins_for_dense_bursts(self):
        # simulated gap of 14 s at compression 8800 → 1.6 ms, but the rate cap
        # forces ≥ 1/200 s per event.
        delay = pace_delay(14.0, DEFAULT_TIME_COMPRESSION, DEFAULT_MAX_RATE_TX_PER_SEC)
        assert delay == pytest.approx(1.0 / 200.0)

    def test_sparse_gap_respects_compression(self):
        # 8800 s simulated gap → exactly 1 s real time
        assert pace_delay(8800.0, 8800.0, 200.0) == pytest.approx(1.0)

    def test_rejects_out_of_order_events(self):
        with pytest.raises(ValueError):
            pace_delay(-1.0)


class TestReplayFrame:
    """CSV → Transaction mapping on a tiny IEEE-CIS-shaped fixture."""

    @pytest.fixture
    def csvs(self, tmp_path):
        tx = tmp_path / "train_transaction.csv"
        tx.write_text(
            "TransactionID,isFraud,TransactionDT,TransactionAmt,ProductCD,card1,card2,card3,card4,"
            "card5,card6,addr1,addr2,dist1,dist2,P_emaildomain,R_emaildomain,"
            + ",".join(f"C{i}" for i in range(1, 15))
            + ","
            + ",".join(f"M{i}" for i in range(1, 10))
            + ","
            + ",".join(f"D{i}" for i in range(1, 16))
            + ",V1,V2\n"
            # deliberately out of TransactionDT order, with nulls
            "2,1,90000,50.0,H,9500,111.0,150.0,visa,226.0,debit,325.0,87.0,,,gmail.com,gmail.com,"
            + ",".join(["2"] * 14)
            + ",T,T,F,M0,,,,,,"
            + ",".join(["14.0"] + [""] * 14)
            + ",,0.5\n"
            "1,0,86400,68.5,W,13926,,150.0,discover,142.0,credit,315.0,87.0,19.0,,,,"
            + ",".join(["1"] * 14)
            + ",T,T,T,M2,F,T,,,,"
            + ",".join(["0.0"] + [""] * 14)
            + ",1.0,\n"
        )
        identity = tmp_path / "train_identity.csv"
        identity.write_text("TransactionID,DeviceType,DeviceInfo\n2,mobile,iOS Device\n")
        return tx, identity

    def test_sorted_joined_and_mapped(self, csvs):
        frame = load_replay_frame(["V1", "V2"], 0, *csvs)
        txs = list(iter_transactions(frame, ["V1", "V2"]))
        assert [t.transaction_id for t in txs] == [1, 2]  # TransactionDT order
        first, second = txs
        assert first.event_ts == datetime(2025, 1, 2, tzinfo=UTC)
        assert first.card_key == derive_card_key(13926, None, 142.0, 315.0)
        assert first.dist2 is None and first.p_emaildomain is None
        assert (first.m1, first.m4, first.m5) == (True, None, False)  # M4 is categorical
        assert first.v == {"V1": 1.0}  # null V omitted → NaN downstream
        assert first.device_type is None
        assert second.device_type == "mobile" and second.is_fraud == 1
        assert second.c14 == 2.0
        assert (first.card1, first.card2, first.addr1) == (13926, None, 315.0)
        assert (first.d1, second.d1, second.d2) == (0.0, 14.0, None)

    def test_limit(self, csvs):
        assert load_replay_frame([], 1, *csvs).height == 1


def test_pipeline_config_parses():
    from streamguard.config import load_pipeline_config

    cfg = load_pipeline_config()
    assert cfg["redis"]["key_prefix"] == "sg:card:"
    assert set(cfg["topics"]) == {"raw", "scored", "alerts"}

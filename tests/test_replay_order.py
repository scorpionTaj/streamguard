"""Replay clock & pacing tests (PLAN.md §7)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from streamguard.producer.replay import (
    DEFAULT_MAX_RATE_TX_PER_SEC,
    DEFAULT_TIME_COMPRESSION,
    pace_delay,
    to_event_ts,
)


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

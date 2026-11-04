"""Drift watcher tests (ops/watch.py, PLAN.md §10) — no server, no MLflow."""

from __future__ import annotations

import httpx
import pytest

from streamguard.ops.watch import DriftWatcher, decide


class TestDecide:
    def test_requires_the_full_run_of_flags(self):
        assert decide([], 3) is False
        assert decide([True, True], 3) is False
        assert decide([True, True, False, True, True], 3) is False

    def test_consecutive_flags_trigger(self):
        assert decide([False, True, True, True], 3) is True
        assert decide([True, True, True], 3) is True

    def test_zero_and_negative_never_trigger(self):
        assert decide([True, True], 0) is False


def _watcher(flags: list[dict], failures: int = 0, clock: list[float] | None = None):
    """Watcher with scripted poll responses and a failing-until-N train stub."""
    calls = {"poll": 0, "train": 0}

    def poll():
        report = flags[min(calls["poll"], len(flags) - 1)]
        calls["poll"] += 1
        return report

    def train(report):
        calls["train"] += 1
        if calls["train"] <= failures:
            raise RuntimeError("train boom")

    watcher = DriftWatcher(api="fake", consecutive=3, cooldown_s=100.0, poll=poll, train=train,
                           clock=lambda: clock[0] if clock else 0.0)  # fmt: skip
    return watcher, calls


FLAG = {"rows": 20000, "drift_flag": True, "flagged": ["c1"]}
CLEAN = {"rows": 20000, "drift_flag": False, "flagged": []}


class TestPollOnce:
    def test_retrains_only_on_three_consecutive_flags(self):
        watcher, calls = _watcher([CLEAN, FLAG, FLAG, FLAG])
        assert watcher.poll_once() is None  # clean
        assert watcher.poll_once() is None  # 1st flag
        assert watcher.poll_once() is None  # 2nd flag
        record = watcher.poll_once()  # 3rd consecutive flag
        assert record == {"triggered_at": record["triggered_at"], "flagged": ["c1"],
                          "rows": 20000}  # fmt: skip
        assert calls["train"] == 1 and watcher.flags == []  # flags reset after triggering

    def test_window_not_ready_is_not_evidence(self):
        watcher, calls = _watcher(
            [{"rows": 500, "window": 20000, "detail": "filling"}, FLAG, FLAG, FLAG]
        )
        for _ in range(4):
            watcher.poll_once()
        assert calls["train"] == 1  # the 'detail' report never counted as a flag

    def test_clean_window_resets_the_run(self):
        watcher, calls = _watcher([FLAG, FLAG, CLEAN, FLAG, FLAG, FLAG])
        for _ in range(6):
            watcher.poll_once()
        assert calls["train"] == 1  # only the post-reset run of 3 triggers

    def test_cooldown_blocks_an_immediate_second_retrain(self):
        watcher, calls = _watcher([FLAG], clock=[0.0])
        times = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        watcher.clock = lambda: next(times)
        for _ in range(3):
            watcher.poll_once()
        assert calls["train"] == 1
        watcher.flags = [True, True, True]  # flags rebuilt during the cooldown
        assert watcher.poll_once() is None  # 30 s < 100 s cooldown
        assert calls["train"] == 1

    def test_train_failure_backs_off_without_hot_looping(self):
        watcher, calls = _watcher([FLAG], failures=1, clock=[0.0])
        times = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        watcher.clock = lambda: next(times)
        for _ in range(3):
            result = watcher.poll_once()
        assert calls["train"] == 1 and result == {"triggered_at": result["triggered_at"],
                                                  "flagged": ["c1"], "rows": 20000,
                                                  "error": True}  # fmt: skip
        watcher.flags = [True, True, True]
        assert watcher.poll_once() is None  # cooldown set even though the train failed
        assert calls["train"] == 1

    def test_stale_history_beyond_the_run_is_dropped(self):
        watcher, _ = _watcher([FLAG])
        for _ in range(10):
            watcher.poll_once()
        assert len(watcher.flags) <= 3


def test_policy_defaults_mirror_the_docs():
    watcher = DriftWatcher(api="fake")
    assert watcher.consecutive == 3 and watcher.cooldown_s == 3600.0
    with pytest.raises(httpx.HTTPError):  # the real poll is wired and fails on a fake api
        watcher._poll_http()

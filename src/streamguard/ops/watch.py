"""Drift watcher — closes the MLOps loop (PLAN.md §10).

Polls the scorer's live drift report every 30 s. N *consecutive* flagged windows
(one window = one scorer cadence) trigger a single `make train` run, which
registers a fresh challenger. The watcher NEVER promotes: the canary gate plus
`make promote`/`make rollback` stay the human decision — an auto-retrain is a
candidate, not a verdict. A cooldown prevents retrain storms, and the flags
reset after every attempt so a persistent shift re-trains at most once per
cooldown.

Requires `make mlflow` (the train run registers through it) and `make api`.

CLI: python -m streamguard.ops.watch [--consecutive 3] [--cooldown 3600]
(`make watch-drift`; stop with Ctrl-C.)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime

log = logging.getLogger("streamguard.watch")

TRAIN_CMD = ("uv", "run", "python", "-m", "streamguard.models.train")


def decide(flags: list[bool], consecutive: int) -> bool:
    """True iff the last `consecutive` drift reports are all flagged.

    A single flagged window is the calibrated monitor doing its job on a volatile
    feature; only a run of them says the champion no longer fits the stream.
    """
    return consecutive > 0 and len(flags) >= consecutive and all(flags[-consecutive:])


@dataclass
class DriftWatcher:
    """Poll → decide → retrain-once-per-cooldown. `poll` and `train` are injectable for tests."""

    api: str
    consecutive: int = 3
    cooldown_s: float = 3600.0
    poll: object = None  # () → report dict
    train: object = None  # (report) → None, raises on failure
    clock: object = time.monotonic

    def __post_init__(self) -> None:
        self._poll = self.poll or self._poll_http
        self._train = self.train or self._train_subprocess
        self._clock = self.clock
        self.flags: list[bool] = []
        self.retrains: list[dict] = []
        self._last_train = -self.cooldown_s  # an early persistent shift may retrain at once

    def _poll_http(self) -> dict:
        import httpx

        return httpx.get(f"{self.api}/drift", timeout=30).json()

    def _train_subprocess(self, report: dict) -> None:
        result = subprocess.run(TRAIN_CMD, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"train failed (exit {result.returncode}): {result.stderr[-2000:]}")

    def poll_once(self) -> dict | None:
        """One drift check; returns the retrain record when a retrain was started."""
        report = self._poll()
        if report.get("detail") or "drift_flag" not in report:
            log.info("drift monitor not ready: %s", report.get("detail") or report.get("rows", 0))
            return None  # window still filling — not evidence either way
        self.flags.append(bool(report["drift_flag"]))
        del self.flags[: -self.consecutive]  # keep only the tail that can trigger
        now = self._clock()
        if not (decide(self.flags, self.consecutive)
                and now - self._last_train >= self.cooldown_s):  # fmt: skip
            return None
        record = {"triggered_at": datetime.now(UTC).isoformat(timespec="seconds"),
                  "flagged": report.get("flagged", []), "rows": report.get("rows", 0)}  # fmt: skip
        log.warning(
            "drift flagged %d× consecutively (%s) → retraining a challenger",
            self.consecutive,
            ", ".join(record["flagged"]) or "unknown features",
        )
        self.flags.clear()
        self._last_train = now  # the cooldown covers a failing trainer too — no hot loop
        try:
            self._train(report)
        except Exception:
            log.exception("retrain failed; backing off for %.0f s", self.cooldown_s)
            return record | {"error": True}
        self.retrains.append(record)
        self._tag_challenger(record)
        log.info("new challenger registered (champion untouched — canary still decides)")
        return record

    def _tag_challenger(self, record: dict) -> None:
        """Record why this challenger exists (mirrors the drift tagging in eval/drift.py)."""
        from streamguard.models.registry import CHALLENGER, MODEL_NAME, _client

        try:
            client = _client()
            version = client.get_model_version_by_alias(MODEL_NAME, CHALLENGER).version
            client.set_model_version_tag(MODEL_NAME, version, "retrain_trigger", "drift")
            client.set_model_version_tag(MODEL_NAME, version, "drift_features",
                                         ",".join(record["flagged"]) or "-")  # fmt: skip
        except Exception:
            log.exception("could not tag the new challenger (retrain itself succeeded)")

    def run(self, every_s: float = 30.0) -> None:
        """Poll until interrupted — same cadence as the scorer's drift loop."""
        log.info("watching %s every %.0f s: %d consecutive flags → retrain "
                 "(cooldown %.0f s, promotion stays manual)", self.api, every_s,
                 self.consecutive, self.cooldown_s)  # fmt: skip
        while True:
            try:
                self.poll_once()
            except Exception:
                log.exception("drift poll failed; retrying next cycle")
            time.sleep(every_s)


def main() -> None:
    """Entry point for `make watch-drift`."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Auto-retrain a challenger on persistent drift")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument(
        "--consecutive",
        type=int,
        default=3,
        help="flagged windows in a row before retraining (30 s each)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=3600.0, help="minimum seconds between retrains"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    DriftWatcher(api=args.api, consecutive=args.consecutive, cooldown_s=args.cooldown).run()


if __name__ == "__main__":
    main()

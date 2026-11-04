"""End-to-end latency benchmark — milestone M6 (PLAN.md §3, §13).

Contract: run the full pipeline (replay → broker → Spark → Redis → scorer →
alert publish) for a fixed event count, record per-hop spans from producer
timestamps, and write artifacts/bench.json with p50/p95/p99 per hop plus the
end-to-end total. KPI: p95 end-to-end < 5 s @ 200 tx/s.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for `make bench`. — M6"""
    raise NotImplementedError("M6: instrumented run → artifacts/bench.json")


if __name__ == "__main__":
    main()

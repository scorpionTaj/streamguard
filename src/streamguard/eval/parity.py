"""Online/offline feature parity test — milestone M2 (PLAN.md §8).

Contract: sample 1k events from transactions.raw, compute features two ways —
offline twin (offline/features.py, polars) vs online path (Spark job output in
Redis) — and assert max per-feature |Δ| < 1e-6. Failure means the streaming
and training feature definitions have diverged; CI must fail.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for `make parity`. — M2"""
    raise NotImplementedError("M2: 1k-event offline vs Redis comparison (δ < 1e-6)")


if __name__ == "__main__":
    main()

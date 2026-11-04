"""PSI drift monitor — milestone M6 (PLAN.md §8, §10).

Contract: hourly, compare the last-24h live feature distributions against the
training reference histogram (stored as a model artifact) using
eval.metrics.psi; flag any top-20-importance feature with PSI > 0.2 (Grafana
panel + MLflow tag = retrain flag).
"""

from __future__ import annotations


def main() -> None:
    """Entry point for `make drift`. — M6"""
    raise NotImplementedError("M6: PSI report vs training reference")


if __name__ == "__main__":
    main()

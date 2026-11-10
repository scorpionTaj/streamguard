"""Write a small IEEE-CIS-shaped transaction CSV for the CI integration job.

The real dataset needs a Kaggle account and cannot be fetched in CI. This file has the
same columns the replay producer reads, so the real producer, Spark job and parity check
run against it unchanged. The values are random: it exercises the streaming plumbing
(ordering, ties, state, Redis), not the model.

Usage: python scripts/synthetic_data.py --out data/synthetic_transaction.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from streamguard.config import load_feature_spec

FIRST_TRANSACTION_DT = 86_400  # the real dataset starts one day after its reference


def synthetic_frame(rows: int, cards: int, seed: int = 42) -> pl.DataFrame:
    """`rows` transactions over `cards` cards, in non-decreasing TransactionDT order.

    Every card appears at least once. About one gap in ten is zero, so several events
    share a timestamp and the replay-order tie-break is exercised.
    """
    if rows < cards:
        raise ValueError("rows must be at least the number of cards")
    rng = np.random.default_rng(seed)
    card = rng.permutation(np.arange(rows) % cards)
    gaps = np.where(rng.random(rows) < 0.1, 0, rng.geometric(1 / 120, rows))
    columns: dict[str, object] = {
        "TransactionID": np.arange(1, rows + 1),
        "isFraud": (rng.random(rows) < 0.035).astype(int),
        "TransactionDT": (FIRST_TRANSACTION_DT + np.cumsum(gaps)).astype(float),
        "TransactionAmt": np.round(rng.lognormal(4.0, 1.0, rows), 3),
        "ProductCD": rng.choice(list("WCRHS"), rows),
        "card1": 1000 + card,
        "card2": 111.0 + card % 7,
        "card3": np.full(rows, 150.0),
        "card4": rng.choice(["visa", "mastercard"], rows),
        "card5": 226.0 - card % 3,
        "card6": rng.choice(["debit", "credit"], rows),
        "addr1": 300.0 + card % 50,
        "addr2": np.full(rows, 87.0),
        "dist1": np.round(rng.exponential(20.0, rows), 1),
        "dist2": pl.Series([None] * rows, dtype=pl.Float64),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], rows),
        "R_emaildomain": pl.Series([None] * rows, dtype=pl.Utf8),
    }
    columns |= {f"C{i}": rng.integers(0, 5, rows).astype(float) for i in range(1, 15)}
    columns |= {f"M{i}": rng.choice(["T", "F"], rows) for i in range(1, 10)}
    columns |= {f"D{i}": np.round(rng.exponential(30.0, rows), 0) for i in range(1, 16)}
    columns |= {name: np.round(rng.random(rows), 4) for name in load_feature_spec()["v_subset"]}
    return pl.DataFrame(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_transaction.csv"))
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--cards", type=int, default=300)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    synthetic_frame(args.rows, args.cards).write_csv(args.out)
    print(f"wrote {args.rows:,} transactions over {args.cards} cards → {args.out}")


if __name__ == "__main__":
    main()

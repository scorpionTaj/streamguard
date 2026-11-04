#!/usr/bin/env python3
"""Dataset download script — milestone M0/M1 (`make data`).

Primary: IEEE-CIS Fraud Detection (Kaggle competition)
    kaggle competitions download -c ieee-fraud-detection -p data/
    unzip into data/train_transaction.csv + data/train_identity.csv
    (requires ~/.kaggle/kaggle.json — see .env.example)

Fallback: Sparkov synthetic (dataset, no competition rules to accept)
    kaggle datasets download -d kartik2112/fraud-detection -p data/

Usage: uv run python scripts/download_data.py --source ieee|sparkov
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["ieee", "sparkov"], default="ieee")
    args = parser.parse_args()
    raise NotImplementedError(f"M1: download {args.source} via kaggle CLI")


if __name__ == "__main__":
    main()

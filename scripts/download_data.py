#!/usr/bin/env python3
"""Dataset download script (`make data`).

Primary: IEEE-CIS Fraud Detection (Kaggle competition)
    → data/train_transaction.csv + data/train_identity.csv
    (requires KAGGLE_USERNAME/KAGGLE_KEY in .env or ~/.kaggle/kaggle.json, and the
    competition rules accepted once at https://www.kaggle.com/c/ieee-fraud-detection/rules)

Fallback: Sparkov synthetic (dataset, no competition rules to accept)
    → data/sparkov/

Usage: uv run python scripts/download_data.py --source ieee|sparkov
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IEEE_FILES = ("train_transaction.csv", "train_identity.csv")


def _kaggle(*args: str) -> None:
    cmd = [sys.executable, "-m", "kaggle", *args]
    if subprocess.run(cmd, check=False).returncode != 0:
        sys.exit(
            "kaggle download failed — check credentials (.env KAGGLE_USERNAME/KAGGLE_KEY) and, "
            "for --source ieee, accept the rules at https://www.kaggle.com/c/ieee-fraud-detection/rules"
        )


def download_ieee() -> None:
    # Train files only: the test split is unlabeled → not needed (PLAN.md §5).
    for name in IEEE_FILES:
        if (DATA_DIR / name).exists():
            continue
        _kaggle("competitions", "download", "-c", "ieee-fraud-detection", "-f", name,
                "-p", str(DATA_DIR))  # fmt: skip
        archive = DATA_DIR / f"{name}.zip"  # kaggle zips large single files
        if archive.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extract(name, DATA_DIR)
            archive.unlink()
    print(f"IEEE-CIS ready: {', '.join(IEEE_FILES)}")


def download_sparkov() -> None:
    target = DATA_DIR / "sparkov"
    _kaggle(
        "datasets", "download", "-d", "kartik2112/fraud-detection", "-p", str(target), "--unzip"
    )
    print(f"Sparkov ready in {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["ieee", "sparkov"], default="ieee")
    args = parser.parse_args()
    load_dotenv()  # kaggle CLI reads credentials from the environment
    # New-style tokens (KGAT_…) are only read from KAGGLE_API_TOKEN by kaggle ≥ 2.
    key = os.getenv("KAGGLE_KEY", "")
    if key.startswith("KGAT_") and not os.getenv("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = key
    if (
        not os.getenv("KAGGLE_API_TOKEN")
        and not (os.getenv("KAGGLE_USERNAME") and key)
        and not any((Path.home() / ".kaggle" / f).exists() for f in ("kaggle.json", "access_token"))
    ):
        sys.exit(
            "no Kaggle credentials: fill KAGGLE_USERNAME/KAGGLE_KEY in .env (see .env.example)"
        )
    DATA_DIR.mkdir(exist_ok=True)
    download_ieee() if args.source == "ieee" else download_sparkov()


if __name__ == "__main__":
    main()

"""Offline (polars) feature twin — milestones M1–M2 (PLAN.md §6, §8).

Responsibilities:
1. `build_training_frame()`: read data/train_transaction.csv (+ identity join),
   select features strictly from configs/features.yaml, apply the same
   transformations as the online job, and emit a feature matrix for training.
2. Serve as the OFFLINE side of the parity test (eval/parity.py): for 1k
   sampled events, offline features must equal the Spark/Redis online values
   within 1e-6, otherwise the pipeline is inconsistent and CI fails.

Implemented: pure helpers shared with tests. Milestone M1: CSV loading,
v_subset selection (mutual information, train split only), transformation body.
"""

from __future__ import annotations

# Known mailbox providers in IEEE-CIS email domains; everything else → "other".
_EMAIL_GROUPS: tuple[str, ...] = (
    "gmail",
    "yahoo",
    "hotmail",
    "outlook",
    "aol",
    "protonmail",
    "icloud",
    "comcast",
    "verizon",
    "me.com",
    "web.de",
    "mail.com",
)

NO_EMAIL = "unknown"
OTHER_EMAIL = "other"


def email_domain_group(domain: str | None) -> str:
    """Map a raw email domain (e.g. 'gmail.com', 'mail.yahoo.fr') to its group.

    >>> email_domain_group("mail.yahoo.fr")
    'yahoo'
    >>> email_domain_group("somecorp.cz")
    'other'
    """
    if not domain or not domain.strip():
        return NO_EMAIL
    cleaned = domain.strip().lower()
    for group in _EMAIL_GROUPS:
        if group in cleaned:
            return group
    return OTHER_EMAIL


def build_training_frame(source_csv: str = "data/train_transaction.csv") -> None:
    """Return the training feature frame per configs/features.yaml. — M1"""
    raise NotImplementedError("M1: polars load + feature build from features.yaml")


if __name__ == "__main__":
    build_training_frame()

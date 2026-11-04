"""Kafka data contracts (PLAN.md §4). Implemented — the rest of the repo builds on these.

Topics (names live in configs/pipeline.yaml):
    transactions.raw    → Transaction
    transactions.scored → ScoredTx
    transactions.alerts → Alert
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskBand = Literal["high", "medium", "review"]


def derive_card_key(card1: int | str, card2: int | str, card5: int | str, addr1: int | str) -> str:
    """Stable pseudo-PAN: sha256 of the IEEE-CIS card composite, first 16 hex chars.

    The composite (card1, card2, card5, addr1) identifies a card in the dataset;
    hashing keeps the raw values out of the event stream.
    """
    composite = f"{card1}|{card2}|{card5}|{addr1}"
    return hashlib.sha256(composite.encode()).hexdigest()[:16]


class TopDriver(BaseModel):
    """One SHAP contribution attached to an alert (filled asynchronously)."""

    feature: str
    impact: float  # signed SHAP value: >0 pushes toward fraud


class Transaction(BaseModel):
    """One labeled payment event on `transactions.raw` (PLAN.md §4)."""

    transaction_id: int
    event_ts: datetime  # synthetic wall clock derived from TransactionDT (§7)
    card_key: str = Field(min_length=16, max_length=16, pattern=r"^[0-9a-f]{16}$")
    transaction_amt: float = Field(ge=0)
    product_cd: str = Field(pattern=r"^[WCRHS]$")
    card4: str | None = None  # network: visa/mastercard/…
    card6: str | None = None  # type: credit/debit/…
    dist1: float | None = None
    dist2: float | None = None
    p_emaildomain: str | None = None
    r_emaildomain: str | None = None
    m1: bool | None = None
    m2: bool | None = None
    m3: bool | None = None
    m4: bool | None = None
    m5: bool | None = None
    m6: bool | None = None
    m7: bool | None = None
    m8: bool | None = None
    m9: bool | None = None
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    c7: float = 0.0
    c8: float = 0.0
    c9: float = 0.0
    c10: float = 0.0
    c11: float = 0.0
    c12: float = 0.0
    c13: float = 0.0
    c14: float = 0.0
    v: dict[str, float] = Field(default_factory=dict)  # only the frozen v_subset (§6)
    device_type: str | None = None
    device_info: str | None = None
    is_fraud: int = Field(ge=0, le=1)  # ground truth, replayed labeled for live eval


class ScoredTx(Transaction):
    """Transaction + champion model output, published on `transactions.scored`."""

    score: float = Field(ge=0.0, le=1.0)
    model_version: str
    scored_at: datetime


class Alert(ScoredTx):
    """Threshold-passing transaction on `transactions.alerts` — drives the dashboard."""

    risk_band: RiskBand
    threshold: float
    velocity_snapshot: dict[str, float] = Field(default_factory=dict)  # Redis features at scoring time
    top_drivers: list[TopDriver] = Field(default_factory=list)  # filled async (SHAP, M5)

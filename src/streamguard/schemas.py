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

RiskBand = Literal["high", "medium"]


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
    # Anonymized IEEE-CIS card/address IDs (not PANs) — strong model features (M1 ablation).
    card1: int | None = None
    card2: float | None = None
    card3: float | None = None
    card5: float | None = None
    addr1: float | None = None
    addr2: float | None = None
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
    d1: float | None = None  # D1–D15: timedeltas (days) since prior card events, as provided
    d2: float | None = None
    d3: float | None = None
    d4: float | None = None
    d5: float | None = None
    d6: float | None = None
    d7: float | None = None
    d8: float | None = None
    d9: float | None = None
    d10: float | None = None
    d11: float | None = None
    d12: float | None = None
    d13: float | None = None
    d14: float | None = None
    d15: float | None = None
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
    # Redis features at scoring time
    velocity_snapshot: dict[str, float] = Field(default_factory=dict)
    top_drivers: list[TopDriver] = Field(default_factory=list)  # filled async (SHAP, M5)

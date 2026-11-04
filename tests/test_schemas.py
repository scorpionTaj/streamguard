"""Data-contract tests for schemas.py (PLAN.md §4)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from streamguard.schemas import Alert, ScoredTx, Transaction, derive_card_key


def make_transaction(**overrides) -> dict:
    base = {
        "transaction_id": 3020851,
        "event_ts": datetime(2025, 3, 14, 12, 30, 0, tzinfo=UTC),
        "card_key": "a" * 16,
        "transaction_amt": 129.95,
        "product_cd": "W",
        "card4": "visa",
        "card6": "credit",
        "dist1": 12.0,
        "p_emaildomain": "gmail.com",
        "m1": True,
        "m5": False,
        "c1": 1.5,
        "c13": 4.0,
        "v": {"V1": 0.5, "V14": -2.1},
        "device_type": "desktop",
        "is_fraud": 0,
    }
    base.update(overrides)
    return base


class TestCardKey:
    def test_deterministic_and_shape(self):
        key = derive_card_key(10674, 315.0, 135.0, 296.0)
        assert len(key) == 16
        assert key == derive_card_key(10674, 315.0, 135.0, 296.0)

    def test_different_cards_differ(self):
        assert derive_card_key(1, 2, 3, 4) != derive_card_key(1, 2, 3, 5)


class TestTransaction:
    def test_roundtrip(self):
        tx = Transaction(**make_transaction())
        assert tx.is_fraud == 0
        assert tx.v["V14"] == -2.1

    def test_rejects_bad_card_key(self):
        with pytest.raises(ValidationError):
            Transaction(**make_transaction(card_key="xyz"))

    def test_rejects_bad_product_cd(self):
        with pytest.raises(ValidationError):
            Transaction(**make_transaction(product_cd="X"))

    def test_rejects_out_of_range_label(self):
        with pytest.raises(ValidationError):
            Transaction(**make_transaction(is_fraud=2))


class TestScoredAndAlert:
    def test_scoredtx_inherits_transaction(self):
        scored = ScoredTx(
            **make_transaction(), score=0.93, model_version="v7", scored_at=datetime.now(UTC)
        )
        assert scored.score == 0.93
        assert scored.transaction_amt == 129.95

    def test_alert_risk_band_validated(self):
        with pytest.raises(ValidationError):
            Alert(
                **make_transaction(),
                score=0.93,
                model_version="v7",
                scored_at=datetime.now(UTC),
                risk_band="critical",  # type: ignore[arg-type]
                threshold=0.5,
            )

    def test_alert_defaults(self):
        alert = Alert(
            **make_transaction(),
            score=0.93,
            model_version="v7",
            scored_at=datetime.now(UTC),
            risk_band="high",
            threshold=0.5,
        )
        assert alert.velocity_snapshot == {}
        assert alert.top_drivers == []

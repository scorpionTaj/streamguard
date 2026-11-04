"""Pure data shaping for the dashboard (no Streamlit, no network) — unit-tested."""

from __future__ import annotations

import pandas as pd

BANDS = ["high", "medium"]
BAND_ICON = {"high": "🔴", "medium": "🟠"}
BAND_HEX = {"high": "#ff5c5c", "medium": "#ffb020"}  # matches .streamlit/config.toml
OK_HEX, BAD_HEX, NEUTRAL_HEX = "#3ddc97", "#ff5c5c", "#5c9dff"
WINDOWS = ["5m", "1h", "24h", "7d"]
LABELS = ["all", "fraud", "legit"]


def filter_alerts(alerts: list[dict], bands: list[str], label: str = "all",
                  min_score: float = 0.0, card_query: str = "") -> list[dict]:  # fmt: skip
    """Newest first, restricted to the sidebar filters."""
    query = card_query.strip().lower()
    keep = [
        a for a in alerts
        if a["risk_band"] in bands
        and a["score"] >= min_score
        and (label == "all" or (label == "fraud") == bool(a["is_fraud"]))
        and (not query or query in a["card_key"].lower() or query == str(a["transaction_id"]))
    ]  # fmt: skip
    return sorted(keep, key=lambda a: a["scored_at"], reverse=True)


def alerts_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame({
        "band": [f"{BAND_ICON[a['risk_band']]} {a['risk_band']}" for a in rows],
        "score": [a["score"] for a in rows],
        "transaction": [a["transaction_id"] for a in rows],
        "event time": [a["event_ts"] for a in rows],
        "card": [a["card_key"] for a in rows],
        "amount": [a["transaction_amt"] for a in rows],
        "product": [a["product_cd"] for a in rows],
        "label": ["fraud" if a["is_fraud"] else "legit" for a in rows],
    })  # fmt: skip


def precision_by_band(alerts: list[dict]) -> pd.DataFrame:
    """Alerts and labeled-fraud share per band (labels are replayed with the stream)."""
    rows = []
    for band in BANDS:
        in_band = [a for a in alerts if a["risk_band"] == band]
        frauds = sum(int(a["is_fraud"]) for a in in_band)
        rows.append({"band": f"{BAND_ICON[band]} {band}", "alerts": len(in_band),
                     "frauds": frauds,
                     "precision": frauds / len(in_band) if in_band else None})  # fmt: skip
    return pd.DataFrame(rows)


def velocity_table(snapshot: dict[str, float]) -> pd.DataFrame:
    """Snapshot fields `<metric>_<window>` → one row per window."""
    return pd.DataFrame(
        {
            "transactions": [snapshot.get(f"txn_count_{w}") for w in WINDOWS],
            "amount sum": [snapshot.get(f"amt_sum_{w}") for w in WINDOWS],
            "amount mean": [snapshot.get(f"amt_mean_{w}") for w in WINDOWS],
        },
        index=pd.Index(WINDOWS, name="window"),
    )


def drivers_frame(drivers: list[dict]) -> pd.DataFrame:
    """SHAP top-drivers → signed series (feature → log-odds impact) for a diverging bar chart."""
    return pd.Series({d["feature"]: d["impact"] for d in drivers}, dtype=float, name="impact")


def drift_table(report: dict, top: int = 12) -> pd.DataFrame:
    """Features with the highest drift ratio (max of PSI / threshold over both statistics)."""
    ratio = report.get("ratio") or {}
    if not ratio:
        return pd.DataFrame(columns=["feature", "top-20", "ratio", "PSI", "present-only PSI"])
    important = set(report.get("important", []))
    names = sorted(ratio, key=ratio.get, reverse=True)[:top]
    return pd.DataFrame({
        "feature": names,
        "top-20": ["★" if n in important else "" for n in names],
        "ratio": [ratio[n] for n in names],
        "PSI": [report["features"][n] for n in names],
        "present-only PSI": [report["finite_psi"][n] for n in names],
    })  # fmt: skip


def canary_verdict(canary: dict, tolerance: float = 0.02, min_events: int = 1000) -> dict:
    """Model-health canary banner state (mirrors the promotion gate in serving/canary.py)."""
    if not canary.get("challenger_version"):
        return {"state": "none",
                "detail": "No challenger registered — `make train` creates one."}  # fmt: skip
    if canary.get("challenger_version") == canary.get("champion_version"):
        return {"state": "same",
                "detail": "Challenger is the current champion — `make train` registers a fresh "
                          "challenger."}  # fmt: skip
    events = canary.get("events", 0)
    if events < min_events:
        return {"state": "collecting",
                "detail": f"Collecting: {events:,} / {min_events:,} events compared."}  # fmt: skip
    p95 = canary["p95_abs_delta"]
    if canary.get("within_tolerance"):
        return {
            "state": "pass",
            "detail": f"p95 |Δ| {p95:.3f} < {tolerance} — `make promote` would succeed.",
        }
    return {"state": "fail",
            "detail": f"p95 |Δ| {p95:.3f} ≥ {tolerance} — promotion is blocked."}  # fmt: skip

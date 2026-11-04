"""StreamGuard live dashboard (PLAN.md §2) — `make dashboard`, expects `make api` on :8000.

Header (every 2 s): scored, throughput, alerts, live precision, p95 latency, models.
Tabs:  🚨 Live alerts  — filtered feed + drill-down (transaction, velocity, SHAP drivers)
       🩺 Model health — champion/challenger canary, calibrated drift, precision by band
Labels are replayed with the stream, so alert precision is measured live.
Data shaping lives in views.py (unit-tested); this file is layout + polling only.
"""

from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from views import (
    BAND_ICON,
    BANDS,
    LABELS,
    alerts_table,
    canary_verdict,
    drift_table,
    filter_alerts,
    precision_by_band,
    velocity_table,
)

API_BASE = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
MAX_ALERTS = 1000

st.set_page_config(page_title="StreamGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ StreamGuard — live fraud alerts")

if "alerts" not in st.session_state:
    st.session_state.alerts = {}  # transaction_id → alert dict
    st.session_state.since = None


def get_json(url: str, **params) -> dict | None:
    """GET → JSON; None when the service is not up (the page keeps rendering)."""
    try:
        response = httpx.get(url, params=params, timeout=1.5)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def prometheus_scalar(query: str) -> float | None:
    body = get_json(f"{PROMETHEUS}/api/v1/query", query=query)
    try:
        value = float(body["data"]["result"][0]["value"][1])
    except (TypeError, KeyError, IndexError, ValueError):
        return None
    return None if value != value else value  # NaN → None


def poll_alerts() -> None:
    params = {"since": st.session_state.since} if st.session_state.since else {}
    body = get_json(f"{API_BASE}/alerts", limit=MAX_ALERTS, **params)
    for alert in (body or {}).get("alerts", []):
        st.session_state.alerts[alert["transaction_id"]] = alert
        st.session_state.since = max(st.session_state.since or "", alert["scored_at"])
    if len(st.session_state.alerts) > MAX_ALERTS:  # keep the newest
        newest = sorted(st.session_state.alerts.values(), key=lambda a: a["scored_at"])
        st.session_state.alerts = {a["transaction_id"]: a for a in newest[-MAX_ALERTS:]}


def current_alerts() -> list[dict]:
    return filter_alerts(
        list(st.session_state.alerts.values()),
        st.session_state.get("band_filter", BANDS),
        st.session_state.get("label_filter", "all"),
        st.session_state.get("min_score", 0.0),
        st.session_state.get("card_query", ""),
    )


@st.fragment(run_every=2)
def header() -> None:
    poll_alerts()
    stats = get_json(f"{API_BASE}/stats")
    if stats is None:
        st.warning("Scorer API not reachable on :8000 — run `make api`.")
    by_band = (stats or {}).get("by_band", {})
    # 1-minute rate from Prometheus: the scorer works in bursts, a 2 s delta jumps 0 ↔ 200
    rate = prometheus_scalar("sum(rate(streamguard_scored_total[1m]))")
    p95 = prometheus_scalar(
        "histogram_quantile(0.95, sum(rate(streamguard_e2e_latency_seconds_bucket[1m])) by (le))"
    )
    alerts = list(st.session_state.alerts.values())
    labeled = [a for a in alerts if a["risk_band"] in ("high", "medium")]
    precision = sum(a["is_fraud"] for a in labeled) / len(labeled) if labeled else None

    cols = [*st.columns(3), *st.columns(3)]  # two rows: readable on narrow screens
    cols[0].metric("Transactions scored", f"{stats['scored']:,}" if stats else "—")
    cols[1].metric("Throughput (1m)", f"{rate:.0f} tx/s" if rate is not None else "—")
    cols[2].metric("Alerts", f"{sum(by_band.values()):,}" if stats else "—",
                   help=" · ".join(f"{b}: {by_band.get(b, 0)}" for b in BANDS))  # fmt: skip
    cols[3].metric("🔴 high / 🟠 medium", f"{by_band.get('high', 0)} / {by_band.get('medium', 0)}")
    cols[4].metric("Live alert precision", f"{precision:.0%}" if precision is not None else "—",
                   help="Share of high+medium alerts on this page labeled fraud")  # fmt: skip
    cols[5].metric("p95 latency (1m)", f"{p95:.2f} s" if p95 is not None else "—",
                   help="Producer send → alert published, from Prometheus; KPI < 5 s")  # fmt: skip
    if stats and stats.get("model_version"):
        st.caption(f"Champion v{stats['model_version']} · "
                   f"{stats.get('feature_misses', 0)} events scored without a velocity snapshot")  # fmt: skip


@st.fragment(run_every=2)
def feed() -> None:
    rows = current_alerts()
    if not st.session_state.alerts:
        st.info("Waiting for alerts… `make features`, `make api`, then `make replay`.")
        return
    st.caption(f"{len(rows):,} of {len(st.session_state.alerts):,} recent alerts match the filters")
    st.dataframe(
        alerts_table(rows), hide_index=True, height=380, use_container_width=True,
        column_config={
            "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0,
                                                     format="%.3f"),
            "amount": st.column_config.NumberColumn("amount", format="$%.2f"),
            "transaction": st.column_config.NumberColumn("transaction", format="%d"),
        },
    )  # fmt: skip


@st.fragment(run_every=2)
def drill_down() -> None:
    st.subheader("Alert drill-down")
    rows = current_alerts()
    if not rows:
        st.caption("Select an alert once some match the filters.")
        return
    by_id = {a["transaction_id"]: a for a in rows}
    tx_id = st.selectbox(
        "Alert", list(by_id), key="selected_alert",
        format_func=lambda i: f"{BAND_ICON[by_id[i]['risk_band']]} tx {i} · "
                              f"score {by_id[i]['score']:.3f} · ${by_id[i]['transaction_amt']:,.2f}",
    )  # fmt: skip
    # re-fetch: SHAP drivers are attached asynchronously after the alert first arrives
    alert = get_json(f"{API_BASE}/alerts/{tx_id}") or by_id[tx_id]
    st.session_state.alerts[tx_id] = alert
    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"**{BAND_ICON[alert['risk_band']]} {alert['risk_band'].upper()}** — "
                    f"score {alert['score']:.3f} ≥ band floor {alert['threshold']:.3f}")  # fmt: skip
        st.markdown(f"Ground truth: **{'fraud' if alert['is_fraud'] else 'legit'}** · "
                    f"model v{alert['model_version']}")  # fmt: skip
        details = {
            "card": alert["card_key"], "amount": f"${alert['transaction_amt']:,.2f}",
            "product": alert["product_cd"], "network / type": f"{alert['card4']} / {alert['card6']}",
            "payer email": alert["p_emaildomain"], "device": alert["device_type"],
            "event time": alert["event_ts"], "scored at": alert["scored_at"],
        }  # fmt: skip
        st.table(pd.Series({k: "—" if v is None else str(v) for k, v in details.items()},
                           name="value"))  # fmt: skip
    with right:
        snapshot = alert.get("velocity_snapshot") or {}
        st.markdown("**Card velocity at scoring time** (Redis snapshot)")
        if snapshot:
            st.dataframe(velocity_table(snapshot).style.format("{:,.2f}"),
                         use_container_width=True)  # fmt: skip
            since = snapshot.get("time_since_last_txn", -1.0)
            st.caption("First transaction seen for this card" if since < 0
                       else f"Previous transaction {since / 60:,.1f} min earlier")  # fmt: skip
        else:
            st.caption("No velocity snapshot (scored on a feature miss).")
        st.markdown("**Top drivers** (SHAP on the XGBoost lane, log-odds)")
        drivers = alert.get("top_drivers") or []
        if drivers:
            st.dataframe(
                pd.DataFrame({
                    "feature": [d["feature"] for d in drivers],
                    "impact": [d["impact"] for d in drivers],
                    "pushes toward": ["fraud ↑" if d["impact"] > 0 else "legit ↓" for d in drivers],
                }),
                hide_index=True, use_container_width=True,
                column_config={"impact": st.column_config.NumberColumn(format="%+.3f")},
            )  # fmt: skip
        else:
            st.caption(
                "Explanation pending — computed asynchronously after the alert is published."
            )


@st.fragment(run_every=5)
def model_health() -> None:
    canary = get_json(f"{API_BASE}/canary")
    drift = get_json(f"{API_BASE}/drift")
    if canary is None:
        st.warning("Scorer API not reachable on :8000 — run `make api`.")
        return

    st.subheader("Champion vs challenger (canary)")
    cols = st.columns(4)
    cols[0].metric("Champion", f"v{canary['champion_version']}")
    cols[1].metric("Challenger", f"v{canary['challenger_version']}"
                   if canary.get("challenger_version") else "—")  # fmt: skip
    p95 = canary.get("p95_abs_delta")
    cols[2].metric("p95 |Δscore|", f"{p95:.3f}" if p95 is not None else "—",
                   help="Promotion gate: < 0.02 over ≥ 1,000 events")  # fmt: skip
    agreement = canary.get("alert_agreement")
    cols[3].metric("Alert agreement", f"{agreement:.1%}" if agreement is not None else "—")
    st.markdown(canary_verdict(canary))
    st.caption("Promote with `make promote`, revert with `make rollback`; "
               "the scorer swaps models within ~30 s, no restart needed.")  # fmt: skip

    st.subheader("Feature drift (calibrated PSI)")
    if not drift or drift.get("detail"):
        st.caption((drift or {}).get("detail", "Drift monitor unavailable."))
    else:
        window, rows = drift.get("window", 0), drift.get("rows", 0)
        if not drift.get("ratio"):
            st.progress(min(1.0, rows / window) if window else 0.0,
                        text=f"Filling the live window: {rows:,} / {window:,} events")  # fmt: skip
        else:
            flagged = drift.get("flagged", [])
            if flagged:
                st.error(f"Retrain flag: top-20 feature(s) beyond their calibrated threshold — "
                         f"{', '.join(flagged)}")  # fmt: skip
            else:
                st.success(f"No drift flag over the last {rows:,} events "
                           "(every top-20 feature within its training-calibrated range).")  # fmt: skip
            st.dataframe(
                drift_table(drift), hide_index=True, use_container_width=True,
                column_config={
                    "ratio": st.column_config.ProgressColumn(
                        "ratio (≥ 1 = drift)", min_value=0.0, max_value=1.5, format="%.2f"),
                    "PSI": st.column_config.NumberColumn(format="%.3f"),
                    "present-only PSI": st.column_config.NumberColumn(format="%.3f"),
                },
            )  # fmt: skip

    st.subheader("Alert precision by band")
    alerts = list(st.session_state.alerts.values())
    if alerts:
        st.dataframe(
            precision_by_band(alerts), hide_index=True, use_container_width=True,
            column_config={"precision": st.column_config.ProgressColumn(
                "precision (labeled fraud)", min_value=0.0, max_value=1.0, format="%.2f")},
        )  # fmt: skip
        st.caption(f"Over the {len(alerts):,} most recent alerts on this page.")


with st.sidebar:
    st.header("Filters")
    st.multiselect("Risk bands", BANDS, default=BANDS, key="band_filter")
    st.radio("Ground truth", LABELS, horizontal=True, key="label_filter")
    st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05, key="min_score")
    st.text_input("Card or transaction id", key="card_query", placeholder="e.g. 3e2947…")
    st.divider()
    st.caption(f"Polling {API_BASE} every 2 s · Prometheus {PROMETHEUS}")

header()
live_tab, health_tab = st.tabs(["🚨 Live alerts", "🩺 Model health"])
with live_tab:
    feed()
    st.divider()
    drill_down()
with health_tab:
    model_health()

"""StreamGuard live alert dashboard (PLAN.md §2) — `make dashboard`, expects `make api` on :8000.

Layout: header metrics (scored, tx/s, alerts, live precision, p95 latency)
→ alert feed (incremental poll of /alerts every 2 s, band filter)
→ drill-down for one alert (transaction, velocity snapshot by window, SHAP top drivers).
Labels are replayed with the stream, so alert precision is measured live.
"""

from __future__ import annotations

import time

import httpx
import pandas as pd
import streamlit as st

API_BASE = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
MAX_ALERTS = 1000
BANDS = ["high", "medium", "review"]
BAND_ICON = {"high": "🔴", "medium": "🟠", "review": "🟡"}
WINDOWS = ["5m", "1h", "24h", "7d"]

st.set_page_config(page_title="StreamGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ StreamGuard — live fraud alerts")

if "alerts" not in st.session_state:
    st.session_state.alerts = {}  # transaction_id → alert dict
    st.session_state.since = None
    st.session_state.last_scored = (None, None)  # (count, monotonic time) for tx/s


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


def tx_rate(scored: int) -> float | None:
    prev_count, prev_t = st.session_state.last_scored
    now = time.monotonic()
    st.session_state.last_scored = (scored, now)
    if prev_count is None or now <= prev_t:
        return None
    return max(0.0, (scored - prev_count) / (now - prev_t))


@st.fragment(run_every=2)
def live_panel() -> None:
    poll_alerts()
    stats = get_json(f"{API_BASE}/stats")
    if stats is None:
        st.warning("Scorer API not reachable on :8000 — run `make api`.")
    alerts = list(st.session_state.alerts.values())
    by_band = (stats or {}).get("by_band", {})
    rate = tx_rate(stats["scored"]) if stats else None
    p95 = prometheus_scalar(
        "histogram_quantile(0.95, sum(rate(streamguard_e2e_latency_seconds_bucket[1m])) by (le))"
    )
    labeled = [a for a in alerts if a["risk_band"] in ("high", "medium")]
    precision = sum(a["is_fraud"] for a in labeled) / len(labeled) if labeled else None

    cols = [*st.columns(3), *st.columns(3)]  # two rows: readable on narrow screens
    cols[0].metric("Transactions scored", f"{stats['scored']:,}" if stats else "—")
    cols[1].metric("Throughput", f"{rate:.0f} tx/s" if rate is not None else "—")
    cols[2].metric("Alerts", f"{sum(by_band.values()):,}" if stats else "—",
                   help=" · ".join(f"{b}: {by_band.get(b, 0)}" for b in BANDS))  # fmt: skip
    cols[3].metric("🔴 high / 🟠 medium", f"{by_band.get('high', 0)} / {by_band.get('medium', 0)}")
    cols[4].metric("Live alert precision", f"{precision:.0%}" if precision is not None else "—",
                   help="Share of high+medium alerts (on this page) that are labeled fraud")  # fmt: skip
    cols[5].metric("p95 latency (1m)", f"{p95:.2f} s" if p95 is not None else "—",
                   help="Producer send → alert published, from Prometheus")  # fmt: skip
    if stats and stats.get("model_version"):
        st.caption(f"Champion model v{stats['model_version']} · "
                   f"{stats.get('feature_misses', 0)} events scored without a velocity snapshot")  # fmt: skip

    chosen = st.session_state.get("band_filter", BANDS)
    rows = sorted((a for a in alerts if a["risk_band"] in chosen),
                  key=lambda a: a["scored_at"], reverse=True)  # fmt: skip
    if not rows:
        st.info("Waiting for alerts… `make features`, `make api`, then `make replay`.")
        return
    table = pd.DataFrame({
        "band": [f"{BAND_ICON[a['risk_band']]} {a['risk_band']}" for a in rows],
        "score": [a["score"] for a in rows],
        "transaction": [a["transaction_id"] for a in rows],
        "event time": [a["event_ts"] for a in rows],
        "card": [a["card_key"] for a in rows],
        "amount": [a["transaction_amt"] for a in rows],
        "product": [a["product_cd"] for a in rows],
        "label": ["fraud" if a["is_fraud"] else "legit" for a in rows],
    })  # fmt: skip
    st.dataframe(
        table, hide_index=True, height=420, use_container_width=True,
        column_config={
            "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0,
                                                     format="%.3f"),
            "amount": st.column_config.NumberColumn("amount", format="$%.2f"),
            "transaction": st.column_config.NumberColumn("transaction", format="%d"),
        },
    )  # fmt: skip


def velocity_table(snapshot: dict[str, float]) -> pd.DataFrame:
    """Snapshot fields `<metric>_<window>` → rows per window."""
    return pd.DataFrame(
        {
            "transactions": [snapshot.get(f"txn_count_{w}") for w in WINDOWS],
            "amount sum": [snapshot.get(f"amt_sum_{w}") for w in WINDOWS],
            "amount mean": [snapshot.get(f"amt_mean_{w}") for w in WINDOWS],
        },
        index=pd.Index(WINDOWS, name="window"),
    )


@st.fragment(run_every=2)
def drill_down() -> None:
    st.subheader("Alert drill-down")
    alerts = sorted(st.session_state.alerts.values(), key=lambda a: a["scored_at"], reverse=True)
    if not alerts:
        st.caption("Select an alert once some arrive.")
        return
    by_id = {a["transaction_id"]: a for a in alerts}
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


with st.sidebar:
    st.multiselect("Risk bands", BANDS, default=BANDS, key="band_filter")
    st.caption(f"Polling {API_BASE} every 2 s · Prometheus {PROMETHEUS}")

live_panel()
st.divider()
drill_down()

"""StreamGuard live dashboard (PLAN.md §2) — `make dashboard`, expects `make api` on :8000.

Design: dark security-ops theme (.streamlit/config.toml + CSS below). Layout:
    header (every 2 s)  — LIVE/IDLE pill + six KPI cards
    🚨 Live alerts      — filtered feed (styled) + drill-down (verdict banner, velocity, SHAP chart)
    🩺 Model health     — canary verdict, calibrated drift, precision by band
Labels are replayed with the stream, so alert precision is measured live.
Data shaping lives in views.py (unit-tested); this file is layout + polling only.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pandas as pd
import streamlit as st

from views import (
    BAD_HEX,
    BAND_HEX,
    BANDS,
    LABELS,
    NEUTRAL_HEX,
    OK_HEX,
    alerts_table,
    canary_verdict,
    drift_table,
    drivers_frame,
    filter_alerts,
    precision_by_band,
    velocity_table,
)

API_BASE = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
MAX_ALERTS = 1000

st.set_page_config(page_title="StreamGuard", page_icon="🛡️", layout="wide")

CSS = f"""
<style>
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.1rem; padding-bottom: 1.4rem; max-width: 1600px; }}

    /* brand line */
    .sg-title {{ display: flex; align-items: baseline; gap: .6rem; margin-bottom: 0; }}
    .sg-title h1 {{ font-size: 1.45rem; font-weight: 800; letter-spacing: -.01em; margin: 0; }}
    .sg-sub {{ color: #8b94ab; font-size: .8rem; }}

    /* status pills */
    .sg-pill {{ display: inline-block; border-radius: 999px; padding: .1rem .7rem;
                font-size: .74rem; font-weight: 700; letter-spacing: .06em;
                border: 1px solid; vertical-align: middle; }}
    .sg-live {{ color: {OK_HEX}; border-color: {OK_HEX}66; background: {OK_HEX}18; }}
    .sg-idle {{ color: #8b94ab; border-color: #3a4460; background: #8b94ab14; }}

    /* KPI cards */
    [data-testid="stMetric"] {{ background: #121829; border: 1px solid #232c44;
                               border-radius: 12px; padding: 13px 18px 9px; height: 100%; }}
    [data-testid="stMetricLabel"] p {{ color: #8b94ab !important; font-size: .72rem;
                                       letter-spacing: .05em; text-transform: uppercase; }}
    [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; font-weight: 700;
                                     font-size: 1.5rem; }}
    .stColumns > div:nth-child(3) [data-testid="stMetric"],
    .stColumns > div:nth-child(4) [data-testid="stMetric"] {{ border-left: 3px solid {BAD_HEX}; }}
    .stColumns > div:nth-child(5) [data-testid="stMetric"] {{ border-left: 3px solid {OK_HEX}; }}
    .stColumns > div:nth-child(1) [data-testid="stMetric"],
    .stColumns > div:nth-child(2) [data-testid="stMetric"],
    .stColumns > div:nth-child(6) [data-testid="stMetric"] {{ border-left: 3px solid {NEUTRAL_HEX}; }}

    /* verdict / status banners */
    .sg-banner {{ border: 1px solid #232c44; border-left: 3px solid {NEUTRAL_HEX};
                  background: #121829; border-radius: 10px; padding: .65rem 1rem;
                  font-size: .9rem; }}
    .sg-banner b {{ letter-spacing: .02em; }}

    /* sidebar */
    [data-testid="stSidebar"] {{ background: #0d1120; border-right: 1px solid #232c44; }}
    [data-testid="stSidebar"] h2 {{ font-size: .8rem; letter-spacing: .1em;
                                    text-transform: uppercase; color: #8b94ab; }}

    /* dataframe cell emphasis set via Styler */
    #feed-table {{ border: 1px solid #232c44; border-radius: 10px; overflow: hidden; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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


def pill(text: str, color: str) -> str:
    return (f'<span class="sg-pill" style="color: {color}; border-color: {color}66; '
            f'background: {color}18">{text}</span>')  # fmt: skip


def banner(html: str, color: str) -> None:
    st.markdown(f'<div class="sg-banner" style="border-left-color: {color}">{html}</div>',
                unsafe_allow_html=True)  # fmt: skip


@st.fragment(run_every=2)
def header() -> None:
    poll_alerts()
    stats = get_json(f"{API_BASE}/stats")
    by_band = (stats or {}).get("by_band", {})
    # 1-minute rate from Prometheus: the scorer works in bursts, a 2 s delta jumps 0 ↔ 200
    rate = prometheus_scalar("sum(rate(streamguard_scored_total[1m]))")
    p95 = prometheus_scalar(
        "histogram_quantile(0.95, sum(rate(streamguard_e2e_latency_seconds_bucket[1m])) by (le))"
    )
    alerts = list(st.session_state.alerts.values())
    labeled = [a for a in alerts if a["risk_band"] in ("high", "medium")]
    precision = sum(a["is_fraud"] for a in labeled) / len(labeled) if labeled else None

    if alerts:
        newest = max(a["scored_at"] for a in alerts)
        age = (datetime.now(UTC) - datetime.fromisoformat(newest)).total_seconds()
        status = pill("● LIVE", OK_HEX) if age < 10 else pill(f"IDLE {age / 60:.0f}m", "#8b94ab")
    else:
        status = pill("STARTING", "#8b94ab")
    version = stats.get("model_version") if stats else None
    chips = status
    if version:
        chips += f" &nbsp; {pill(f'champion v{version}', NEUTRAL_HEX)}"
    st.markdown(
        '<div class="sg-title"><h1>🛡️ StreamGuard</h1>'
        '<span class="sg-sub">real-time fraud detection · Kafka → Spark → Redis → XGBoost</span>'
        f'</div><div style="margin: .15rem 0 .8rem">{chips}</div>',
        unsafe_allow_html=True,  # fmt: skip
    )

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


def verdict_banner(alert: dict) -> None:
    band = alert["risk_band"]
    color = BAND_HEX[band]
    truth = f'<b style="color: {BAD_HEX}">FRAUD</b>' if alert["is_fraud"] else "legit"
    banner(f'{pill(band.upper(), color)} &nbsp; score <b>{alert["score"]:.3f}</b> '
           f'≥ floor {alert["threshold"]:.3f} &nbsp;·&nbsp; ground truth: {truth} '
           f'&nbsp;·&nbsp; model v{alert["model_version"]}', color)  # fmt: skip
    st.progress(min(1.0, alert["score"]), text=f"score {alert['score']:.3f} "
                f"(band floor {alert['threshold']:.3f})")  # fmt: skip


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
        format_func=lambda i: f"{by_id[i]['risk_band'].upper()} · tx {i} · "
                              f"score {by_id[i]['score']:.3f} · ${by_id[i]['transaction_amt']:,.2f}",
    )  # fmt: skip
    # re-fetch: SHAP drivers are attached asynchronously after the alert first arrives
    alert = get_json(f"{API_BASE}/alerts/{tx_id}") or by_id[tx_id]
    st.session_state.alerts[tx_id] = alert
    verdict_banner(alert)
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Transaction**")
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
        st.markdown("**Top drivers** — SHAP log-odds impact (right = toward fraud)")
        drivers = alert.get("top_drivers") or []
        if drivers:
            st.bar_chart(drivers_frame(drivers), horizontal=True, color=BAD_HEX, height=230)
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
    verdict = canary_verdict(canary)
    if verdict["state"] == "pass":
        banner(f"{pill('PROMOTABLE', OK_HEX)} &nbsp; {verdict['detail']}", OK_HEX)  # fmt: skip
    elif verdict["state"] == "fail":
        banner(f"{pill('PROMOTION BLOCKED', BAD_HEX)} &nbsp; {verdict['detail']}", BAD_HEX)
    else:
        banner(verdict["detail"], NEUTRAL_HEX)
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
                banner(f"{pill('RETRAIN FLAG', BAD_HEX)} top-20 feature(s) beyond their "
                       f"calibrated threshold — <b>{', '.join(flagged)}</b>", BAD_HEX)  # fmt: skip
            else:
                banner(
                    f"{pill('NOMINAL', OK_HEX)} no drift flag over the last {rows:,} events "
                    "(every top-20 feature within its training-calibrated range).",
                    OK_HEX,
                )
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
    st.markdown("## Filters")
    st.multiselect("Risk bands", BANDS, default=BANDS, key="band_filter")
    st.radio("Ground truth", LABELS, horizontal=True, key="label_filter")
    st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05, key="min_score")
    st.text_input("Card or transaction id", key="card_query", placeholder="e.g. 3e2947…")
    st.divider()
    st.markdown("## System")
    stats = get_json(f"{API_BASE}/stats")
    st.caption(
        f"Champion **v{stats['model_version']}** serving · API {API_BASE} · Prometheus {PROMETHEUS}"
        if stats else f"Scorer API on {API_BASE} is down — run `make api`."
    )  # fmt: skip
    st.caption("Ops: `make promote` / `make rollback` / `make bench` / `make drift`")

header()
live_tab, health_tab = st.tabs(["🚨 Live alerts", "🩺 Model health"])
with live_tab:
    feed()
    st.divider()
    drill_down()
with health_tab:
    model_health()

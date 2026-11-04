"""StreamGuard live alert dashboard — milestone M4 (PLAN.md §2).

Run: `make dashboard` (expects `make api` on :8000).
Layout: header metrics row (tx/s, alerts, p95) → alert feed table (poll 2 s)
→ selected-alert drill-down (velocity snapshot + SHAP top drivers, M5).
"""

from __future__ import annotations

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="StreamGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ StreamGuard — live fraud alerts")

header = st.container()
feed = st.container()
detail = st.container()


def fetch_alerts() -> list[dict]:
    """Poll the scorer API; degrade gracefully when it is not up yet."""
    try:
        response = httpx.get(f"{API_BASE}/alerts", timeout=1.0)
        response.raise_for_status()
        return response.json().get("alerts", [])
    except (httpx.HTTPError, ValueError):
        return []


@st.fragment(run_every=2)
def alerts_panel() -> None:
    alerts = fetch_alerts()
    with header:
        cols = st.columns(3)
        cols[0].metric("Transactions scored", "—", help="M3 wires the counter")
        cols[1].metric("Open alerts", str(len(alerts)))
        cols[2].metric("p95 latency", "—", help="make bench (M6) fills this")
    with feed:
        if alerts:
            st.dataframe(alerts, use_container_width=True)
        else:
            st.info("Waiting for alerts… start `make replay` + `make api` (milestones M0–M3).")


alerts_panel()

with detail:
    st.subheader("Alert drill-down")
    st.caption("Velocity snapshot + SHAP top drivers land here in M4/M5.")

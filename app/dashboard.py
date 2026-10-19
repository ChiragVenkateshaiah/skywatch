"""SkyWatch — KATL Arrival Manager: the dashboard page (app.py routes to it and to chat.py).

One screen of what matters *now*; everything else is a question for Chat with SkyWatch
(decided 2026-10-19 — the full arrival sequence, click-to-predict, surge detail and Model 3
history moved there):
  * header: how fresh the data is, and Refresh
  * four numbers: inbound now, predicted next 3 h vs capacity, peak 15-min headroom, alerts
  * the ops briefing (LLM-written, grounding-checked — src/brief_ops.py)
  * one map — inbound aircraft by predicted ETA, flagged aircraft (Model 3) as rings — and one
    chart — Model 2's next 3 h against the AAR
  * "Ask SkyWatch": questions that open the chat already asked

Panels read Delta tables written by the batch scoring jobs (no live endpoint on Free Edition;
freshness = the scoring cadence, or the last pre-interview refresh).
"""

from __future__ import annotations

import datetime as dt
import os

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

import data

APT_ICAO = os.environ.get("SKYWATCH_APT_ICAO", "KATL")
APT_LAT = float(os.environ.get("SKYWATCH_APT_LAT", "33.6407"))
APT_LON = float(os.environ.get("SKYWATCH_APT_LON", "-84.4277"))
AAR_PER_HOUR = float(os.environ.get("SKYWATCH_AAR_PER_HOUR", "90"))
AAR_PER_BIN = AAR_PER_HOUR / 4.0

# ETA bands for the map: (upper-bound minutes, label, [r,g,b])
ETA_BANDS = [(10, "≤ 10 min", [46, 160, 67]), (25, "10–25 min", [219, 149, 40]),
             (999, "> 25 min", [120, 120, 200])]
KIND_RING = {"emergency": [214, 39, 40], "go_around": [239, 85, 59], "holding": [255, 140, 0]}
KIND_ICON = {"emergency": "🚨", "go_around": "🔴", "holding": "🟠"}
# a flag older than this before the current Model 1 run isn't current — score_irregularities
# writes no rows on a quiet run (same rule as lib/briefing.build_facts and the chat tool)
FLAGS_WINDOW = pd.Timedelta(minutes=15)


def eta_colour(mins: float) -> list:
    for hi, _, rgb in ETA_BANDS:
        if mins <= hi:
            return rgb
    return [120, 120, 120]


def _span(mins: float) -> str:
    """'45 min' / '6.5 h' / '34 days' for a duration in minutes."""
    if mins < 90:
        return f"{mins:.0f} min"
    if mins < 48 * 60:
        return f"{mins / 60:.1f} h"
    return f"{mins / 1440:.0f} days"


def _ago(ts, now) -> str:
    """Human 'N min ago' for a UTC timestamp, tolerant of NaT."""
    if ts is None or pd.isna(ts):
        return "—"
    mins = (now - ts.to_pydatetime()).total_seconds() / 60.0
    return "just now" if mins < 1 else f"{_span(mins)} ago"


@st.cache_data(ttl=60)
def load():
    return data.latest_predictions(), data.latest_demand_forecast()


@st.cache_data(ttl=60)
def load_flags():
    # Model 3 is optional — the table may not exist until score_irregularities has run once.
    try:
        return data.latest_irregularity_flags(APT_ICAO)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_briefing():
    # optional — ops_briefings exists only after skywatch_brief_ops has run once.
    try:
        return data.latest_briefing(APT_ICAO)
    except Exception:
        return pd.DataFrame()


def ask(question: str) -> None:
    """Open Chat with SkyWatch with `question` already asked (chat.py picks it up)."""
    st.session_state["pending_question"] = question
    st.switch_page("chat.py")


try:
    preds, fc = load()
except Exception as exc:  # surface config / grant problems instead of a blank page
    st.error(f"Could not load data from the warehouse: {exc}")
    st.stop()

now = dt.datetime.now(dt.timezone.utc)
pred_as_of = preds["scored_at"].max() if not preds.empty else None
flags = load_flags()
if not flags.empty and pred_as_of is not None and flags["scored_at"].max() < pred_as_of - FLAGS_WINDOW:
    flags = flags.iloc[0:0]


# ---------------------------------------------------------------------------- header
left, mid, right = st.columns([0.52, 0.33, 0.15])
left.title(f"✈️ {APT_ICAO} Arrival Manager")
with mid:
    st.write("")
    if pred_as_of is None or pd.isna(pred_as_of):
        st.markdown("🔴 **No live data yet**")
    else:
        age_min = (now - pred_as_of.to_pydatetime()).total_seconds() / 60
        dot = "🟢" if age_min <= 20 else ("🟡" if age_min <= 120 else "🔴")
        st.markdown(f"{dot} **Data as of {pred_as_of:%H:%M} UTC** · {_span(age_min)} old")
        if age_min > 120:
            st.caption("Not a picture of now — refresh it from the chat page sidebar.")
with right:
    st.write("")
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ------------------------------------------------------------------------------ KPIs
next3h = float(fc["predicted_mean"].sum()) if not fc.empty else float("nan")
capacity_3h = AAR_PER_HOUR * len(fc) / 4 if not fc.empty else float("nan")
peak_i = int(fc["predicted_mean"].values.argmax()) if not fc.empty else None
headroom = AAR_PER_BIN - float(fc["predicted_mean"].iloc[peak_i]) if peak_i is not None else float("nan")
n_kind = flags["kind"].value_counts().to_dict() if not flags.empty else {}

k1, k2, k3, k4 = st.columns(4)
k1.metric("Inbound now", "—" if preds.empty else f"{len(preds)}")
k2.metric("Predicted next 3 h", "—" if pd.isna(next3h) else f"{next3h:.0f}",
          delta=None if pd.isna(capacity_3h) else f"capacity {capacity_3h:.0f}", delta_color="off",
          help="Model 2's predicted arrivals over the next 3 hours, against the AAR over the same time.")
k3.metric("Peak 15-min headroom", "—" if pd.isna(headroom) else f"{headroom:+.1f}",
          delta=None if peak_i is None else f"at {fc['bin_start_ts'].iloc[peak_i]:%H:%M} UTC", delta_color="off",
          help=f"AAR per 15 min ({AAR_PER_BIN:.1f}) minus the busiest predicted bin. Negative = over capacity.")
k4.metric("Alerts", str(sum(n_kind.values())),
          delta=" · ".join(f"{n} {k.replace('_', '-')}" for k, n in n_kind.items()) or "none", delta_color="off",
          help="Model 3 rules: emergency squawk, go-around, holding pattern — in the current Model 1 picture.")

if not flags.empty:
    st.markdown("**Alerts:** " + " · ".join(
        f"{KIND_ICON.get(r['kind'], '⚪')} {r['callsign'] or r['icao']} {r['kind'].replace('_', '-')}, "
        f"{r['dist_to_apt_nm']:.0f} nm" for _, r in flags.iterrows()))


# ----------------------------------------------------------------------- ops briefing
brief = load_briefing()
with st.container(border=True):
    st.markdown("**Ops briefing**")
    if brief.empty:
        st.caption("No ops briefing yet — it is written by the refresh (or `skywatch_brief_ops`).")
    else:
        b = brief.iloc[0]
        st.markdown(b["briefing"])
        who = "written by an LLM, checked against the data" if b["source"] == "llm" else "template (the LLM text failed its check)"
        st.caption(f"{who} · {_ago(b['generated_at'], now)}")


# ---------------------------------------------------------------------- map + demand
col_map, col_chart = st.columns(2)

with col_map:
    st.markdown("**Inbound aircraft**")
    mappable = preds.dropna(subset=["lat", "lon"]) if not preds.empty else preds
    if mappable.empty:
        st.info("No inbound aircraft with a position in the latest Model 1 run.")
    else:
        md = mappable.copy()
        md["colour"] = md["predicted_eta_min"].apply(eta_colour)
        md["tip"] = (md["callsign"].fillna(md["icao"]) + " · " + md["ac_type"].fillna("?")
                     + " · ETA " + md["predicted_eta_min"].round(0).astype(int).astype(str) + " min")
        layers = [pdk.Layer("ScatterplotLayer", data=md, get_position="[lon, lat]", get_fill_color="colour",
                            get_radius=2200, pickable=True, opacity=0.85)]
        fm = flags.dropna(subset=["lat", "lon"]) if not flags.empty else flags
        if not fm.empty:
            fm = fm.copy()
            fm["ring"] = fm["kind"].map(lambda k: KIND_RING.get(k, [120, 120, 120]))
            fm["tip"] = fm["callsign"].fillna(fm["icao"]) + " · " + fm["kind"].str.replace("_", "-")
            layers.append(pdk.Layer("ScatterplotLayer", data=fm, get_position="[lon, lat]", filled=False,
                                    stroked=True, get_line_color="ring", line_width_min_pixels=3,
                                    get_radius=6000, pickable=True))
        layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame([{"lon": APT_LON, "lat": APT_LAT}]),
                                get_position="[lon, lat]", get_fill_color="[30,30,30]", get_radius=3000))
        st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, tooltip={"text": "{tip}"},
                                 initial_view_state=pdk.ViewState(latitude=APT_LAT, longitude=APT_LON, zoom=7)))
        st.caption("Dot colour = predicted ETA (" + " · ".join(lbl for _, lbl, _ in ETA_BANDS)
                   + ") · rings = alerts · black dot = " + APT_ICAO)

with col_chart:
    st.markdown("**Arrival demand — next 3 hours**")
    if fc.empty:
        st.info("No Model 2 forecast yet.")
    else:
        base = alt.Chart(fc).encode(x=alt.X("bin_start_ts:T", title="UTC"))
        band = base.mark_area(opacity=0.2, color="#4c78a8").encode(
            y=alt.Y("predicted_q10:Q", title="arrivals per 15 min"), y2="predicted_q90:Q")
        line = base.mark_line(point=True, color="#4c78a8").encode(y="predicted_mean:Q")
        aar = alt.Chart(pd.DataFrame({"y": [AAR_PER_BIN]})).mark_rule(color="#d62728", strokeDash=[6, 4]).encode(y="y:Q")
        st.altair_chart((band + line + aar).properties(height=330), use_container_width=True)
        st.caption("Line = predicted · band = likely range · red dashed = capacity (AAR per 15 min)")


# --------------------------------------------------------------------- ask SkyWatch
st.markdown("**💬 Ask SkyWatch** — everything else is a question away")
# the next aircraft to land is always in Model 1's set, so explain_eta can answer for it
subject = preds.sort_values("predicted_eta_min").iloc[0]["callsign"] if not preds.empty else None
questions = ["Which aircraft land next, and when?",
             f"Why does the model predict that {subject} lands when it does?" if subject
             else "How does Model 1 predict landing times?",
             "Will arrivals exceed capacity in the next three hours?"]
for col, q in zip(st.columns(3), questions):
    if col.button(q, use_container_width=True):
        ask(q)

st.caption(f"Data: adsb.lol (ODbL) · {APT_ICAO} · Model 1 `eta_touchdown@champion`, Model 2 "
           f"`demand_forecast@champion`, Model 3 rules · all times UTC")

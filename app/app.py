"""SkyWatch — KATL Arrival Manager (Databricks App, Streamlit).

Serves the two models to an arrival coordinator:
  * Model 1 (time-to-touchdown)  -> the predicted arrival sequence
  * Model 2 (15-min demand forecast) -> the next 3 h vs the Airport Acceptance Rate

Both read Delta tables written by the batch scoring jobs; there is no live endpoint
on Free Edition. Data freshness = the scoring-job cadence (~10 min).
"""

from __future__ import annotations

import datetime as dt
import os

import altair as alt
import pandas as pd
import streamlit as st

import data

APT_ICAO = os.environ.get("SKYWATCH_APT_ICAO", "KATL")
AAR_PER_HOUR = float(os.environ.get("SKYWATCH_AAR_PER_HOUR", "90"))
AAR_PER_BIN = AAR_PER_HOUR / 4.0
BIN_MINUTES = 15

st.set_page_config(page_title=f"SkyWatch — {APT_ICAO} Arrival Manager", layout="wide")


def _ago(ts, now) -> str:
    """Human 'N min ago' for a UTC timestamp, tolerant of NaT."""
    if ts is None or pd.isna(ts):
        return "—"
    mins = (now - ts.to_pydatetime()).total_seconds() / 60.0
    if mins < 1:
        return "just now"
    if mins < 90:
        return f"{mins:.0f} min ago"
    return f"{mins / 60:.1f} h ago"


@st.cache_data(ttl=60)
def load():
    return (
        data.latest_predictions(),
        data.latest_demand_forecast(),
        data.historical_hourly_profile(APT_ICAO),
    )


@st.cache_data(ttl=60)
def load_flags():
    # Model 3 is optional — the table may not exist until score_irregularities has run once.
    try:
        return data.latest_irregularity_flags(APT_ICAO)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- header
left, right = st.columns([0.75, 0.25])
with left:
    st.title(f"✈️  {APT_ICAO} Arrival Manager")
with right:
    st.write("")
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    preds, fc, hist = load()
except Exception as exc:  # surface config / grant problems instead of a blank page
    st.error(f"Could not load data from the warehouse: {exc}")
    st.stop()

now = dt.datetime.now(dt.timezone.utc)
pred_as_of = preds["scored_at"].max() if not preds.empty else None
fc_as_of = fc["scored_at"].max() if not fc.empty else None
st.caption(
    f"Model 1 scored {_ago(pred_as_of, now)} · Model 2 scored {_ago(fc_as_of, now)} · "
    f"AAR {AAR_PER_HOUR:.0f}/h ({AAR_PER_BIN:.1f} per 15 min) · all times UTC"
)


# ------------------------------------------------------------------------------ KPIs
arrivals_so_far = float(fc["arrivals_so_far"].iloc[0]) if not fc.empty else float("nan")
next3h = float(fc["predicted_mean"].sum()) if not fc.empty else float("nan")
peak_bin = float(fc["predicted_mean"].max()) if not fc.empty else float("nan")
headroom = AAR_PER_BIN - peak_bin

k1, k2, k3, k4 = st.columns(4)
k1.metric("Inbound now", "—" if preds.empty else f"{len(preds)}")
k2.metric("Arrivals so far today", "—" if pd.isna(arrivals_so_far) else f"{arrivals_so_far:.0f}")
k3.metric("Predicted next 3 h", "—" if pd.isna(next3h) else f"{next3h:.0f}")
k4.metric(
    "Peak-bin AAR headroom",
    "—" if pd.isna(headroom) else f"{headroom:+.1f}",
    help="AAR per 15-min bin minus the busiest predicted bin. Negative = predicted demand exceeds capacity.",
)

st.divider()


# -------------------------------------------------------------------- demand forecast
col_curve, col_surge = st.columns([0.62, 0.38])

with col_curve:
    st.subheader("Predicted arrival demand — next 3 hours")
    if fc.empty:
        st.info("No Model 2 forecast yet. Run `skywatch_score_demand`.")
    else:
        chart_df = fc.copy()
        hist_map = dict(zip(hist["hour_utc"], hist["avg_arrivals_per_bin"])) if not hist.empty else {}
        chart_df["typical"] = chart_df["bin_start_ts"].dt.hour.map(hist_map)

        base = alt.Chart(chart_df).encode(
            x=alt.X("bin_start_ts:T", title="bin start (UTC)")
        )
        band = base.mark_area(opacity=0.20, color="#4c78a8").encode(
            y=alt.Y("predicted_q10:Q", title="arrivals per 15-min bin"),
            y2="predicted_q90:Q",
        )
        mean_line = base.mark_line(point=True, color="#4c78a8").encode(y="predicted_mean:Q")
        typical_line = base.mark_line(strokeDash=[4, 3], color="#888").encode(y="typical:Q")
        aar_rule = (
            alt.Chart(pd.DataFrame({"y": [AAR_PER_BIN]}))
            .mark_rule(color="#d62728", strokeDash=[6, 4])
            .encode(y="y:Q")
        )
        st.altair_chart(
            (band + typical_line + mean_line + aar_rule).properties(height=340),
            use_container_width=True,
        )
        st.caption(
            "Shaded = q10–q90 interval · solid = mean · grey dashed = typical for this hour "
            "(all collected days) · red dashed = AAR per bin"
        )

with col_surge:
    st.subheader("Surge alerts")
    if fc.empty:
        st.info("—")
    else:
        surge = fc[fc["predicted_q50"] > AAR_PER_BIN].copy()
        if surge.empty:
            st.success(f"No bin exceeds the AAR ({AAR_PER_BIN:.1f}/15 min) in the next 3 h.")
        else:
            surge["lead_min"] = (
                (surge["bin_start_ts"] - now).dt.total_seconds() / 60.0
            ).round().astype(int)
            surge["excess"] = (surge["predicted_q50"] - AAR_PER_BIN).round(1)
            for _, r in surge.iterrows():
                when = r["bin_start_ts"].strftime("%H:%M")
                lead = r["lead_min"]
                if lead >= 30:
                    action = f"Meter now — shed ~{r['excess']:.0f} arrivals from this bin"
                elif lead >= 0:
                    action = "Imminent — plan holding / vectoring"
                else:
                    action = "In progress"
                st.warning(
                    f"**{when}Z**  ·  q50 {r['predicted_q50']:.1f} vs AAR {AAR_PER_BIN:.1f}  "
                    f"·  lead {lead} min\n\n{action}"
                )

st.divider()


# -------------------------------------------------------------------- arrival sequence
st.subheader("Predicted arrival sequence")
if preds.empty:
    st.info("No inbound aircraft in the latest Model 1 run.")
else:
    seq = preds.copy()
    seq["ETA (min)"] = seq["predicted_eta_min"].round(1)
    seq["touchdown (UTC)"] = seq["predicted_touchdown_ts"].dt.strftime("%H:%M:%S")
    seq["dist (nm)"] = seq["dist_to_apt_nm"].round(1)
    seq["gs (kt)"] = seq["gs_kt"].round(0)
    seq["alt (ft)"] = seq["alt_ft"].round(0)
    seq = seq.rename(columns={"callsign": "callsign", "ac_type": "type"})
    st.dataframe(
        seq[["callsign", "type", "dist (nm)", "gs (kt)", "alt (ft)", "ETA (min)", "touchdown (UTC)"]],
        hide_index=True,
        use_container_width=True,
        height=min(560, 44 + 35 * len(seq)),
    )

st.divider()


# ---------------------------------------------------------------------- irregularities
st.subheader("Irregularity flags — Model 3 (rules)")
flags = load_flags()
_ICON = {"emergency": "🚨", "go_around": "🔴", "holding": "🟠"}
if flags.empty:
    st.info("No aircraft flagged in the latest run. (Rule-based: emergency squawk, go-around, or "
            "racetrack geometry — no learned model, see the roadmap for why.)")
else:
    for _, r in flags.iterrows():
        icon = _ICON.get(r["kind"], "⚪")
        line = (f"{icon}  **{r['callsign'] or r['icao']}** ({r['ac_type'] or '?'}) — "
                f"**{r['kind'].replace('_', ' ')}** · {r['dist_to_apt_nm']:.0f} nm, "
                f"{r['alt_ft']:.0f} ft · {r['detail']}")
        (st.error if r["severity"] >= 3 else st.warning)(line)

st.caption(
    f"Data: adsb.lol (ODbL) · {APT_ICAO} · Model 1 `eta_touchdown@champion`, "
    f"Model 2 `demand_forecast@champion`, Model 3 rule-based · batch-scored to Delta "
    f"(no live endpoint on Free Edition)"
)

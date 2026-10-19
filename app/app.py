"""SkyWatch — KATL Arrival Manager (Databricks App, Streamlit): the entry point.

Two pages: the dashboard (dashboard.py — Models 1-3, the ops briefing) and Chat with SkyWatch
(chat.py, docs/CHAT_PLAN.md). The Apps runtime starts this file (app.yaml).
"""

from __future__ import annotations

import os

import streamlit as st

APT_ICAO = os.environ.get("SKYWATCH_APT_ICAO", "KATL")

st.set_page_config(page_title=f"SkyWatch — {APT_ICAO} Arrival Manager", layout="wide")

st.navigation([
    st.Page("dashboard.py", title=f"{APT_ICAO} Arrival Manager", icon="✈️", default=True),
    st.Page("chat.py", title="Chat with SkyWatch", icon="💬"),
]).run()

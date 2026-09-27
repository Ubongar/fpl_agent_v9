"""Legacy Streamlit companion for FPL Agent v9.

The primary product UI is the Next.js app in dashboard/. This file intentionally
keeps a lightweight Streamlit path for local diagnostics and API smoke testing.
"""
from __future__ import annotations
import os
import requests
import streamlit as st

API = os.getenv("FPL_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="FPL Agent v9", layout="wide")
st.title("FPL Agent v9 — Legacy Companion")
st.caption("The production dashboard lives in dashboard/. This page is a lightweight diagnostics companion.")

try:
    health = requests.get(f"{API}/health", timeout=5).json()
    st.success(f"API online: {health}")
except Exception as exc:
    st.error(f"Backend unavailable at {API}: {exc}")
    st.stop()

st.header("Player snapshot")
try:
    data = requests.get(f"{API}/predictions", params={"limit": 20, "sort": "xp_mean", "desc": "true"}, timeout=10).json()
    players = data.get("players", [])
    if players:
        cols = [c for c in ["web_name","position","now_cost","xp_mean","xp_p10","xp_p50","xp_p90","p_play","p_start","p_sixty","ownership"] if c in players[0]]
        st.dataframe([{c: p.get(c) for c in cols} for p in players], use_container_width=True)
    else:
        st.info("No predictions stored yet. Run the pipeline first.")
except Exception as exc:
    st.warning(f"Could not read predictions: {exc}")

st.header("How to launch the primary dashboard")
st.code("cd dashboard\nnpm ci\nnpm run dev", language="bash")

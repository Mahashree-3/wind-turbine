"""
streamlit_app.py — Backup dashboard using Streamlit

Calls the SAME backend API (http://localhost:8000) that the React frontend
uses, so it shows the SAME real model predictions, just with a different UI.

SETUP:
    pip install streamlit requests

RUN (backend must already be running in another terminal):
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import time

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Wind Turbine Bearing Health Monitor",
    page_icon="🌬️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center; color:#22d3ee;'>🌬️ Wind Turbine Bearing Health Monitor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>Multi-sensor fault diagnosis and predictive maintenance</p>", unsafe_allow_html=True)

turbine = st.selectbox("Select Turbine", ["Turbine 1", "Turbine 2", "Turbine 3"])

col_a, col_b = st.columns([1, 1])
live_mode = col_a.toggle("🔴 Live Monitoring Mode", value=False)
refresh_seconds = col_b.slider("Refresh interval (sec)", 2, 10, 4, disabled=not live_mode)

if not live_mode:
    if st.button("🔄 Refresh (get new real sample)"):
        st.rerun()
else:
    st.info(f"🔴 LIVE — auto-refreshing every {refresh_seconds}s, cycling through real recorded samples "
            f"(simulated stream, not physical live sensors)")

# --- Fetch data from the backend ---
try:
    dashboard_resp = requests.get(f"{API_BASE}/api/dashboard", timeout=5)
    dashboard_resp.raise_for_status()
    data = dashboard_resp.json()
    sensors = data["sensors"]
    diagnosis = data["diagnosis"]

    trend_resp = requests.get(f"{API_BASE}/api/trend", timeout=5)
    trend_resp.raise_for_status()
    trend = trend_resp.json()

except requests.exceptions.ConnectionError:
    st.error("⚠️ Cannot connect to backend. Make sure it's running: "
             "cd backend && uvicorn main:app --reload --port 8000")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error fetching data: {e}")
    st.stop()

# --- Sensor cards ---
st.subheader("Live Sensor Readings")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acoustic", sensors["acoustic"])
col2.metric("Vibration", sensors["vibration"])
col3.metric("Temperature (°C)", sensors["temperature"])
col4.metric("Current", sensors["current"])

# --- Diagnosis panel ---
st.subheader("Diagnosis Results")

severity_colors = {
    "Normal": "🟢",
    "Minor": "🟡",
    "Moderate": "🟠",
    "Severe": "🔴",
}
severity_icon = severity_colors.get(diagnosis["severity"], "⚪")

col1, col2 = st.columns([2, 1])
col1.markdown(f"**Fault Type:** {diagnosis['fault_type']}")
col2.markdown(f"**Severity:** {severity_icon} {diagnosis['severity']}")

st.progress(diagnosis["confidence"], text=f"Confidence: {diagnosis['confidence']*100:.0f}%")
st.markdown(f"### Estimated Remaining Useful Life: **{diagnosis['rul_days']} days**")

# --- Trend chart ---
st.subheader("30-Day Bearing Health Trend")
trend_df = pd.DataFrame(trend).set_index("day")
st.line_chart(trend_df, y="score")

# --- Maintenance alert ---
st.warning(f"⚠️ Maintenance recommended within {diagnosis['rul_days']} days")

st.caption("Live predictions from a trained CAVF-Net-style model (MaFaulDa bearing fault dataset, "
           "91.56% test accuracy). RUL is a rule-based estimate from severity, not model-predicted.")

# --- Live streaming simulation: auto-refresh the page ---
if live_mode:
    time.sleep(refresh_seconds)
    st.rerun()
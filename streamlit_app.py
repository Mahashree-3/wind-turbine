"""
streamlit_app.py - Wind Turbine Bearing Health Monitor (extended)

Features:
  1. Audio upload prediction        (tab: Audio Upload)
  2. Fault history log + CSV export (tab: Fault History)
  3. Severe alert banner + optional Telegram notification (sidebar)
  4. Confusion matrix + accuracy    (tab: Model Performance)
  5. Fault probability chart        (tab: Live Monitor)
  6. Live vibration/acoustic charts (tab: Live Monitor)
  7. Multi-turbine overview         (tab: Fleet Overview)

RUN (backend must be running):
    pip install streamlit requests pandas altair
    streamlit run streamlit_app.py
"""

import time
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
TURBINES = ["Turbine 1", "Turbine 2", "Turbine 3"]
SEV_ICON = {"Normal": "🟢", "Minor": "🟡", "Moderate": "🟠", "Severe": "🔴"}
SEV_COLOR = {"Normal": "#16a34a", "Minor": "#facc15", "Moderate": "#fb923c", "Severe": "#dc2626"}
REPORTED_TEST_ACCURACY = 91.56
BUFFER_SIZE = 60

st.set_page_config(page_title="Wind Turbine Bearing Health Monitor", page_icon="🌬️", layout="wide")

st.markdown(
    "<style>.stApp { background-color: #0f172a; color: #e2e8f0; }</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("buffer", [])
ss.setdefault("latest", None)
ss.setdefault("trend", None)
ss.setdefault("last_turbine", None)
ss.setdefault("fleet", None)
ss.setdefault("audio_result", None)
ss.setdefault("eval_result", None)
ss.setdefault("alert_sent", {})


# ---------------------------------------------------------------- helpers
def api_get(path, **kwargs):
    r = requests.get(f"{API_BASE}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)
    r.raise_for_status()
    return r.json()


def send_telegram(text):
    token = ss.get("tg_token", "").strip()
    chat_id = ss.get("tg_chat", "").strip()
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=5,
        )
    except Exception:
        pass


def record(turbine, source, diagnosis):
    """Log a prediction to history and fire a Telegram alert if Severe (max 1/min per turbine)."""
    ss.history.append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Turbine": turbine,
            "Source": source,
            "Fault": diagnosis.get("fault_type"),
            "Severity": diagnosis.get("severity"),
            "Confidence %": round(float(diagnosis.get("confidence", 0)) * 100, 1),
            "RUL (days)": diagnosis.get("rul_days"),
        }
    )
    if diagnosis.get("severity") == "Severe":
        last = ss.alert_sent.get(turbine, 0)
        if time.time() - last > 60:
            ss.alert_sent[turbine] = time.time()
            send_telegram(
                f"🔴 SEVERE fault on {turbine}: {diagnosis.get('fault_type')} "
                f"(confidence {float(diagnosis.get('confidence', 0))*100:.0f}%, "
                f"RUL {diagnosis.get('rul_days')} days)"
            )


def severe_banner(turbine, diagnosis):
    if diagnosis.get("severity") == "Severe":
        st.error(
            f"🚨 SEVERE ALERT - {turbine}: {diagnosis.get('fault_type')} detected. "
            f"Immediate maintenance required (RUL ~{diagnosis.get('rul_days')} days)."
        )


def probability_chart(diagnosis):
    probs = diagnosis.get("probabilities")
    if probs is None:
        st.info(
            "Backend doesn't return per-class probabilities yet. "
            "Add a `probabilities` key in predict() to enable this chart."
        )
        return
    if isinstance(probs, dict):
        df = pd.DataFrame({"Class": list(probs.keys()), "Probability": list(probs.values())})
    else:
        names = ["Normal", "Outer Race", "Ball Fault", "Cage Fault"]
        df = pd.DataFrame({"Class": names[: len(probs)], "Probability": list(probs)})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Class:N", sort=None),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Class:N", legend=None),
            tooltip=["Class", alt.Tooltip("Probability:Q", format=".3f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


def show_diagnosis(diagnosis):
    icon = SEV_ICON.get(diagnosis.get("severity"), "⚪")
    c1, c2 = st.columns([2, 1])
    c1.markdown(f"**Fault Type:** {diagnosis.get('fault_type')}")
    c2.markdown(f"**Severity:** {icon} {diagnosis.get('severity')}")
    conf = float(diagnosis.get("confidence", 0))
    st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence: {conf*100:.0f}%")
    st.markdown(f"### Estimated Remaining Useful Life: **{diagnosis.get('rul_days')} days**")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    turbine = st.selectbox("Select Turbine", TURBINES)
    live_mode = st.toggle("🔴 Live Monitoring Mode", value=False)
    refresh_seconds = st.slider("Refresh interval (sec)", 2, 10, 5, disabled=not live_mode)
    manual_refresh = st.button("🔄 Get new sample", disabled=live_mode)

    with st.expander("📣 Telegram alerts (optional)"):
        st.text_input("Bot token", type="password", key="tg_token")
        st.text_input("Chat ID", key="tg_chat")
        st.caption("Leave empty to disable. Alerts fire only for Severe faults (max 1/min per turbine).")

# ---------------------------------------------------------------- header
st.markdown(
    "<h1 style='text-align:center; color:#22d3ee;'>🌬️ Wind Turbine Bearing Health Monitor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>Multi-sensor fault diagnosis and predictive maintenance</p>",
    unsafe_allow_html=True,
)
if live_mode:
    st.info(
        f"🔴 LIVE - auto-refreshing every {refresh_seconds}s, cycling through real recorded samples "
        "(simulated stream, not physical live sensors)"
    )

# ---------------------------------------------------------------- fetch
need_fetch = (
    live_mode
    or manual_refresh
    or ss.latest is None
    or ss.last_turbine != turbine
)
if need_fetch:
    try:
        data = api_get("/api/dashboard")
        ss.latest = data
        ss.trend = api_get("/api/trend")
        ss.last_turbine = turbine
        s = data["sensors"]
        ss.buffer.append(
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Acoustic": s["acoustic"],
                "Vibration": s["vibration"],
            }
        )
        ss.buffer = ss.buffer[-BUFFER_SIZE:]
        record(turbine, "Live sample", data["diagnosis"])
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot connect to backend. Start it with: "
            "cd backend && uvicorn main:app --reload --port 8000"
        )
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        st.stop()

sensors = ss.latest["sensors"]
diagnosis = ss.latest["diagnosis"]

# ---------------------------------------------------------------- tabs
tab_live, tab_fleet, tab_hist, tab_audio, tab_perf = st.tabs(
    ["📡 Live Monitor", "🏭 Fleet Overview", "📜 Fault History", "🎙️ Audio Upload", "📊 Model Performance"]
)

# ---- Live Monitor
with tab_live:
    severe_banner(turbine, diagnosis)

    st.subheader(f"Live Sensor Readings - {turbine}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acoustic", sensors["acoustic"])
    c2.metric("Vibration", sensors["vibration"])
    c3.metric("Temperature (°C)", sensors["temperature"])
    cur = sensors["current"]
    c4.metric("Current", "N/A" if isinstance(cur, str) else cur)

    st.subheader("Diagnosis Results")
    left, right = st.columns([1, 1])
    with left:
        show_diagnosis(diagnosis)
    with right:
        st.markdown("**Fault probability by class**")
        probability_chart(diagnosis)

    st.subheader("Live Sensor Trends")
    if len(ss.buffer) >= 2:
        buf = pd.DataFrame(ss.buffer).set_index("Time")
        ca, cv = st.columns(2)
        ca.markdown("**Acoustic level**")
        ca.line_chart(buf["Acoustic"])
        cv.markdown("**Vibration level**")
        cv.line_chart(buf["Vibration"])
    else:
        st.caption("Turn on Live Monitoring Mode (or refresh a few times) to build the live chart.")

    st.subheader("30-Day Bearing Health Trend")
    trend_df = pd.DataFrame(ss.trend).set_index("day")
    st.line_chart(trend_df, y="score")
    st.warning(f"⚠️ Maintenance recommended within {diagnosis.get('rul_days')} days")

# ---- Fleet Overview
with tab_fleet:
    st.subheader("All Turbines - Status")
    if st.button("🔍 Scan all turbines"):
        rows = []
        for t in TURBINES:
            try:
                d = api_get("/api/dashboard")
                record(t, "Fleet scan", d["diagnosis"])
                rows.append(
                    {
                        "Turbine": t,
                        "Fault": d["diagnosis"]["fault_type"],
                        "Severity": d["diagnosis"]["severity"],
                        "Confidence %": round(float(d["diagnosis"]["confidence"]) * 100, 1),
                        "RUL (days)": d["diagnosis"]["rul_days"],
                        "Temp (°C)": d["sensors"]["temperature"],
                    }
                )
            except Exception as e:
                rows.append({"Turbine": t, "Fault": f"error: {e}", "Severity": "-"})
        ss.fleet = pd.DataFrame(rows)

    if ss.fleet is not None:
        def color_sev(v):
            c = SEV_COLOR.get(v)
            return f"background-color: {c}; color: black; font-weight: 700" if c else ""

        st.dataframe(ss.fleet.style.map(color_sev, subset=["Severity"]), width="stretch", hide_index=True)
        if (ss.fleet["Severity"] == "Severe").any():
            st.error("🚨 One or more turbines need immediate attention.")
    else:
        st.caption("Click 'Scan all turbines' to fetch the current status of every turbine.")

# ---- Fault History
with tab_hist:
    st.subheader("Fault History Log")
    if ss.history:
        hist_df = pd.DataFrame(ss.history).iloc[::-1]
        st.dataframe(hist_df, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Download CSV",
            hist_df.to_csv(index=False).encode("utf-8"),
            file_name=f"fault_history_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )
        if c2.button("🗑️ Clear history"):
            ss.history = []
            st.rerun()
    else:
        st.caption("No predictions logged yet.")
    st.caption("History is kept for this browser session only.")

# ---- Audio Upload
with tab_audio:
    st.subheader("Predict from your own audio")
    st.caption(
        "The model needs acoustic + vibration input. Your audio supplies the acoustic part; "
        "the vibration part is taken from a real dataset sample (the backend states this in its response)."
    )
    up = st.file_uploader("Upload a .wav file", type=["wav", "mp3", "flac", "ogg"])
    if up is not None:
        st.audio(up)
        if st.button("▶️ Predict from audio"):
            with st.spinner("Extracting features and predicting..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/api/predict-audio",
                        files={"file": (up.name, up.getvalue(), up.type or "audio/wav")},
                        timeout=60,
                    )
                    if r.status_code != 200:
                        st.error(f"Backend error: {r.json().get('detail', r.text)}")
                    else:
                        ss.audio_result = (up.name, r.json())
                        record("Uploaded audio", up.name, r.json())
                except Exception as e:
                    st.error(f"Could not reach backend: {e}")

    if ss.audio_result:
        name, res = ss.audio_result
        st.markdown(f"#### Result for `{name}`")
        severe_banner("Uploaded audio", res)
        show_diagnosis(res)
        probability_chart(res)
        st.info(res.get("transparency_note", ""))
        st.caption(f"Acoustic: {res.get('acoustic_source')}  |  Vibration: {res.get('vibration_source')}")

# ---- Model Performance
with tab_perf:
    st.subheader("Model Performance")
    m1, m2 = st.columns(2)
    m1.metric("Reported test accuracy", f"{REPORTED_TEST_ACCURACY}%")

    n = st.slider("Samples to evaluate", 50, 1000, 300, step=50)
    if st.button("🧪 Run evaluation"):
        with st.spinner("Running model on dataset samples..."):
            try:
                ss.eval_result = api_get(f"/api/evaluate?n={n}", timeout=300)
            except Exception as e:
                st.error(f"Evaluation failed (is /api/evaluate added to main.py?): {e}")

    ev = ss.eval_result
    if ev:
        classes = ev["classes"]
        cm = pd.DataFrame(ev["matrix"], index=classes, columns=classes)
        m2.metric(f"Measured accuracy ({ev['n']} samples)", f"{ev['accuracy']*100:.2f}%")

        long = cm.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count")
        long = long.rename(columns={"index": "Actual"})
        base = alt.Chart(long).encode(
            x=alt.X("Predicted:N", sort=classes),
            y=alt.Y("Actual:N", sort=classes),
        )
        heat = base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")))
        text = base.mark_text(fontSize=16).encode(
            text="Count:Q",
            color=alt.condition(alt.datum.Count > long["Count"].max() / 2, alt.value("white"), alt.value("black")),
        )
        st.markdown("**Confusion matrix**")
        st.altair_chart((heat + text).properties(height=350), width="stretch")

        per_class = []
        for i, c in enumerate(classes):
            total = sum(ev["matrix"][i])
            per_class.append({"Class": c, "Accuracy %": round(100 * ev["matrix"][i][i] / total, 1) if total else 0})
        st.markdown("**Per-class accuracy**")
        st.bar_chart(pd.DataFrame(per_class).set_index("Class"))
        st.caption(
            "Measured on random samples from the processed dataset, which may include training data, "
            "so it can differ from the held-out test accuracy above."
            + (f" ({ev['skipped']} predictions skipped: unrecognised class name.)" if ev.get("skipped") else "")
        )

st.caption(
    "Live predictions from a trained CAVF-Net-style model (MaFaulDa bearing fault dataset, "
    f"{REPORTED_TEST_ACCURACY}% test accuracy). RUL is a rule-based estimate from severity, not model-predicted."
)

# ---------------------------------------------------------------- auto refresh
if live_mode:
    time.sleep(refresh_seconds)
    st.rerun()
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Live Fraud Monitor", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#4F46E5,#7C3AED); padding: 24px 30px;
       border-radius: 16px; margin-bottom: 20px;}
.hero h1 {color: #fff; margin: 0; font-size: 1.8rem;}
.hero p {color: #E0E7FF; margin: 6px 0 0 0;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #E5E7EB;
       border-left: 5px solid #4F46E5; padding: 14px 18px; border-radius: 12px;
       box-shadow: 0 1px 3px rgba(0,0,0,.06);}
</style>
<div class="hero">
  <h1>📡 Live Fraud Monitor</h1>
  <p>Simulated real-time transactions scored by the FastAPI model | auto-refresh every 2 seconds</p>
</div>
""", unsafe_allow_html=True)

LOG = "data/live_log.csv"
SHOW = ["time", "amount", "hour", "new_receiver", "device_changed",
        "location_mismatch", "fraud_probability", "decision", "top_reason"]


@st.fragment(run_every=2)
def live():
    if not os.path.exists(LOG):
        st.info("Inka data ledu. Simulator run cheyandi: python api/simulator.py")
        return
    try:
        df = pd.read_csv(LOG)
    except Exception:
        return
    if df.empty:
        st.info("Simulator start ayyindi, transactions kosam wait chestunnam...")
        return

    fraud = df[df["decision"] == "FRAUD"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{len(df):,}")
    c2.metric("Fraud alerts", f"{len(fraud):,}")
    c3.metric("Fraud rate", f"{len(fraud) / len(df):.1%}")
    c4.metric("Avg fraud probability", f"{df['fraud_probability'].mean():.2f}")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Fraud probability (last 50 transactions)")
        st.line_chart(df.tail(50).reset_index(drop=True)["fraud_probability"],
                      color="#4F46E5")
    with right:
        st.subheader("Top reasons behind fraud alerts")
        if fraud.empty:
            st.info("Inka fraud alerts levu.")
        else:
            st.bar_chart(fraud["top_reason"].value_counts(), color="#EF4444")

    st.subheader("🚨 Latest fraud alerts")
    if fraud.empty:
        st.info("Inka fraud alerts levu.")
    else:
        st.dataframe(fraud.tail(10).iloc[::-1][SHOW], hide_index=True,
                     use_container_width=True)

    st.subheader("Recent transactions")
    st.dataframe(df.tail(15).iloc[::-1][SHOW], hide_index=True,
                 use_container_width=True)


live()
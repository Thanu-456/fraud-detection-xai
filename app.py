import os
import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#4F46E5,#7C3AED); padding: 28px 32px;
       border-radius: 16px; margin-bottom: 20px;}
.hero h1 {color: #fff; margin: 0; font-size: 2rem;}
.hero p {color: #E0E7FF; margin: 6px 0 0 0;}
button[data-baseweb="tab"] {background: #fff; border-radius: 10px 10px 0 0; padding: 10px 18px;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #E5E7EB;
       border-left: 5px solid #4F46E5; padding: 14px 18px; border-radius: 12px;
       box-shadow: 0 1px 3px rgba(0,0,0,.06);}
.result-card {padding: 22px; border-radius: 14px; margin: 12px 0;}
.result-card h2 {color: #fff; margin: 0;}
.result-card p {color: #fff; margin: 4px 0 0 0;}
.fraud {background: linear-gradient(135deg,#DC2626,#F97316);}
.normal {background: linear-gradient(135deg,#059669,#10B981);}
</style>
""", unsafe_allow_html=True)

COLS = ["amount", "hour", "account_age_days", "txn_count_24h",
        "new_receiver", "device_changed", "location_mismatch"]
NUM_COLS = ["amount", "account_age_days", "txn_count_24h"]

PRESETS = {
    "Normal transaction": dict(amount=450.0, hour=14, age=900, cnt=2, new=0, dev=0, loc=0),
    "Suspicious transaction": dict(amount=45000.0, hour=2, age=15, cnt=12, new=1, dev=1, loc=1),
}


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_upi.pkl")
    scaler = joblib.load("models/scaler_upi.pkl")
    return model, scaler, shap.TreeExplainer(model)


def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Image not found: {path}")


def show_results(path, title):
    st.subheader(title)
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(
            df.set_index("Model")[["Precision", "Recall", "F1", "PR-AUC"]],
            color=["#4F46E5", "#10B981", "#F59E0B", "#EF4444"],
        )
    else:
        st.info(f"File not found: {path}")


model, scaler, explainer = load_artifacts()

st.markdown("""
<div class="hero">
  <h1>🛡️ Explainable ML for UPI and Card Fraud Detection</h1>
  <p>XGBoost + SHAP &nbsp;|&nbsp; UPI data is synthetic &nbsp;|&nbsp; Card data is from Kaggle (ULB)</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Check Transaction", "Model Results", "Explainability", "About"]
)

# ---------------- Tab 1: live check ----------------
with tab1:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.subheader("Transaction details")
        preset = st.selectbox("Load an example", list(PRESETS.keys()))
        p = PRESETS[preset]

        amount = st.number_input("Amount (Rs)", 1.0, 1_000_000.0, p["amount"])
        hour = st.slider("Hour of day", 0, 23, p["hour"])
        age = st.number_input("Account age (days)", 1, 5000, p["age"])
        cnt = st.number_input("Transactions in last 24h", 0, 100, p["cnt"])
        yes_no = lambda x: "Yes" if x else "No"
        new = st.selectbox("New receiver?", [0, 1], index=p["new"], format_func=yes_no)
        dev = st.selectbox("Device changed?", [0, 1], index=p["dev"], format_func=yes_no)
        loc = st.selectbox("Location mismatch?", [0, 1], index=p["loc"], format_func=yes_no)
        threshold = st.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.05)
        check = st.button("Check transaction", type="primary")

    with right:
        st.subheader("Result")
        if check:
            raw = pd.DataFrame([[amount, hour, age, cnt, new, dev, loc]], columns=COLS)
            scaled = raw.copy()
            scaled[NUM_COLS] = scaler.transform(raw[NUM_COLS])

            prob = float(model.predict_proba(scaled)[0, 1])

            m1, m2 = st.columns(2)
            m1.metric("Fraud probability", f"{prob:.1%}")
            m2.metric("Decision", "FRAUD" if prob >= threshold else "NORMAL")
            st.progress(min(prob, 1.0))

            if prob >= threshold:
                st.markdown(
                    '<div class="result-card fraud"><h2>🚨 FRAUD ALERT</h2>'
                    '<p>This transaction is flagged as suspicious.</p></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="result-card normal"><h2>✅ LOOKS NORMAL</h2>'
                    '<p>This transaction looks safe.</p></div>',
                    unsafe_allow_html=True)

            sv = explainer(scaled)
            contrib = pd.DataFrame({
                "Feature": COLS,
                "Value": raw.iloc[0].values,
                "Impact": sv.values[0].round(3),
            })
            contrib["Pushes toward"] = contrib["Impact"].apply(
                lambda x: "Fraud" if x > 0 else "Normal"
            )
            contrib = contrib.sort_values("Impact", ascending=False)

            st.markdown("**Why this result? (SHAP)**")
            st.dataframe(contrib, hide_index=True, use_container_width=True)

            exp = shap.Explanation(
                values=sv.values[0],
                base_values=sv.base_values[0],
                data=raw.iloc[0].values,
                feature_names=COLS,
            )
            plt.figure()
            shap.plots.waterfall(exp, show=False)
            st.pyplot(plt.gcf())
            plt.close()
        else:
            st.info("Left side lo details select chesi, 'Check transaction' click cheyandi.")

# ---------------- Tab 2: results ----------------
with tab2:
    show_results("reports/upi_model_results.csv", "UPI model comparison (synthetic data)")
    show_results("reports/model_results.csv", "Card model comparison (Kaggle data)")
    st.caption("Accuracy use cheyaledu, because data heavily imbalanced. "
               "Precision, Recall, F1 mariyu PR-AUC use chesam.")

# ---------------- Tab 3: explainability ----------------
with tab3:
    st.subheader("UPI model: SHAP")
    c1, c2 = st.columns(2)
    with c1:
        show_image("reports/figures/upi_shap_bar.png", "Feature importance")
    with c2:
        show_image("reports/figures/upi_shap_summary.png", "Feature effects")

    st.subheader("Card model: SHAP")
    c3, c4 = st.columns(2)
    with c3:
        show_image("reports/figures/shap_bar.png", "Feature importance")
    with c4:
        show_image("reports/figures/shap_summary.png", "Feature effects")

    st.subheader("Data exploration (card data)")
    c5, c6, c7 = st.columns(3)
    with c5:
        show_image("reports/figures/class_imbalance.png", "Class imbalance")
    with c6:
        show_image("reports/figures/amount_boxplot.png", "Amount: normal vs fraud")
    with c7:
        show_image("reports/figures/fraud_by_hour.png", "Fraud rate by hour")

# ---------------- Tab 4: about ----------------
with tab4:
    st.subheader("About this project")
    st.markdown(
        """
**Goal:** Fraud transactions ni detect cheyadam, mariyu model enduku alaa decide chesindo explain cheyadam.

**Approach**
1. Card data (Kaggle) mariyu synthetic UPI data
2. Class imbalance handle cheyadam (`class_weight`, `scale_pos_weight`)
3. Logistic Regression, Random Forest, XGBoost compare cheyadam
4. SHAP tho global mariyu per-transaction explanations

**Limitations**
- UPI data synthetic, so results real-world performance ni chupavu
- Card data lo V1-V28 anonymized (PCA), so avi human-readable kaadu
- Idi demo app, real bank system ki connect kaadu
        """
    )
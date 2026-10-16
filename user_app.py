import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
DB = "data/app.db"
THRESHOLD = 0.5

REASON_TEXT = {
    "device_changed": "Kotha device nundi transaction",
    "new_receiver": "Kotha receiver (mundu pay cheyaledu)",
    "location_mismatch": "Usual location kaadu",
    "amount": "Amount chala ekkuva",
    "hour": "Raatri time lo transaction",
    "account_age_days": "Account chala kotha",
    "txn_count_24h": "24 gantallo chala transactions",
}

st.set_page_config(page_title="SecurePay", layout="wide")
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
.result-card {padding: 22px; border-radius: 14px; margin: 12px 0;}
.result-card h2 {color: #fff; margin: 0;}
.result-card p {color: #fff; margin: 4px 0 0 0;}
.fraud {background: linear-gradient(135deg,#DC2626,#F97316);}
.normal {background: linear-gradient(135deg,#059669,#10B981);}
</style>
<div class="hero">
  <h1>💳 SecurePay</h1>
  <p>Real-time fraud protection with explainable AI</p>
</div>
""", unsafe_allow_html=True)


# ---------------- Database ----------------
def db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE, salt TEXT, pw_hash TEXT,
            role TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS transactions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT, receiver TEXT, amount REAL, created_at TEXT,
            fraud_probability REAL, decision TEXT, top_reason TEXT, status TEXT)""")


init_db()


# ---------------- Auth ----------------
def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def register(username, pw):
    username = username.strip().lower()
    if len(username) < 3 or len(pw) < 6:
        return False, "Username min 3 characters, password min 6 characters undali."
    salt = secrets.token_hex(8)
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            return False, "Ee username already undi."
        n = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        role = "admin" if n == 0 else "user"
        c.execute(
            "INSERT INTO users(username,salt,pw_hash,role,created_at) VALUES(?,?,?,?,?)",
            (username, salt, hash_pw(pw, salt), role,
             datetime.now().isoformat(timespec="seconds")),
        )
    return True, f"Account create ayyindi (role: {role}). Ippudu Login tab lo login cheyandi."


def login(username, pw):
    username = username.strip().lower()
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and secrets.compare_digest(row["pw_hash"], hash_pw(pw, row["salt"])):
        return dict(row)
    return None


# ---------------- Fraud check ----------------
def build_features(user, receiver, amount, device_changed, location_mismatch):
    now = datetime.now()
    created = datetime.fromisoformat(user["created_at"])
    since = (now - timedelta(hours=24)).isoformat(timespec="seconds")
    with db() as c:
        cnt = c.execute(
            "SELECT COUNT(*) FROM transactions WHERE username=? AND created_at>=?",
            (user["username"], since)).fetchone()[0]
        seen = c.execute(
            "SELECT 1 FROM transactions WHERE username=? AND receiver=?",
            (user["username"], receiver)).fetchone()
    return {
        "amount": float(amount),
        "hour": now.hour,
        "account_age_days": max(1, (now - created).days),
        "txn_count_24h": int(cnt),
        "new_receiver": 0 if seen else 1,
        "device_changed": int(device_changed),
        "location_mismatch": int(location_mismatch),
    }


def send_money(user, receiver, amount, device_changed, location_mismatch):
    feats = build_features(user, receiver, amount, device_changed, location_mismatch)
    resp = requests.post(f"{API_URL}?threshold={THRESHOLD}", json=feats, timeout=5)
    resp.raise_for_status()
    r = resp.json()
    status = "BLOCKED" if r["decision"] == "FRAUD" else "APPROVED"
    with db() as c:
        c.execute(
            """INSERT INTO transactions(username,receiver,amount,created_at,
               fraud_probability,decision,top_reason,status) VALUES(?,?,?,?,?,?,?,?)""",
            (user["username"], receiver, float(amount),
             datetime.now().isoformat(timespec="seconds"),
             r["fraud_probability"], r["decision"],
             r["top_reasons"][0]["feature"], status),
        )
    return r, status


def load_txns(username=None):
    with db() as c:
        if username:
            q, p = "SELECT * FROM transactions WHERE username=? ORDER BY id DESC", (username,)
        else:
            q, p = "SELECT * FROM transactions ORDER BY id DESC", ()
        return pd.read_sql_query(q, c, params=p)


# ---------------- Login screen ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        t_login, t_reg = st.tabs(["Login", "Register"])
        with t_login:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login", type="primary"):
                    user = login(u, p)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Username ledha password tappu.")
        with t_reg:
            with st.form("reg_form"):
                u2 = st.text_input("Choose username")
                p2 = st.text_input("Choose password", type="password")
                if st.form_submit_button("Create account"):
                    ok, msg = register(u2, p2)
                    (st.success if ok else st.error)(msg)
    st.stop()


# ---------------- Logged-in app ----------------
user = st.session_state.user
with st.sidebar:
    st.markdown(f"### 👤 {user['username']}")
    st.caption(f"Role: {user['role']}")
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

tab_names = ["Send Money", "My Transactions"]
if user["role"] == "admin":
    tab_names.append("Admin Monitor")
tabs = st.tabs(tab_names)

# ----- Send money -----
with tabs[0]:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Send money")
        with st.form("send_form"):
            receiver = st.text_input("Receiver (UPI ID / name)")
            amount = st.number_input("Amount (Rs)", 1.0, 1_000_000.0, 500.0)
            st.caption("Demo kosam: ee rendu browser nundi detect cheyalem, so manual ga simulate chestam.")
            dev = st.checkbox("New device nundi login (simulate)")
            loc = st.checkbox("Different location nundi (simulate)")
            go = st.form_submit_button("Pay now", type="primary")

    with right:
        st.subheader("Result")
        if go:
            if not receiver.strip():
                st.warning("Receiver enter cheyandi.")
            else:
                try:
                    r, status = send_money(user, receiver.strip().lower(), amount, dev, loc)
                except Exception as e:
                    st.error(f"API run avvatledu. Uvicorn terminal check cheyandi. ({e})")
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Fraud probability", f"{r['fraud_probability']:.1%}")
                    m2.metric("Status", status)
                    if status == "BLOCKED":
                        st.markdown(
                            '<div class="result-card fraud"><h2>🚨 TRANSACTION BLOCKED</h2>'
                            '<p>Suspicious activity detect ayyindi. Money pampabadaledu.</p></div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="result-card normal"><h2>✅ PAYMENT SUCCESSFUL</h2>'
                            f'<p>Rs {amount:,.2f} pampabadindi.</p></div>',
                            unsafe_allow_html=True)
                    risky = [x for x in r["top_reasons"] if x["impact"] > 0]
                    if risky:
                        st.markdown("**Enduku ee risk? (SHAP)**")
                        for x in risky:
                            st.write("⚠️", REASON_TEXT.get(x["feature"], x["feature"]))
        else:
            st.info("Details enter chesi 'Pay now' click cheyandi.")

# ----- My transactions -----
with tabs[1]:
    st.subheader("My transactions")
    mine = load_txns(user["username"])
    if mine.empty:
        st.info("Inka transactions levu.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(mine))
        c2.metric("Approved", int((mine["status"] == "APPROVED").sum()))
        c3.metric("Blocked", int((mine["status"] == "BLOCKED").sum()))
        st.dataframe(
            mine[["created_at", "receiver", "amount", "status", "fraud_probability", "top_reason"]],
            hide_index=True, use_container_width=True)

# ----- Admin -----
if user["role"] == "admin":
    with tabs[2]:
        st.subheader("Admin monitor (all users)")
        allt = load_txns()
        with db() as c:
            n_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if allt.empty:
            st.info("Inka transactions levu.")
        else:
            blocked = allt[allt["status"] == "BLOCKED"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Users", n_users)
            c2.metric("Transactions", len(allt))
            c3.metric("Blocked", len(blocked))
            c4.metric("Fraud rate", f"{len(blocked) / len(allt):.1%}")
            st.markdown("**🚨 Blocked transactions**")
            if blocked.empty:
                st.info("Inka blocked transactions levu.")
            else:
                st.dataframe(
                    blocked[["created_at", "username", "receiver", "amount",
                             "fraud_probability", "top_reason"]],
                    hide_index=True, use_container_width=True)
                st.bar_chart(blocked["top_reason"].value_counts(), color="#EF4444")
            st.markdown("**All transactions**")
            st.dataframe(
                allt[["created_at", "username", "receiver", "amount", "status", "fraud_probability"]],
                hide_index=True, use_container_width=True)
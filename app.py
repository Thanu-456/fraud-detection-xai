import hashlib
import math
import os
import secrets
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st


def _cfg(key, default=""):
    """Read a value from an environment variable, falling back to Streamlit Secrets."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default


# Optional: only this username becomes admin (if empty, the first registered user is admin)
ADMIN_USERNAME = _cfg("ADMIN_USERNAME").strip().lower()

# Address of the FastAPI service (api/main.py)
API_URL = _cfg("API_URL", "").rstrip("/")  # leave empty to score inside the app (Streamlit Cloud)


def fraud_score(kind, feats, threshold):
    """Score via the FastAPI service if API_URL is set, otherwise score inside the app."""
    if not API_URL:
        from fraud_engine import score
        return score(kind, feats, threshold)
    path = "/predict" if kind == "upi" else "/predict-card"
    try:
        resp = requests.post(f"{API_URL}{path}", params={"threshold": threshold},
                             json=feats, timeout=60)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach the fraud API at {API_URL}. "
            "Start it with: uvicorn api.main:app --host 127.0.0.1 --port 8000")
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")
    return resp.json()


DB = "data/app.db"
THRESHOLD = 0.5

REASON_TEXT = {
    "device_changed": "Transaction from a new device",
    "new_receiver": "New receiver (never paid before)",
    "location_mismatch": "Not the usual location",
    "amount": "Unusually high amount",
    "hour": "Transaction at night time",
    "account_age_days": "Very new account",
    "txn_count_24h": "Too many transactions in 24 hours",
}

CATEGORIES = {"Grocery": 0.0, "Restaurant": 0.1, "Fuel": 0.2, "ATM": 0.4,
              "Travel": 0.5, "Online shopping": 0.6, "Electronics": 0.8,
              "Jewellery": 1.0}
CITIES = {"Hyderabad": (17.385, 78.487), "Bengaluru": (12.972, 77.595),
          "Mumbai": (19.076, 72.878), "Chennai": (13.083, 80.271),
          "Delhi": (28.614, 77.209)}

CARD_REASON_TEXT = {
    "amount_vs_avg": "Amount much higher than your normal spending",
    "amount": "Unusually high amount",
    "txn_count_1h": "Too many transactions in one hour",
    "minutes_since_last": "Too soon after the last transaction",
    "distance_from_last_km": "Very far from the last transaction",
    "travel_speed_kmh": "Impossible travel (speed not physically possible)",
    "merchant_risk": "Risky merchant category",
    "is_online": "Online payment (card not present)",
    "is_foreign": "International merchant",
}

st.set_page_config(page_title="SecurePay", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#4F46E5,#7C3AED); padding: 28px 32px;
       border-radius: 16px; margin-bottom: 20px;}
.hero h1 {color: #fff; margin: 0; font-size: 2rem;}
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
  <p>Real-time UPI and card fraud protection with explainable AI</p>
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
        c.execute("""CREATE TABLE IF NOT EXISTS card_transactions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT, merchant TEXT, category TEXT, city TEXT, amount REAL,
            created_at TEXT, fraud_probability REAL, status TEXT, top_reason TEXT)""")


init_db()


# ---------------- Error handling ----------------
def api_error_message(e):
    """Short, readable message when scoring fails."""
    return f"Could not score the transaction: {e}"


# ---------------- Auth ----------------
def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def register(username, pw):
    username = username.strip().lower()
    if len(username) < 3 or len(pw) < 6:
        return False, "Username must be at least 3 characters and password at least 6 characters."
    salt = secrets.token_hex(8)
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            return False, "This username already exists."
        n = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        # If ADMIN_USERNAME is set, only that username becomes admin. Otherwise the first user does.
        is_first_admin = (username == ADMIN_USERNAME) if ADMIN_USERNAME else (n == 0)
        role = "admin" if is_first_admin else "user"
        c.execute(
            "INSERT INTO users(username,salt,pw_hash,role,created_at) VALUES(?,?,?,?,?)",
            (username, salt, hash_pw(pw, salt), role,
             datetime.now().isoformat(timespec="seconds")),
        )
    return True, f"Account created (role: {role}). Please log in using the Login tab."


def login(username, pw):
    username = username.strip().lower()
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and secrets.compare_digest(row["pw_hash"], hash_pw(pw, row["salt"])):
        return dict(row)
    return None


# ---------------- UPI payments ----------------
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
    r = fraud_score("upi", feats, THRESHOLD)
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


# ---------------- Card payments ----------------
def haversine(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    d = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(d))


def build_card_features(user, category, city, amount, online, foreign):
    now = datetime.now()
    since = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    with db() as c:
        cnt = c.execute(
            "SELECT COUNT(*) FROM card_transactions WHERE username=? AND created_at>=?",
            (user["username"], since)).fetchone()[0]
        avg = c.execute(
            "SELECT AVG(amount) FROM card_transactions WHERE username=? AND status='APPROVED'",
            (user["username"],)).fetchone()[0]
        last = c.execute(
            "SELECT city, created_at FROM card_transactions "
            "WHERE username=? AND status='APPROVED' ORDER BY id DESC LIMIT 1",
            (user["username"],)).fetchone()

    if last:
        minutes = (now - datetime.fromisoformat(last["created_at"])).total_seconds() / 60
        minutes = min(max(minutes, 1.0), 10080.0)
        dist = haversine(CITIES[last["city"]], CITIES[city])
    else:
        minutes, dist = 1440.0, 0.0

    return {
        "amount": float(amount),
        "amount_vs_avg": float(amount / avg) if avg else 1.0,
        "txn_count_1h": int(cnt),
        "minutes_since_last": round(minutes, 1),
        "distance_from_last_km": round(dist, 1),
        "travel_speed_kmh": round(min(dist / (minutes / 60), 3000.0), 1),
        "merchant_risk": CATEGORIES[category],
        "is_online": int(online),
        "is_foreign": int(foreign),
    }


def card_pay(user, merchant, category, city, amount, online, foreign):
    feats = build_card_features(user, category, city, amount, online, foreign)
    r = fraud_score("card", feats, THRESHOLD)
    status = "BLOCKED" if r["decision"] == "FRAUD" else "APPROVED"
    with db() as c:
        c.execute(
            """INSERT INTO card_transactions(username,merchant,category,city,amount,
               created_at,fraud_probability,status,top_reason) VALUES(?,?,?,?,?,?,?,?,?)""",
            (user["username"], merchant, category, city, float(amount),
             datetime.now().isoformat(timespec="seconds"),
             r["fraud_probability"], status, r["top_reasons"][0]["feature"]),
        )
    return r, status, feats


def load_card_txns(username=None):
    with db() as c:
        if username:
            q, p = "SELECT * FROM card_transactions WHERE username=? ORDER BY id DESC", (username,)
        else:
            q, p = "SELECT * FROM card_transactions ORDER BY id DESC", ()
        return pd.read_sql_query(q, c, params=p)


def result_card(fraud, title_fraud, text_fraud, title_ok, text_ok):
    cls, title, text = ("fraud", title_fraud, text_fraud) if fraud else ("normal", title_ok, text_ok)
    st.markdown(f'<div class="result-card {cls}"><h2>{title}</h2><p>{text}</p></div>',
                unsafe_allow_html=True)


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
                        st.error("Incorrect username or password.")
        with t_reg:
            with st.form("reg_form"):
                u2 = st.text_input("Choose username")
                p2 = st.text_input("Choose password", type="password")
                st.caption("Demo app: do not use a real password, use a dummy one.")
                if st.form_submit_button("Create account"):
                    ok, msg = register(u2, p2)
                    (st.success if ok else st.error)(msg)
    st.stop()


# ---------------- Logged-in app ----------------
user = st.session_state.user
is_admin = user["role"] == "admin"

with st.sidebar:
    st.markdown(f"### 👤 {user['username']}")
    st.caption(f"Role: {user['role']}")
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

names = ["Send Money (UPI)", "Card Payment", "My Transactions"]
if is_admin:
    names.append("Admin Monitor")
tabs = st.tabs(names)

# ----- UPI: send money -----
with tabs[0]:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("UPI Fraud Test")
        with st.form("send_form"):
            receiver = st.text_input("Test Receiver ID")
            amount = st.number_input("Amount (Rs)", 1.0, 1_000_000.0, 500.0, key="upi_amount")
            st.caption("For the demo, device and location are simulated manually.")
            dev = st.checkbox("Simulate New Device Login (simulate)")
            loc = st.checkbox("Simulate Location Change (simulate)")
            go = st.form_submit_button("Analyze Transaction", type="primary")

    with right:
        st.subheader("Result")
        if go:
            if not receiver.strip():
                st.warning("Please enter the receiver.")
            else:
                try:
                    with st.spinner("Checking transaction... (the first request may take a little longer)"):
                        r, status = send_money(user, receiver.strip().lower(), amount, dev, loc)
                except Exception as e:
                    st.error(api_error_message(e))
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Fraud probability", f"{r['fraud_probability']:.1%}")
                    m2.metric("Status", status)
                    result_card(status == "BLOCKED",
                                "🚨 This was a fraud detection simulation.",
                                "Suspicious activity detected. This was a fraud detection simulation..",
                                "✅ PAYMENT SUCCESSFUL",
                                f"Rs {amount:,.2f} has been sent.")
                    risky = [x for x in r["top_reasons"] if x["impact"] > 0]
                    if risky:
                        st.markdown("**Why is this risky?**")
                        for x in risky:
                            st.write("⚠️", REASON_TEXT.get(x["feature"], x["feature"]))
        else:
            st.info("Enter the details and click 'Analyze Transaction'.")

# ----- Card payment -----
with tabs[1]:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Card Fraud Test")
        with st.form("card_form"):
            merchant = st.text_input("Merchant name")
            category = st.selectbox("Merchant category", list(CATEGORIES.keys()))
            city = st.selectbox("Transaction city", list(CITIES.keys()))
            c_amount = st.number_input("Amount (Rs)", 1.0, 1_000_000.0, 1500.0, key="card_amount")
            online = st.checkbox("Online payment (card not present)")
            foreign = st.checkbox("International merchant")
            pay = st.form_submit_button("Analyze Card Transaction", type="primary")

    with right:
        st.subheader("Result")
        if pay:
            if not merchant.strip():
                st.warning("Please enter the merchant name.")
            else:
                try:
                    with st.spinner("Checking card payment... (the first request may take a little longer)"):
                        r, status, feats = card_pay(user, merchant.strip(), category, city,
                                                    c_amount, online, foreign)
                except Exception as e:
                    st.error(api_error_message(e))
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Fraud probability", f"{r['fraud_probability']:.1%}")
                    m2.metric("Status", status)
                    result_card(status == "BLOCKED",
                                "🚨 CARD PAYMENT BLOCKED",
                                "Suspicious activity detected.",
                                "✅ CARD PAYMENT APPROVED",
                                f"Rs {c_amount:,.2f} has been charged.")
                    risky = [x for x in r["top_reasons"] if x["impact"] > 0]
                    if risky:
                        st.markdown("**Why is this risky?**")
                        for x in risky:
                            st.write("⚠️", CARD_REASON_TEXT.get(x["feature"], x["feature"]))
                    st.markdown("**Automatically calculated features**")
                    st.dataframe(
                        pd.DataFrame({"Feature": list(feats.keys()),
                                      "Value": [round(v, 2) for v in feats.values()]}),
                        hide_index=True, use_container_width=True)
        else:
            st.info("Enter the details and click 'Analyze Card Transaction'.")

# ----- My transactions -----
with tabs[2]:
    st.subheader("My UPI transactions")
    mine = load_txns(user["username"])
    if mine.empty:
        st.info("No UPI transactions yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(mine))
        c2.metric("Approved", int((mine["status"] == "APPROVED").sum()))
        c3.metric("Blocked", int((mine["status"] == "BLOCKED").sum()))
        st.dataframe(
            mine[["created_at", "receiver", "amount", "status", "fraud_probability", "top_reason"]],
            hide_index=True, use_container_width=True)

    st.subheader("My card transactions")
    mine_c = load_card_txns(user["username"])
    if mine_c.empty:
        st.info("No card transactions yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(mine_c))
        c2.metric("Approved", int((mine_c["status"] == "APPROVED").sum()))
        c3.metric("Blocked", int((mine_c["status"] == "BLOCKED").sum()))
        st.dataframe(
            mine_c[["created_at", "merchant", "category", "city", "amount",
                    "status", "fraud_probability", "top_reason"]],
            hide_index=True, use_container_width=True)

# ----- Admin monitor -----
if is_admin:
    with tabs[3]:
        st.subheader("Admin monitor (all users)")
        with db() as c:
            n_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        st.metric("Users", n_users)

        for label, data, cols in [
            ("UPI", load_txns(),
             ["created_at", "username", "receiver", "amount", "fraud_probability", "top_reason"]),
            ("Card", load_card_txns(),
             ["created_at", "username", "merchant", "city", "amount", "fraud_probability", "top_reason"]),
        ]:
            st.markdown(f"### {label} transactions")
            if data.empty:
                st.info(f"No {label} transactions yet.")
                continue
            blocked = data[data["status"] == "BLOCKED"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions", len(data))
            c2.metric("Blocked", len(blocked))
            c3.metric("Fraud rate", f"{len(blocked) / len(data):.1%}")
            if blocked.empty:
                st.info("No blocked transactions yet.")
            else:
                st.dataframe(blocked[cols], hide_index=True, use_container_width=True)
                st.bar_chart(blocked["top_reason"].value_counts(), color="#EF4444")
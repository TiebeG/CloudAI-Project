import streamlit as st
import requests
from datetime import datetime
import holidays

# API endpoints
ND_API = "http://127.0.0.1:8000/predict_nd"
HP_API = "http://127.0.0.1:8000/predict_house_price"

st.set_page_config(page_title="AI Predictor", page_icon="📈")
st.title("📈 Demand & 🏡 House Price Predictor")

# ─────────────────────────────────────────────
# NATIONAL DEMAND
# ─────────────────────────────────────────────
st.header("⚡ National Demand Prediction")

day = st.number_input("Day", 1, 31, 1)
month = st.number_input("Month", 1, 12, 1)
year = st.number_input("Year", 2009, 2100, 2025)
hour = st.number_input("Hour", 0, 23, 12)

try:
    dt = datetime(year, month, day, hour)
except ValueError:
    st.error("Invalid date")
    st.stop()

uk_holidays = holidays.UnitedKingdom()
dayofweek = dt.weekday()
is_weekend = int(dayofweek >= 5)
is_holiday = int(dt.date() in uk_holidays)

nd_inputs = {
    "hour": hour,
    "day": day,
    "month": month,
    "year": year,
    "dayofweek": dayofweek,
    "is_weekend": is_weekend,
    "is_holiday": is_holiday
}

if st.button("Predict National Demand"):
    with st.spinner("Predicting..."):
        try:
            res = requests.post(ND_API, json=nd_inputs)
            st.success(f"🔌 Demand: {res.json()['prediction']:.2f} MW")
        except Exception as e:
            st.error(f"Backend error: {e}")

# ─────────────────────────────────────────────
# HOUSE PRICE
# ─────────────────────────────────────────────
st.header("🏡 House Price Prediction")

col1, col2 = st.columns(2)
with col1:
    county = st.text_input("County", "BEDFORDSHIRE").upper().strip()
    ptype = st.selectbox("Property Type", [("Detached", "D"), ("Semi-Detached", "S"), ("Terraced", "T"), ("Flat", "F")])
with col2:
    oldnew = st.selectbox("New Build?", [("Yes", "Y"), ("No", "N")])
    duration = st.selectbox("Tenure", [("Freehold", "F"), ("Leasehold", "L")])

year_future = st.number_input("Prediction Year", 2025, 2100, 2030)

hp_inputs = {
    "property_type": ptype[1],
    "oldnew": oldnew[1],
    "duration": duration[1],
    "county": county,
    "year": year_future,
    "month": 1,
    "dayofweek": 0
}

if st.button("Predict House Price"):
    with st.spinner("Predicting..."):
        try:
            res = requests.post(HP_API, json=hp_inputs)
            st.success(f"💷 Predicted Price: £{res.json()['prediction']:,.0f}")
        except Exception as e:
            st.error(f"Backend error: {e}")

st.markdown("---")
st.markdown("Created by CloudAI Project Group 2")

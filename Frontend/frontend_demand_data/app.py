# Frontend/app.py

import streamlit as st
import requests
import json
from datetime import datetime
import importlib
import subprocess
import sys

# -----------------------------
# BACKEND URLs
# -----------------------------
ND_BACKEND_URL = "http://127.0.0.1:8000/predict"
HP_BACKEND_URL = "http://127.0.0.1:8000/predict_house_price"

st.set_page_config(page_title="CloudAI Predictor", page_icon="📈")

st.title("📈 National Demand & 🏡 House Price Predictor")
st.write("Use the trained models to predict national electricity demand or future house prices.")

# -----------------------------
# HOLIDAYS (for ND model)
# -----------------------------
try:
    holidays = importlib.import_module("holidays")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "holidays"])
    holidays = importlib.import_module("holidays")

uk_holidays = holidays.UnitedKingdom()

# ============================================================
# 1️⃣ NATIONAL DEMAND PREDICTION
# ============================================================
st.header("⚡ National Demand Prediction")

nd_inputs = {}

col1, col2, col3, col4 = st.columns(4)
with col1:
    day = st.number_input("Day", value=1, min_value=1, max_value=31, step=1)
with col2:
    month = st.number_input("Month", value=1, min_value=1, max_value=12, step=1)
with col3:
    year = st.number_input("Year", value=2009, min_value=2009, max_value=2025, step=1)
with col4:
    hour = st.number_input("Hour", value=0, min_value=0, max_value=23, step=1)

# Validate date/time
try:
    dt = datetime(int(year), int(month), int(day), int(hour))
except ValueError:
    st.error("❌ Invalid date. Please select a valid combination.")
    st.stop()

dayofweek = dt.weekday()  # 0=Mon
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
is_weekend = 1 if dayofweek in [5, 6] else 0
is_holiday = 1 if dt.date() in uk_holidays else 0

st.markdown("**Auto-computed time features (ND):**")
st.write(f"- Day of week: {dayofweek} ({day_names[dayofweek]})")
st.write(f"- Weekend: {is_weekend}")
st.write(f"- Holiday: {is_holiday}")

# NOTE:
# Your ND model expects the full feature set defined in time_based_feature_names_with_extra.json.
# Here we only send the time-based ones. If your model requires more, you need
# to extend this with additional inputs or default values.
nd_inputs["day"] = int(day)
nd_inputs["month"] = int(month)
nd_inputs["year"] = int(year)
nd_inputs["hour"] = int(hour)
nd_inputs["dayofweek"] = int(dayofweek)
nd_inputs["is_weekend"] = int(is_weekend)
nd_inputs["is_holiday"] = int(is_holiday)

if st.button("Predict National Demand"):
    with st.spinner("Requesting prediction from ND backend..."):
        try:
            resp = requests.post(ND_BACKEND_URL, json=nd_inputs)
            if resp.status_code == 200:
                nd_pred = resp.json()["prediction"]
                st.success(f"**Predicted National Demand:** {nd_pred:,.2f} MW")
            else:
                st.error(f"Backend error: {resp.text}")
        except Exception as e:
            st.error(f"Error contacting backend: {e}")

st.markdown("---")

# ============================================================
# 2️⃣ HOUSE PRICE PREDICTION
# ============================================================
st.header("🏡 House Price Prediction")

st.write(
    "Estimate the future price of a property based on its type, county, "
    "and a future year. This uses the PyCaret-trained model on the UK "
    "Price Paid dataset."
)

hp_inputs = {}

col_hp1, col_hp2 = st.columns(2)
with col_hp1:
    county = st.text_input("County (e.g. BEDFORDSHIRE)", value="BEDFORDSHIRE")
    ptype_label_val = st.selectbox(
        "Property Type",
        options=[
            ("Detached (D)", "D"),
            ("Semi-Detached (S)", "S"),
            ("Terraced (T)", "T"),
            ("Flat/Maisonette (F)", "F")
        ],
        format_func=lambda x: x[0]
    )
with col_hp2:
    oldnew_label_val = st.selectbox(
        "Is it a new build?",
        options=[("New (Y)", "Y"), ("Old (N)", "N")],
        format_func=lambda x: x[0]
    )
    duration_label_val = st.selectbox(
        "Tenure (Duration)",
        options=[("Freehold (F)", "F"), ("Leasehold (L)", "L")],
        format_func=lambda x: x[0]
    )

future_year = st.number_input("Prediction Year", value=2030, min_value=2000, max_value=2100, step=1)
st.caption("For simplicity, the model assumes January (month=1) and Monday (dayofweek=0) for the prediction date.")

hp_inputs["property_type"] = ptype_label_val[1]   # "D","S","T","F"
hp_inputs["oldnew"] = oldnew_label_val[1]         # "Y"/"N"
hp_inputs["duration"] = duration_label_val[1]     # "F"/"L"
hp_inputs["county"] = county.upper().strip()
hp_inputs["year"] = int(future_year)
hp_inputs["month"] = 1
hp_inputs["dayofweek"] = 0

if st.button("Predict House Price"):
    with st.spinner("Requesting prediction from house price backend..."):
        try:
            resp = requests.post(HP_BACKEND_URL, json=hp_inputs)
            if resp.status_code == 200:
                data = resp.json()
                price = data["prediction"]
                st.success(
                    f"**Predicted price in {hp_inputs['county']} for {hp_inputs['year']}:** "
                    f"£{price:,.0f}"
                )
            else:
                st.error(f"Backend error: {resp.text}")
        except Exception as e:
            st.error(f"Error contacting backend: {e}")

st.markdown("---")
st.markdown("Developed by Cloud AI Project Group 2")
st.text("© 2025")

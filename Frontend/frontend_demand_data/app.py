import streamlit as st
import requests
import json
import pandas as pd

# Title
st.title("📈 National Demand Predictor")
st.write("Enter feature values and get a prediction from the model.")

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/predict"   # Change this if backend is remote

# Load feature names so UI builds automatically
with open("../../Backend/backend_demand_data/feature_names.json") as f:
    feature_names = json.load(f)

# Hardcoded selected feature list
feature_names = [
    "settlement_period", "tsd", "england_wales_demand", "embedded_wind_generation",
    "embedded_wind_capacity", "embedded_solar_generation", "embedded_solar_capacity",
    "non_bm_stor", "pump_storage_pumping", "ifa_flow", "ifa2_flow", "britned_flow",
    "moyle_flow", "east_west_flow", "nemo_flow", "hour", "dayofweek", "month",
    "is_weekend", "nd_lag_1", "nd_lag_48", "nd_lag_336", "nd_roll_mean_3", "nd_roll_std_3",
    "nd_roll_mean_6", "nd_roll_std_6", "nd_roll_mean_24", "nd_roll_std_24",
    "nd_roll_mean_48", "nd_roll_std_48", "day", "year", "is_holiday"
]

time_based_features = ['day', 'month', 'year', 'hour']

time_based_features += ['dayofweek', 'is_weekend', 'is_holiday']


import importlib
import subprocess
import sys
try:
    holidays = importlib.import_module("holidays")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "holidays"])
    holidays = importlib.import_module("holidays")
from datetime import datetime

st.subheader("Model Inputs")

inputs = {}

# --- User only selects date + time ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    day = st.number_input("Day", value=1, step=1, min_value=1, max_value=31)

with col2:
    month = st.number_input("Month", value=1, step=1, min_value=1, max_value=12)

with col3:
    year = st.number_input("Year", value=2009, step=1, min_value=2009, max_value=2025)

with col4:
    hour = st.number_input("Hour", value=0, step=1, min_value=0, max_value=23)

# Assemble datetime safely
try:
    dt = datetime(year, month, day, hour)
except ValueError:
    st.error("❌ Invalid date — please select a valid combination.")
    st.stop()

# --- AUTO-COMPUTED FEATURES ---
uk_holidays = holidays.UnitedKingdom()

dayofweek = dt.weekday()                        # 0=Mon, 6=Sun
dayofweeknames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
is_weekend = 1 if dayofweek in [5, 6] else 0
is_holiday = 1 if dt.date() in uk_holidays else 0

# Store all inputs
inputs["day"] = day
inputs["month"] = month
inputs["year"] = year
inputs["hour"] = hour

inputs["dayofweek"] = dayofweek
inputs["is_weekend"] = is_weekend
inputs["is_holiday"] = is_holiday

# Show as read-only for transparency
st.markdown("### Auto-computed features")
st.write(f"**Day of Week:** {dayofweek} ({dayofweeknames[dayofweek]})")
st.write(f"**Weekend:** {is_weekend}")
st.write(f"**Holiday:** {is_holiday}")


# -----------------------------
# Session State initialization
# -----------------------------
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None

if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None

if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# -----------------------------
# Detect if inputs changed
# -----------------------------
if st.session_state.last_inputs is not None and inputs != st.session_state.last_inputs:
    # Inputs changed → move current → previous
    st.session_state.previous_prediction = st.session_state.current_prediction


# Initialize session state
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None
if "current_inputs" not in st.session_state:
    st.session_state.current_inputs = None
if "previous_inputs" not in st.session_state:
    st.session_state.previous_inputs = None

# Predict button
if st.button("Predict"):
    with st.spinner("Requesting prediction..."):
        try:
            response = requests.post(BACKEND_URL, json=inputs)

            if response.status_code == 200:
                result = response.json()

                # Move current → previous
                st.session_state.previous_prediction = st.session_state.current_prediction
                st.session_state.previous_inputs = st.session_state.current_inputs

                # Store new current prediction + inputs
                st.session_state.current_prediction = result["prediction"]
                st.session_state.current_inputs = inputs.copy()

            else:
                st.error(f"Backend error: {response.text}")

        except Exception as e:
            st.error(f"Could not connect to backend: {e}")


# -----------------------------
# Display results with previous/current prediction + delta
# -----------------------------

def format_input_summary(data):
    """Return a readable summary of the 4 time fields."""
    return f"{data['day']:02d}/{data['month']:02d}/{data['year']} at {data['hour']:02d}:00"

# --- Show CURRENT prediction ---
if st.session_state.current_prediction is not None:
    summary = format_input_summary(st.session_state.current_inputs)
    st.success(
        f"**Predicted ND:** {st.session_state.current_prediction:.2f} MW\n"
        f" **(Inputs:** {summary})"
    )

# --- Show PREVIOUS prediction ---
if st.session_state.previous_prediction is not None:
    prev_summary = format_input_summary(st.session_state.previous_inputs)
    st.info(
        f"**Previous ND:** {st.session_state.previous_prediction:.2f} MW\n"
        f" **(Inputs:** {prev_summary})"
    )

# --- Show DELTA between predictions ---
if (
    st.session_state.current_prediction is not None and 
    st.session_state.previous_prediction is not None
):
    delta = st.session_state.current_prediction - st.session_state.previous_prediction
    sign = "+" if delta >= 0 else ""
    st.warning(f"**Change:** {sign}{delta:.2f} MW")




st.markdown("---")
st.markdown("Developed by Cloud AI Project Group 2")
st.text("© 2025")
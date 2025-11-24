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

# Create form
inputs = {}

feature_names = ["settlement_period", "tsd", "england_wales_demand", "embedded_wind_generation", "embedded_wind_capacity", "embedded_solar_generation", "embedded_solar_capacity", "non_bm_stor", "pump_storage_pumping", "ifa_flow", "ifa2_flow", "britned_flow", "moyle_flow", "east_west_flow", "nemo_flow", "hour", "dayofweek", "month", "is_weekend", "nd_lag_1", "nd_lag_48", "nd_lag_336", "nd_roll_mean_3", "nd_roll_std_3", "nd_roll_mean_6", "nd_roll_std_6", "nd_roll_mean_24", "nd_roll_std_24", "nd_roll_mean_48", "nd_roll_std_48", "day", "year", "is_holiday"]

st.subheader("Model Inputs")

for feature in feature_names:
    if feature in ['hour', 'day', 'month', 'year']:
    #if feature in ["settlement_period", "tsd", "england_wales_demand", "embedded_wind_generation", "embedded_wind_capacity", "embedded_solar_generation", "embedded_solar_capacity", "non_bm_stor", "pump_storage_pumping", "ifa_flow", "ifa2_flow", "britned_flow", "moyle_flow", "east_west_flow", "nemo_flow"]:
        # Choose widget depending on type
        if feature.startswith("is_"):
            inputs[feature] = st.selectbox(feature, [0, 1], index=0)
        elif feature == "year":
            inputs[feature] = st.number_input(feature, value=2009, step=1, min_value=2009, max_value=2025)
        elif feature == "day":
            inputs[feature] = st.number_input(feature, value=1, step=1, min_value=1, max_value=31)
        elif feature == "month":
            inputs[feature] = st.number_input(feature, value=1, step=1, min_value=1, max_value=12)
        elif feature == "hour":
            inputs[feature] = st.number_input(feature, value=0, step=1, min_value=0, max_value=23)
        else:
            inputs[feature] = st.number_input(feature, value=0, step=1)

# Button
if st.button("Predict"):
    with st.spinner("Requesting prediction..."):
        try:
            response = requests.post(BACKEND_URL, json=inputs)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted ND: **{result['prediction']:.2f} MW**")
            else:
                st.error(f"Backend error: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")
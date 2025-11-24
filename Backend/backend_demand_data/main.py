import importlib
import subprocess
import sys

# Ensure fastapi is installed and import FastAPI
try:
    fastapi = importlib.import_module("fastapi")
    FastAPI = fastapi.FastAPI
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi"])
    fastapi = importlib.import_module("fastapi")
    FastAPI = fastapi.FastAPI

try:
    lgb = importlib.import_module("lightgbm")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    lgb = importlib.import_module("lightgbm")

import numpy as np

try:
    pd = importlib.import_module("pandas")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    pd = importlib.import_module("pandas")
import json

app = FastAPI(title="National Demand Predictor")

# Load model + features
model = lgb.Booster(model_file="nd_time_model_with_extra.txt")

try:
    joblib = importlib.import_module("joblib")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    joblib = importlib.import_module("joblib")

import os
if os.path.exists("nd_model.pkl"):
    model = joblib.load("nd_model.pkl")

with open("time_based_feature_names_with_extra.json") as f:
    feature_names = json.load(f)

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(features: dict):

    # Convert to DataFrame so order matches training
    df = pd.DataFrame([features], columns=feature_names)
    
    pred = model.predict(df)[0]
    return {"prediction": float(pred)}
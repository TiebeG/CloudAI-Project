# Backend/backend_demand_data/main.py

from fastapi import FastAPI, HTTPException
from pathlib import Path
import os
import json
import sys
import types
import tempfile

import numpy as np
import pandas as pd
import joblib

from pycaret.regression import load_model, predict_model
import pycaret.internal.memory as pcm
from joblib import Memory as JoblibMemory

app = FastAPI(title="National Demand + House Price API")

# ------------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ROOT_ABOVE = PROJECT_ROOT.parent
MODELS_DIR = ROOT_ABOVE / "Models"

# ------------------------------------------------------------------------------------
# NATIONAL DEMAND MODEL (.pkl only)
# ------------------------------------------------------------------------------------

nd_model = None

# You must fix this path to your actual model location
nd_model_path = MODELS_DIR / "lightgbm_nd_time.pkl"

if not nd_model_path.exists():
    raise RuntimeError(f"National demand model not found at: {nd_model_path}")

nd_model = joblib.load(nd_model_path)

ND_FEATURES = ["hour", "day", "month", "year"]

@app.post("/predict_nd")
def predict_nd(features: dict):
    try:
        df = pd.DataFrame([features], columns=ND_FEATURES)
        pred = float(nd_model.predict(df)[0])
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ND prediction failed: {e}")

# ------------------------------------------------------------------------------------
# HOUSE PRICE MODEL (PyCaret with .pkl)
# ------------------------------------------------------------------------------------

house_price_model = None
HOUSE_FEATURES = [
    "property_type", "oldnew", "duration", "towncity", "district", "county",
    "ppdcategory_type", "year", "month", "dayofweek"
]

def load_house_price_model():
    global house_price_model
    if house_price_model is not None:
        return house_price_model

    # Fix pandas index import compatibility for older PyCaret pickles
    if "pandas.core.indexes.numeric" not in sys.modules:
        numeric_mod = types.ModuleType("pandas.core.indexes.numeric")
        numeric_mod.Index = pd.Index
        numeric_mod.Int64Index = pd.Index
        sys.modules["pandas.core.indexes.numeric"] = numeric_mod

    # Patch PyCaret joblib memory cache dir
    def memory_setstate(self, state):
        JoblibMemory.__init__(self, location=tempfile.gettempdir())
    pcm.Memory.__setstate__ = memory_setstate

    # Patch expected FastMemory attributes
    for attr in ["min_time_to_cache", "_cache_counter", "caches_between_reduce"]:
        if not hasattr(pcm.FastMemory, attr):
            setattr(pcm.FastMemory, attr, 0)

    # Adjust this to match your housing model file
    model_path = MODELS_DIR / "housing_prices" / "pycaret_best_housing_model_10_lgbm"
    if not model_path.with_suffix(".pkl").exists():
        raise RuntimeError(f"Housing model file not found at: {model_path}.pkl")

    house_price_model = load_model(str(model_path))
    return house_price_model

@app.post("/predict_house_price")
def predict_house_price(features: dict):
    try:
        model = load_house_price_model()

        required = ["property_type", "oldnew", "duration", "county", "year"]
        for r in required:
            if r not in features:
                raise HTTPException(status_code=400, detail=f"Missing required field: {r}")

        data = {
            "property_type": features["property_type"],
            "oldnew": features["oldnew"],
            "duration": features["duration"],
            "towncity": str(features.get("towncity", "UNKNOWN")).upper(),
            "district": str(features.get("district", "UNKNOWN")).upper(),
            "county": str(features["county"]).upper().strip(),
            "ppdcategory_type": str(features.get("ppdcategory_type", "A")),
            "year": int(features["year"]),
            "month": int(features.get("month", 1)),
            "dayofweek": int(features.get("dayofweek", 0)),
        }

        df = pd.DataFrame([data], columns=HOUSE_FEATURES)
        pred_df = predict_model(model, data=df)

        pred_col = next((c for c in pred_df.columns if c not in df.columns), None)
        if pred_col is None:
            raise RuntimeError("Prediction column not found in model output.")

        prediction = float(pred_df[pred_col].iloc[0])
        return {"prediction": prediction, "used_features": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error in predict_house_price: {e}")

@app.get("/")
def root():
    return {"status": "API running", "endpoints": ["/predict", "/predict_house_price"]}

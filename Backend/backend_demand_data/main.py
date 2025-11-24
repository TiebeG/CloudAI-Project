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
import lightgbm as lgb

app = FastAPI(title="National Demand + House Price API")

# ------------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../Backend/backend_demand_data
PROJECT_ROOT = BASE_DIR.parent                      # .../Backend
ROOT_ABOVE = PROJECT_ROOT.parent                    # .../CloudAI-Project
MODELS_DIR = ROOT_ABOVE / "Models"

# ------------------------------------------------------------------------------------
# NATIONAL DEMAND MODEL
# ------------------------------------------------------------------------------------

# Expect:
#   - nd_time_model_with_extra.txt   (LightGBM Booster)
#   - OR nd_model.pkl                (sklearn model)
#   - time_based_feature_names_with_extra.json

nd_model = None

feature_names_path = BASE_DIR / "time_based_feature_names_with_extra.json"
if not feature_names_path.exists():
    raise RuntimeError(f"ND feature_names JSON not found at: {feature_names_path}")

with open(feature_names_path, "r") as f:
    nd_feature_names = json.load(f)

nd_pkl_path = BASE_DIR / "nd_model.pkl"
nd_lgbm_path = BASE_DIR / "nd_time_model_with_extra.txt"

if nd_pkl_path.exists():
    nd_model = joblib.load(nd_pkl_path)
elif nd_lgbm_path.exists():
    nd_model = lgb.Booster(model_file=str(nd_lgbm_path))
else:
    raise RuntimeError("No ND model found: expected nd_model.pkl or nd_time_model_with_extra.txt")


@app.post("/predict")
def predict_nd(features: dict):
    """
    National Demand prediction.

    Expects JSON with keys matching time_based_feature_names_with_extra.json.
    """

    try:
        df = pd.DataFrame([features], columns=nd_feature_names)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ND features: {e}")

    # LightGBM Booster vs sklearn-like model
    if isinstance(nd_model, lgb.Booster):
        pred = float(nd_model.predict(df)[0])
    else:
        pred = float(nd_model.predict(df)[0])

    return {"prediction": pred}

# ------------------------------------------------------------------------------------
# HOUSE PRICE MODEL (PyCaret) – LAZY LOAD WITH SAFE CACHING
# ------------------------------------------------------------------------------------

house_price_model = None
HOUSE_FEATURES = [
    "property_type",
    "oldnew",
    "duration",
    "towncity",          # NEW
    "district",          # NEW
    "county",
    "ppdcategory_type",  # NEW
    "year",
    "month",
    "dayofweek"
]



def load_house_price_model():
    """
    Lazy-load the PyCaret housing model, with:
    - pandas shim for old pickles
    - joblib cache redirected to a temp dir (avoid C:\\Users\\Admin)
    - FastMemory patched so missing attributes don't crash
    """
    global house_price_model
    if house_price_model is not None:
        return house_price_model

    # ---- pandas compatibility shim (for old PyCaret pickles) ----
    if "pandas.core.indexes.numeric" not in sys.modules:
        numeric_mod = types.ModuleType("pandas.core.indexes.numeric")
        try:
            from pandas import Index, Int64Index
        except ImportError:
            Index = pd.Index
            Int64Index = pd.Index
        numeric_mod.Index = Index
        numeric_mod.Int64Index = Int64Index
        sys.modules["pandas.core.indexes.numeric"] = numeric_mod

    # ---- patch PyCaret's joblib Memory so it uses a temp dir instead of C:\\Users\\Admin ----
    from pycaret.regression import load_model  # noqa: F401
    import pycaret.internal.memory as pcm
    from joblib import Memory as JoblibMemory

    safe_cache_dir = os.path.join(tempfile.gettempdir(), "pycaret_cache")

    def memory_setstate(self, state):
        # Ignore original cache path in pickle; reset to safe temp folder
        JoblibMemory.__init__(self, location=safe_cache_dir)

    # Patch base Memory __setstate__ to avoid using old path in pickle
    pcm.Memory.__setstate__ = memory_setstate

    # Some versions of joblib/PyCaret expect this attribute on FastMemory
    if not hasattr(pcm.FastMemory, "min_time_to_cache"):
        pcm.FastMemory.min_time_to_cache = 0.0

    if not hasattr(pcm.FastMemory, "_cache_counter"):
        pcm.FastMemory._cache_counter = 0

    if not hasattr(pcm.FastMemory, "caches_between_reduce"):
        pcm.FastMemory.caches_between_reduce = 0
    # ---- NOW load the model ----
    housing_models_dir = MODELS_DIR / "housing_prices"

    # ⚠️ ADJUST THIS if your filename is different.
    # If you have ../Models/housing_prices/pycaret_best_housing_model_10_lgbm.pkl
    # then here we pass the *base* name without .pkl:
    model_stem = housing_models_dir / "pycaret_best_housing_model_10_lgbm"

    model_pkl = model_stem.with_suffix(".pkl")
    if not model_pkl.exists():
        raise RuntimeError(f"Housing model file not found at: {model_pkl}")

    house_price_model = load_model(str(model_stem))
    return house_price_model


from pycaret.regression import predict_model  # imported for endpoint use


@app.post("/predict_house_price")
def predict_house_price(features: dict):
    """
    Predict future house price using the PyCaret-trained model.

    Expected JSON body:
    {
      "property_type": "D",
      "oldnew": "N",
      "duration": "F",
      "county": "BEDFORDSHIRE",
      "year": 2030,
      "month": 1,
      "dayofweek": 0
    }
    """
    try:
        # 1) Load model (lazy, with all the shims we already set up)
        model = load_house_price_model()

        # 2) Validate required fields
        required = ["property_type", "oldnew", "duration", "county", "year"]
        for r in required:
            if r not in features:
                raise HTTPException(status_code=400, detail=f"Missing required field: {r}")

        # 3) Build data dict with defaults for missing fields
        data = {
            "property_type": features["property_type"],
            "oldnew": features["oldnew"],
            "duration": features["duration"],
            # Fake / default values for columns the model expects but we don't ask the user for:
            "towncity": str(features.get("towncity", "UNKNOWN")).upper(),
            "district": str(features.get("district", "UNKNOWN")).upper(),
            "county": str(features["county"]).upper().strip(),
            "ppdcategory_type": str(features.get("ppdcategory_type", "A")),  # 'A' is the common code in the dataset
            "year": int(features["year"]),
            "month": int(features.get("month", 1)),
            "dayofweek": int(features.get("dayofweek", 0)),
        }

        import pandas as pd
        df = pd.DataFrame([data], columns=HOUSE_FEATURES)

        # 4) Run PyCaret prediction
        from pycaret.regression import predict_model
        pred_df = predict_model(model, data=df)

        # 5) Detect prediction column name
        extra_cols = [c for c in pred_df.columns if c not in df.columns]
        if not extra_cols:
            raise RuntimeError("Prediction column not found in model output.")
        pred_col = extra_cols[0]

        prediction = float(pred_df[pred_col].iloc[0])

        return {
            "prediction": prediction,
            "used_features": data
        }

    except HTTPException:
        # re-raise HTTPExceptions (400, 500 we explicitly raise)
        raise
    except Exception as e:
        # Surface *real* internal error so we stop guessing
        raise HTTPException(status_code=500, detail=f"Internal error in predict_house_price: {e}")


@app.get("/")
def root():
    return {"status": "API running", "endpoints": ["/predict", "/predict_house_price"]}

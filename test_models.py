import os
import sys
import pickle
import lightgbm as lgb # Needed so Python recognizes the LightGBM object
import pycaret         # Needed so Python recognizes the PyCaret object

# --- CONFIGURATION ---
# Just list the paths to your files here. 
# Both should end in .pkl because that is how they are stored on disk.
MODELS_TO_CHECK = [
    "Models/housing_prices/pycaret_best_housing_model_10_lgbm.pkl",
    "Models/lightgbm_nd_time.pkl"
]

def check_model_health(file_path):
    print(f"\n🔎 INSPECTING: {file_path}")
    
    # 1. Existence Check
    if not os.path.exists(file_path):
        print(f"❌ FAIL: File not found at {file_path}")
        return False
    
    # 2. Integrity Check (Can we unpickle it?)
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            
        model_type = type(model).__name__
        print(f"✅ PASS: File successfully loaded.")
        print(f"   ℹ️  Model Type detected: {model_type}")
        return True
    
    except Exception as e:
        print(f"❌ FAIL: File is corrupted or dependencies missing.")
        print(f"   Error details: {e}")
        return False

if __name__ == "__main__":
    print("--- 🚀 STARTING UNIVERSAL MODEL HEALTH CHECK ---")
    
    all_passed = True
    
    for path in MODELS_TO_CHECK:
        success = check_model_health(path)
        if not success:
            all_passed = False
            
    if all_passed:
        print("\n🎉 SUCCESS: All models are healthy.")
        sys.exit(0) # Pass
    else:
        print("\n💀 FAILURE: One or more models are missing or broken.")
        sys.exit(1) # Fail

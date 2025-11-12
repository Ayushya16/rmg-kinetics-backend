import os, json, joblib
from pathlib import Path
from tensorflow.keras.models import load_model

# Base directories
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"

def load_artifacts():
    """Loads all ML model artifacts (features, scaler, models, metadata)."""

    # --- Define paths ---
    features_path = MODELS_DIR / "features.json"
    scaler_path   = MODELS_DIR / "scaler.save"
    meta_path     = MODELS_DIR / "model_meta.json"

    # --- Load features + meta ---
    if not features_path.exists():
        raise FileNotFoundError("models/features.json not found")

    features = json.loads(features_path.read_text())
    meta     = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # --- Load scaler safely ---
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"⚠️ Warning: could not load scaler ({e}). Proceeding with scaler=None.")
            scaler = None

    # --- Load models ---
    rf_path = MODELS_DIR / "model_rf.pkl"
    xgb_paths = {
        "logA": MODELS_DIR / "model_xgb_logA.pkl",
        "n": MODELS_DIR / "model_xgb_n.pkl",
        "Ea_kJ_per_mol": MODELS_DIR / "model_xgb_Ea_kJ_per_mol.pkl",
    }

    rf_model = joblib.load(rf_path) if rf_path.exists() else None
    xgb_models = {}
    for name, path in xgb_paths.items():
        if path.exists():
            xgb_models[name] = joblib.load(path)

    # --- Load keras model safely ---
    keras_path = MODELS_DIR / "model_nn.keras"
    keras_model = None
    if keras_path.exists():
        try:
            keras_model = load_model(str(keras_path))
            print("✅ Loaded Keras model successfully.")
        except Exception as e:
            print(f"⚠️ Warning: could not load keras model ({e}). Proceeding with keras_model=None.")
            keras_model = None

    # --- Return dictionary of all artifacts ---
    return {
        "features": features,
        "scaler": scaler,
        "meta": meta,
        "rf": rf_model,
        "xgb": xgb_models,
        "nn": keras_model,
    }

# Global preloaded artifacts
ART = load_artifacts()

import os
import json
import joblib
import requests
import zipfile
from io import BytesIO
from pathlib import Path
from tensorflow.keras.models import load_model

# Base directories
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"

# ------------------------------------------------------------
# Ensure models exist: auto-download from Google Drive if missing
# ------------------------------------------------------------
def ensure_models_exist():
    """Downloads models.zip from Google Drive if missing locally."""
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        "model_rf.pkl", "model_xgb_logA.pkl", "model_xgb_n.pkl",
        "model_xgb_Ea_kJ_per_mol.pkl", "model_nn.keras",
        "features.json", "model_meta.json", "scaler.save"
    ]

    # Check if all models exist
    if not all((MODELS_DIR / f).exists() for f in required_files):
        print("📦 Models not found locally — downloading from Google Drive...")

        # ✅ Direct download link (works with Render)
        url = "https://drive.google.com/uc?export=download&id=1HqGyVE5RELSkGChyGlQ6CuGxBwliUERr"

        # Streamed download to avoid corrupted zip files
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise Exception(f"❌ Failed to download models.zip (HTTP {r.status_code})")

        # Extract from in-memory ZIP
        with zipfile.ZipFile(BytesIO(r.content)) as zip_ref:
            zip_ref.extractall(MODELS_DIR)

        print("✅ Models extracted successfully!")

# ------------------------------------------------------------
# Load all artifacts (features, scalers, models, metadata)
# ------------------------------------------------------------
def load_artifacts():
    """Loads all ML model artifacts (features, scaler, models, metadata)."""

    # Ensure model files exist
    ensure_models_exist()

    # Define paths
    features_path = MODELS_DIR / "features.json"
    scaler_path   = MODELS_DIR / "scaler.save"
    meta_path     = MODELS_DIR / "model_meta.json"

    # Load features + metadata
    if not features_path.exists():
        raise FileNotFoundError("models/features.json not found")

    features = json.loads(features_path.read_text())
    meta     = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Load scaler
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"⚠️ Warning: could not load scaler ({e}). Proceeding with scaler=None.")
            scaler = None

    # Load models
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

    # Load Keras model safely
    keras_path = MODELS_DIR / "model_nn.keras"
    keras_model = None
    if keras_path.exists():
        try:
            keras_model = load_model(str(keras_path))
            print("✅ Loaded Keras model successfully.")
        except Exception as e:
            print(f"⚠️ Warning: could not load Keras model ({e}). Proceeding with keras_model=None.")
            keras_model = None

    # Return everything
    return {
        "features": features,
        "scaler": scaler,
        "meta": meta,
        "rf": rf_model,
        "xgb": xgb_models,
        "nn": keras_model,
    }

# ------------------------------------------------------------
# Preload models globally
# ------------------------------------------------------------
ART = load_artifacts()

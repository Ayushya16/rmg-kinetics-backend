import os, json, joblib, requests, zipfile
from io import BytesIO
from pathlib import Path
from tensorflow.keras.models import load_model

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"

def ensure_models_exist():
    """Ensure models exist locally or download from Google Drive (non-blocking)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        "model_rf.pkl", "model_xgb_logA.pkl", "model_xgb_n.pkl",
        "model_xgb_Ea_kJ_per_mol.pkl", "model_nn.keras",
        "features.json", "model_meta.json", "scaler.save"
    ]

    missing = [f for f in required_files if not (MODELS_DIR / f).exists()]
    if not missing:
        print("✅ All required model files are present.")
        return

    print(f"⚠️ Missing files detected: {missing}")
    print("⬇️ Attempting to download models.zip from Google Drive...")

    try:
        url = "https://drive.google.com/uc?export=download&id=1HqGyVE5RELSkGChyGlQ6CuGxBwliUERr"
        r = requests.get(url, timeout=120)

        if r.status_code != 200:
            raise Exception(f"Bad response {r.status_code}")

        with zipfile.ZipFile(BytesIO(r.content)) as z:
            z.extractall(MODELS_DIR)
        print("✅ Models successfully downloaded and extracted.")
    except Exception as e:
        print(f"🚨 Model download failed: {e}")
        print("⚠️ Continuing startup without models (safe mode).")


def load_artifacts():
    """Load models safely (never raises, even if missing)."""
    ensure_models_exist()

    def safe_json_load(path):
        try:
            if path.exists():
                return json.loads(path.read_text())
        except Exception as e:
            print(f"⚠️ Failed to load {path.name}: {e}")
        return {}

    def safe_joblib_load(path):
        try:
            if path.exists():
                return joblib.load(path)
        except Exception as e:
            print(f"⚠️ Failed to load {path.name}: {e}")
        return None

    def safe_keras_load(path):
        try:
            if path.exists():
                return load_model(str(path))
        except Exception as e:
            print(f"⚠️ Failed to load keras model: {e}")
        return None

    print("🔹 Loading artifacts (tolerant mode)...")

    artifacts = {
        "features": safe_json_load(MODELS_DIR / "features.json"),
        "scaler": safe_joblib_load(MODELS_DIR / "scaler.save"),
        "meta": safe_json_load(MODELS_DIR / "model_meta.json"),
        "rf": safe_joblib_load(MODELS_DIR / "model_rf.pkl"),
        "xgb": {
            "logA": safe_joblib_load(MODELS_DIR / "model_xgb_logA.pkl"),
            "n": safe_joblib_load(MODELS_DIR / "model_xgb_n.pkl"),
            "Ea_kJ_per_mol": safe_joblib_load(MODELS_DIR / "model_xgb_Ea_kJ_per_mol.pkl"),
        },
        "nn": safe_keras_load(MODELS_DIR / "model_nn.keras"),
    }

    print("✅ Safe model loading complete.")
    return artifacts


try:
    ART = load_artifacts()
except Exception as e:
    print(f"🚨 Non-fatal error while loading artifacts: {e}")
    ART = {}

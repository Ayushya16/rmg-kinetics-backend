from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, os

from model_loader import ART, load_artifacts
from featurizer import build_feature_vector_from_row
from schemas import PredictByFeatures, PredictResponse
from auth import check_api_key

# -----------------------------------------------------------
# FastAPI App Configuration
# -----------------------------------------------------------
app = FastAPI(title="RMG Kinetics Predictor", version="1.0")

# --- Allow frontend connections (CORS setup) ---
origins = os.getenv("CORS_ORIGINS", "*")
allow_origins = ["*"] if origins == "*" else origins.split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": bool(ART.get("rf"))}


@app.post("/predict/features", response_model=PredictResponse, dependencies=[Depends(check_api_key)])
async def predict_by_features(payload: PredictByFeatures):
    try:
        # Build feature vector first (no scaling yet)
        x = build_feature_vector_from_row(payload.features, ART["features"], scaler=None)
        print("✅ Feature vector created with shape:", x.shape)

        # Apply scaler only if compatible
        if ART["scaler"] is not None:
            try:
                trained_n_features = getattr(ART["scaler"], "n_features_in_", None)
                if trained_n_features == x.shape[1]:
                    x = ART["scaler"].transform(x)
                    print("✅ Scaler applied successfully.")
                else:
                    print(f"⚠️ Skipping scaler — mismatch ({x.shape[1]} input vs {trained_n_features} trained).")
            except Exception as e:
                print(f"⚠️ Skipping scaler due to error: {e}")

    except Exception as e:
        print("❌ Error building features:", e)
        raise HTTPException(status_code=400, detail=str(e))

    model_used = "Unknown"
    pred = [0, 0, 0]

    # Try RandomForest first
    try:
        if ART["rf"] is not None:
            pred = ART["rf"].predict(x)[0]
            model_used = "RandomForest"
            print("✅ Prediction successful using RandomForest.")
        # Else fallback to XGBoost ensemble
        elif len(ART["xgb"]) > 0:
            preds = []
            for m in ART["xgb"].values():
                preds.append(m.predict(x))
            pred = np.mean(preds, axis=0)
            model_used = "XGBoost Ensemble"
            print("✅ Prediction successful using XGBoost Ensemble.")
        else:
            print("⚠️ No valid model found in ART.")
            raise HTTPException(status_code=500, detail="No trained models loaded.")
    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))

    # --- Return response ---
    return {
        "A": float(10 ** pred[0]),
        "n": float(pred[1]),
        "Ea_kJ_per_mol": float(pred[2]),
        "model_used": model_used,
        "meta": ART["meta"],
    }


@app.post("/reload")
async def reload_models():
    global ART
    ART = load_artifacts()
    print("🔄 Models reloaded successfully.")
    return {"status": "reloaded"}
@app.get("/meta")
async def meta():
    """Returns model metadata and expected feature count."""
    try:
        expected_features = len(ART["features"]) if ART.get("features") is not None else 0
        return {
            "expected_features": expected_features,
            "models": ART["meta"]["models"] if ART.get("meta") else {}
        }
    except Exception as e:
        return {"error": str(e)}

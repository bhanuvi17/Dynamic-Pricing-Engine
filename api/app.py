"""
FastAPI Backend for Dynamic Pricing System
Run: uvicorn api.app:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import DataPreprocessor
from src.surge_pricing import SurgePricingEngine
from src.prediction import PricingPredictor

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Dynamic Pricing API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

try:
    predictor = PricingPredictor(os.path.join(BASE_DIR, 'models/model.pkl'))
    print("✅ XGBoost model loaded")

    preprocessor = DataPreprocessor()
    preprocessor.load_scaler(os.path.join(BASE_DIR, 'models/scaler.pkl'))
    print("✅ Scaler loaded")

    with open(os.path.join(BASE_DIR, 'models/surge_engine.pkl'), 'rb') as f:
        surge_engine = pickle.load(f)
    print("✅ Surge engine loaded")

    print("✅ All artifacts loaded successfully")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    print(traceback.format_exc())
    raise

# ── Mount Static UI ────────────────────────────────────────────────────────────
ui_dir = os.path.join(BASE_DIR, 'ui')
if os.path.exists(ui_dir):
    app.mount("/static", StaticFiles(directory=ui_dir), name="static")


# ── Request Model ──────────────────────────────────────────────────────────────
class RideRequest(BaseModel):
    Number_of_Riders: int
    Number_of_Drivers: int
    Location_Category: str          # Urban | Suburban | Rural
    Customer_Loyalty_Status: str    # Silver | Regular | Gold
    Number_of_Past_Rides: int
    Time_of_Booking: str            # Morning | Evening | Night | Afternoon
    Vehicle_Type: str               # Economy | Premium
    Expected_Ride_Duration: int
    Historical_Cost_of_Ride: float


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    index = os.path.join(BASE_DIR, 'ui', 'index.html')
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Dynamic Pricing API v2.0", "status": "running",
            "docs": "/docs"}


@app.post("/predict")
def predict_price(request: RideRequest):
    """Predict ride price: ML predicts base price, then surge multiplier is applied"""
    try:
        data = request.dict()

        # 1. Calculate surge multiplier from demand/supply signals
        _, surge_multiplier = surge_engine.calculate_single_surge(
            riders=data['Number_of_Riders'],
            drivers=data['Number_of_Drivers'],
            past_rides=data['Number_of_Past_Rides'],
            duration=data['Expected_Ride_Duration'],
            vehicle_type=data['Vehicle_Type'],
            location=data['Location_Category'],
            time_booking=data['Time_of_Booking'],
            base_price=data['Historical_Cost_of_Ride']
        )

        # 2. ML feature preparation
        features_df = predictor.prepare_features(data)

        # 3. Scale features
        features_scaled = preprocessor.transform(features_df)

        # 4. ML predicts the base price (trained on Historical_Cost_of_Ride patterns)
        ml_base = float(predictor.predict(features_scaled)[0])

        # 5. Apply surge multiplier on top of ML-predicted base price
        ml_surge_price = ml_base * surge_multiplier

        # 6. Surge-formula price (for breakdown reference: historical base × multiplier)
        surge_adjusted_price = data['Historical_Cost_of_Ride'] * surge_multiplier

        percent_change = ((ml_surge_price - data['Historical_Cost_of_Ride']) /
                          data['Historical_Cost_of_Ride']) * 100

        return {
            "success": True,
            "base_price": round(data['Historical_Cost_of_Ride'], 2),
            "ml_base_price": round(ml_base, 2),
            "surge_multiplier": round(surge_multiplier, 3),
            "surge_adjusted_price": round(surge_adjusted_price, 2),
            "ml_surge_price": round(ml_surge_price, 2),
            "percent_change": round(percent_change, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}\n{traceback.format_exc()}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "XGBoost", "artifacts_loaded": True}


@app.get("/debug")
def debug_info():
    return {
        "cwd": os.getcwd(),
        "base_dir": BASE_DIR,
        "model_exists": os.path.exists(os.path.join(BASE_DIR, 'models/model.pkl')),
        "scaler_exists": os.path.exists(os.path.join(BASE_DIR, 'models/scaler.pkl')),
        "surge_exists": os.path.exists(os.path.join(BASE_DIR, 'models/surge_engine.pkl')),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
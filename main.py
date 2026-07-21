import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="BakuLink AI Microservice",
    description="Service AI untuk Price Forecasting, Demand Forecasting, & Procurement Advisor BakuLink",
    version="1.0.0"
)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# 1. Price Forecasting Request Model
# ----------------------------------------------------------------------
class PricePredictRequest(BaseModel):
    commodity: str = "Tepung Terigu"
    days: int = 7
    total_local_stock_kg: float = 500.0

# ----------------------------------------------------------------------
# 2. Demand Forecasting Request Model
# ----------------------------------------------------------------------
class DemandPredictRequest(BaseModel):
    commodity: str = "Tepung Terigu"
    category_pattern: str = "rendah"  # pilihan: rendah, musiman, fluktuatif
    unit_price: float = 11000.0
    is_high_season: int = 1
    day_of_week: int = 2
    month: int = 7
    recent_sales_history: list[float] = Field(
        default_factory=lambda: [500.0, 480.0, 520.0, 510.0, 490.0, 505.0, 515.0]
    )

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "BakuLink AI Microservice",
        "endpoints": ["/predict-price", "/predict-demand"]
    }

# ----------------------------------------------------------------------
# ENDPOINT: POST /predict-price
# ----------------------------------------------------------------------
@app.post("/predict-price")
def predict_price(req: PricePredictRequest):
    formatted_name = req.commodity.lower().strip().replace(" ", "_")
    model_path = os.path.join("models", f"model_price_{formatted_name}.pkl")
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model peramalan harga untuk komoditas '{req.commodity}' tidak ditemukan di folder models/"
        )
        
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memuat model price: {str(e)}")

    future = model.make_future_dataframe(periods=req.days)
    forecast = model.predict(future)
    
    base_recommended = float(forecast['yhat'].iloc[-1])
    
    THRESHOLD_STOK = 1000.0
    scarcity_delta = 0.042 if req.total_local_stock_kg < THRESHOLD_STOK else 0.0
    final_recommended = int(base_recommended * (1 + scarcity_delta))
    
    return {
        "status": "success",
        "commodity": req.commodity,
        "days_ahead": req.days,
        "base_forecast_price": int(base_recommended),
        "stock_scarcity_applied": scarcity_delta > 0,
        "scarcity_percentage": f"{scarcity_delta * 100:.1f}%",
        "final_recommended_price": final_recommended
    }

# ----------------------------------------------------------------------
# Helper: Find Model File in Various Directory Naming Conventions
# ----------------------------------------------------------------------
def locate_demand_model(pattern_name: str):
    pattern_clean = pattern_name.lower().strip()
    possible_dirs = [
        os.path.join("models", "model demand forecasting"),
        os.path.join("models", "Model Demand Forecasting"),
        "models"
    ]
    
    filename = f"model_demand_{pattern_clean}.pkl"
    
    for dir_path in possible_dirs:
        candidate = os.path.join(dir_path, filename)
        if os.path.exists(candidate):
            return candidate
    return None

# ----------------------------------------------------------------------
# ENDPOINT: POST /predict-demand
# ----------------------------------------------------------------------
@app.post("/predict-demand")
def predict_demand(req: DemandPredictRequest):
    pattern = req.category_pattern.lower().strip()
    model_path = locate_demand_model(pattern)
    
    daily_pred = None
    method_used = ""
    
    # Attempt loading Scikit-Learn .pkl model
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # Predict using dataframe with features: unit_price, is_high_season, day_of_week, month
            X_input = pd.DataFrame(
                [[req.unit_price, req.is_high_season, req.day_of_week, req.month]],
                columns=['unit_price', 'is_high_season', 'day_of_week', 'month']
            )
            raw_pred = model.predict(X_input)
            daily_pred = float(raw_pred[0])
            method_used = f"Random Forest ({pattern.capitalize()})"
        except Exception as e:
            print(f"Warning: Exception loading/predicting with model {model_path}: {e}")
            daily_pred = None

    # Fallback to Simple Moving Average (SMA) if model not found or failed
    if daily_pred is None:
        if req.recent_sales_history and len(req.recent_sales_history) > 0:
            daily_pred = float(np.mean(req.recent_sales_history))
        else:
            daily_pred = 500.0 # Standard fallback default
        method_used = f"Simple Moving Average (Fallback - {pattern.capitalize()})"

    # Calculate 7-day demand prediction
    predicted_7days_kg = int(round(daily_pred * 7))
    predicted_daily_kg = round(daily_pred, 1)
    
    # Historical recent sales average for comparison
    recent_avg = float(np.mean(req.recent_sales_history)) if req.recent_sales_history else daily_pred
    
    # Calculate percentage change compared to historical average
    if recent_avg > 0:
        pct_change = round(((daily_pred - recent_avg) / recent_avg) * 100, 1)
    else:
        pct_change = 0.0

    alert_triggered = abs(pct_change) >= 10.0

    # Determine safety buffer based on pattern
    if pattern == "musiman":
        safety_buffer_pct = 15
    elif pattern in ["fluktuatif", "lumpy", "erratic"]:
        safety_buffer_pct = 20
    else:
        safety_buffer_pct = 10  # default for "rendah" / "smooth"

    target_stock_kg = int(round(predicted_7days_kg * (1 + safety_buffer_pct / 100.0)))

    # Dynamic Pricing Intelligence Range Calculation
    demand_factor = pct_change / 100.0
    markup_min = 0.05
    markup_max = 0.15
    
    rec_min = int(round(req.unit_price * (1 + max(0, demand_factor * markup_min))))
    rec_max = int(round(req.unit_price * (1 + max(0, demand_factor * markup_max))))
    
    # Ensure recommendation isn't below unit price
    rec_min = max(int(req.unit_price), rec_min)
    rec_max = max(rec_min, rec_max)

    return {
        "status": "success",
        "commodity": req.commodity,
        "method_used": method_used,
        "predicted_demand_daily_kg": predicted_daily_kg,
        "predicted_demand_7days_kg": predicted_7days_kg,
        "alert_triggered": alert_triggered,
        "pct_change": pct_change,
        "inventory_advice": {
            "target_stock_kg": target_stock_kg,
            "safety_buffer_pct": safety_buffer_pct
        },
        "pricing_intelligence": {
            "recommended_price_min": rec_min,
            "recommended_price_max": rec_max
        }
    }
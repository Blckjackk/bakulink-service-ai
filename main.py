import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prophet import Prophet

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="BakuLink AI Service",
    description="Service AI untuk Price Forecasting, Demand Forecasting, & Procurement Advisor",
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

# Model data input request
class PricePredictRequest(BaseModel):
    commodity: str                  # Contoh: "Tepung Terigu", "Minyak Goreng", "Gula Pasir"
    days: int = 7                   # Berapa hari ke depan yang mau diprediksi
    total_local_stock_kg: float = 500.0  # Total stok lokal saat ini di platform

@app.get("/")
def root():
    return {"message": "BakuLink AI Service is Running!"}

@app.post("/predict-price")
def predict_price(req: PricePredictRequest):
    # 1. Format nama file .pkl dari input komoditas
    # Contoh: "Tepung Terigu" -> "model_price_tepung_terigu.pkl"
    formatted_name = req.commodity.lower().strip().replace(" ", "_")
    model_path = os.path.join("models", f"model_price_{formatted_name}.pkl")
    
    # Cek apakah file model ada
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model untuk komoditas '{req.commodity}' tidak ditemukan di folder models/"
        )
        
    # 2. Load Model .pkl
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memuat model: {str(e)}")

    # 3. Prophet Prediction
    future = model.make_future_dataframe(periods=req.days)
    forecast = model.predict(future)
    
    base_recommended = float(forecast['yhat'].iloc[-1])
    
    # 4. Supply-Demand Adjustment Factor
    # Jika stok lokal platform < 1000 kg, berikan penyesuaian kelangkaan +4.2%
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
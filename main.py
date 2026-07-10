from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="BakuLink AI Service", description="API Service untuk peramalan harga pangan dan asisten pengadaan AI BakuLink")

# Pintu 1: Tes koneksi
@app.get("/")
def home():
    return {"message": "Server AI BakuLink Aktif!"}

# Pintu 2: Menerima request dari Laravel untuk Prediksi Harga
@app.post("/predict-price")
def predict_price(commodity: str):
    # Nanti di sini kode untuk me-load file .pkl hasil Google Colab
    # Contoh hasil tebakan AI:
    recommended_price = 14500 
    return {
        "status": "success",
        "commodity": commodity,
        "recommended_price": recommended_price
    }

# Catatan AI 2: Demand Forecasting (Peramalan Permintaan Pasar) 📦

Dokumen ini menjelaskan rancangan, spesifikasi data, dan langkah demi langkah pelatihan model AI untuk meramal volume permintaan komoditas oleh pembeli di BakuLink.

---

## 📝 1. Deskripsi AI
AI Demand Forecasting bertugas memprediksi kuantitas/volume kebutuhan pasar terhadap suatu komoditas pangan untuk periode mendatang (minggu/bulan depan). Fitur ini membantu para Supplier dan Petani (UMKM produsen) agar dapat mempersiapkan jumlah pasokan barang di gudang dengan pas, menghindari risiko barang busuk karena kelebihan stok, atau kehilangan penjualan karena kekurangan stok.

*   **Tipe AI:** Regression & Time Series Forecasting.
*   **Algoritma Rekomendasi:** **Random Forest Regressor** (jika memprediksi volume berdasarkan multi-variabel seperti tren historis, hari raya, dan harga) ATAU **SARIMAX** (jika memprediksi murni berdasarkan deret waktu historis volume transaksi).

---

## 📊 2. Spesifikasi Data untuk Training
Model peramalan permintaan dilatih menggunakan data agregasi transaksi bulanan atau mingguan dari internal platform BakuLink.

*   **Sumber Data:** Database transaksi penjualan internal BakuLink (tabel `orders` dan `order_items`).
*   **Kebutuhan Data Historis:** Data rekam transaksi minimal **1-2 tahun ke belakang** (dapat menggunakan data tiruan/mock transaction data untuk kebutuhan lomba).
*   **Struktur Dataset (Format CSV):**
    Dataset harus merangkum total penjualan mingguan/bulanan per komoditas:
    
    | Nama Kolom | Tipe Data | Deskripsi | Contoh Isi |
    | :--- | :--- | :--- | :--- |
    | `tanggal` | Date | Tanggal akhir periode pencatatan (format YYYY-MM-DD) | `2026-07-10` |
    | `komoditas` | Text | Jenis komoditas pangan | `Beras` |
    | `total_harga` | Numeric | Harga rata-rata komoditas pada periode itu | `14200` |
    | `is_high_season` | Binary | Indikator bulan perayaan (1 jika Lebaran/Natal, 0 jika biasa) | `1` |
    | `volume_permintaan` | Numeric | **Target Variable**: Total kuantitas barang yang terjual (dalam kg) | `5200` |

---

## 💻 3. Langkah demi Langkah Pelatihan di Google Colab

### Langkah 3.1: Buat Notebook Baru & Import Library
Buka Google Colab, buat notebook baru, lalu import library Machine Learning standar:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
```

### Langkah 3.2: Load & Siapkan Data
Unggah file dataset penjualan Anda (misalnya `volume_penjualan_historis.csv`) lalu muat ke DataFrame:
```python
# Load dataset
df = pd.read_csv('volume_penjualan_historis.csv')

# Pisahkan Fitur (X) dan Target yang ingin ditebak (y)
# Di sini kita menebak 'volume_permintaan' berdasarkan 'total_harga' dan 'is_high_season'
X = df[['total_harga', 'is_high_season']]
y = df['volume_permintaan']

# Bagi data menjadi 80% untuk latihan (train) dan 20% untuk pengujian (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data berhasil dibagi!")
```

### Langkah 3.3: Latih Model Random Forest
```python
# Inisialisasi model Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Latih model
model.fit(X_train, y_train)

# Uji performa model
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print(f"Model berhasil dilatih! Nilai rata-rata error prediksi: {error:.2f} kg")
```

### Langkah 3.4: Ekspor Model Terlatih (Hasil Akhir)
Simpan model terlatih ke berkas `.pkl` untuk diunduh:
```python
# Simpan model
with open('model_demand_beras.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Berkas model_demand_beras.pkl siap diunduh!")
```

---

## 💾 4. Cara Penggunaan File Model di FastAPI

1.  Unduh file **`model_demand_beras.pkl`** dari Google Colab.
2.  Pindahkan file tersebut ke folder proyek Anda di:
    `bakulink-service-ai/models/model_demand_beras.pkl`
3.  Di file `main.py` FastAPI, buat endpoint baru `/predict-demand`:

```python
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Muat model demand
with open("models/model_demand_beras.pkl", "rb") as f:
    model_demand = pickle.load(f)

class DemandRequest(BaseModel):
    commodity: str
    current_price: int
    is_holiday: int  # 1 jika ya, 0 jika tidak

@app.post("/predict-demand")
def predict_demand(request: DemandRequest):
    if request.commodity.lower() == "beras":
        # Susun input data sesuai urutan fitur saat training
        input_data = np.array([[request.current_price, request.is_holiday]])
        
        # Lakukan prediksi volume permintaan
        predicted_volume = int(model_demand.predict(input_data)[0])
        
        return {
            "status": "success",
            "commodity": request.commodity,
            "predicted_demand_volume_kg": predicted_volume
        }
    
    return {
        "status": "fallback",
        "commodity": request.commodity,
        "predicted_demand_volume_kg": 1000  # Default dummy volume
    }
```

# Catatan AI 1: Price Forecasting (Peramalan Harga Pangan) 📈

Dokumen ini menjelaskan rancangan, spesifikasi data, matematika penyesuaian faktor stok, dan langkah demi langkah pelatihan model AI untuk meramal harga komoditas pangan di BakuLink.

---

## 📝 1. Deskripsi AI & Mekanisme Peramalan

AI Price Forecasting bertugas memproyeksikan harga komoditas pangan (seperti beras, cabai, bawang, minyak goreng, dll.) untuk **7 hingga 30 hari ke depan**. Fitur ini membantu UMKM membeli komoditas di waktu termurah, dan membantu Supplier menentukan strategi harga jual yang kompetitif.

### Arsitektur Peramalan Hybrid (3 Pilar Data):
1. **Data Internal Platform:** Rata-rata harga riil dari transaksi yang berhasil diselesaikan di BakuLink.
2. **Data Eksternal (Web Scraping / API Govt):** Memantau harga pangan harian dari portal resmi **PIHPS Bank Indonesia** (`hargapangan.id`) atau **Panel Harga Bapanas** (`panelharga.badanpangan.go.id`).
3. **Faktor Pasokan Supplier (Supply-Demand Adjustment Engine):**
   - AI membaca total stok komoditas yang dimiliki seluruh supplier di platform.
   - Jika terjadi kelangkaan pasokan (stok menipis sedangkan permintaan naik), AI menyesuaikan nilai ramalan dengan kenaikan eksplosif (misalnya $+4.2\%$ di atas tren linier).

*   **Tipe AI:** Time Series Forecasting (Deret Waktu).
*   **Algoritma Utama:** **Meta Prophet** (tangguh terhadap data hilang & menangani tren keagamaan/musim liburan otomatis) atau **SARIMAX**.

---

## 🧮 2. Formulasi Matematika Penyesuaian Harga

### A. Rata-Rata Harga Pasar Mingguan (Internal Data Aggregation)
$$\text{Harga Rata-Rata}_t = \frac{\sum_{i=1}^{K} \text{Harga Transaksi}_i}{K}$$
*di mana $K$ adalah jumlah transaksi komoditas pada periode $t$.*

### B. Proyeksi Penyesuaian Faktor Stok Supplier (Supply-Demand Factor)
$$\text{Harga Forecast Akhir} = \text{Harga Base Prophet} \times (1 + \Delta_{\text{scarcity}})$$

$$\Delta_{\text{scarcity}} = \begin{cases} 
+0.042 \ (+4.2\%), & \text{jika } \text{Total Stok Platform} < \text{Threshold Kritis Stok} \\
0, & \text{jika } \text{Total Stok Normal}
\end{cases}$$

---

## 📊 3. Spesifikasi Data untuk Training

Untuk melatih model Prophet, dibutuhkan data historis harga pangan harian.
* **Sumber Data:** Portal PIHPS Bank Indonesia atau Panel Harga Bapanas.
* **Kebutuhan Data Historis:** Minimal **3 tahun ke belakang** agar model mengenali pola inflasi tahunan & hari besar keagamaan.
* **Struktur Dataset (CSV):**

| Nama Kolom | Tipe Data | Deskripsi | Contoh Isi |
| :--- | :--- | :--- | :--- |
| `ds` (Date Stamp) | Date/Text | Tanggal pencatatan harga (YYYY-MM-DD) | `2026-07-10` |
| `y` (Target Value) | Numeric | Harga rata-rata komoditas (Rupiah) | `14500` |

---

## 💻 4. Langkah demi Langkah Pelatihan di Google Colab

### Langkah 4.1: Buat Notebook Baru & Install Library
```python
!pip install prophet
```

### Langkah 4.2: Unggah Dataset & Muat Data
```python
import pandas as pd
from prophet import Prophet
import pickle

# Load dataset harga historis
df = pd.read_csv('harga_beras_historis.csv')
df['ds'] = pd.to_datetime(df['ds'])
print(df.head())
```

### Langkah 4.3: Inisialisasi & Latih Model Prophet
```python
# Inisialisasi model Prophet
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# Tambahkan hari libur nasional Indonesia
model.add_country_holidays(country_name='ID')

# Latih model
model.fit(df)
print("Model Price Forecasting berhasil dilatih!")
```

### Langkah 4.4: Lakukan Prediksi & Ekspor Model
```python
# Buat dataframe 30 hari ke depan
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Tampilkan grafik
model.plot(forecast)

# Ekspor model ke .pkl
with open('model_price_beras.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Berkas model_price_beras.pkl siap diunduh!")
```

---

## 💾 5. Integration di FastAPI (`main.py`)

```python
import pickle
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

with open("models/model_price_beras.pkl", "rb") as f:
    model_price_beras = pickle.load(f)

@app.post("/predict-price")
def predict_price(commodity: str, days: int = 7, total_local_stock_kg: float = 500):
    if commodity.lower() == "beras":
        future = model_price_beras.make_future_dataframe(periods=days)
        forecast = model_price_beras.predict(future)
        
        base_recommended = float(forecast['yhat'].iloc[-1])
        
        # Apply Supply Factor Logic
        scarcity_delta = 0.042 if total_local_stock_kg < 1000 else 0.0
        final_recommended = int(base_recommended * (1 + scarcity_delta))
        
        return {
            "status": "success",
            "commodity": commodity,
            "base_forecast_price": int(base_recommended),
            "stock_scarcity_applied": scarcity_delta > 0,
            "final_recommended_price": final_recommended
        }
```

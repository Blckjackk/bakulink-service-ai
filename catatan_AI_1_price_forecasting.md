# Catatan AI 1: Price Forecasting (Peramalan Harga Pangan) 📈

Dokumen ini menjelaskan rancangan, spesifikasi data, dan langkah demi langkah pelatihan model AI untuk meramal harga komoditas pangan di BakuLink.

---

## 📝 1. Deskripsi AI
AI Price Forecasting bertugas memproyeksikan harga komoditas pangan (seperti beras, cabai, bawang, minyak goreng, dll.) untuk 7 hingga 30 hari ke depan. Fitur ini membantu UMKM membeli komoditas di waktu termurah, dan membantu Supplier menentukan strategi harga jual yang kompetitif.

*   **Tipe AI:** Time Series Forecasting (Deret Waktu).
*   **Algoritma Rekomendasi:** **Prophet (dari Meta)** karena sangat tangguh menghadapi data kosong (missing values) dan dapat menangani efek hari libur/keagamaan (Lebaran, Natal) secara otomatis.

---

## 📊 2. Spesifikasi Data untuk Training
Untuk melatih model ini, kita membutuhkan data historis harga pangan harian.
*   **Sumber Data:** Data portal PIHPS Bank Indonesia (`hargapangan.id`) atau Panel Harga Badan Pangan Nasional (`panelharga.badanpangan.go.id`).
*   **Kebutuhan Data Historis:** Minimal **3 tahun ke belakang** (contoh: data dari Januari 2023 - Juli 2026) agar model bisa mengenali pola inflasi tahunan.
*   **Struktur Dataset (Format CSV):**
    Dataset harus memiliki minimal dua kolom dengan format berikut:
    
    | Nama Kolom | Tipe Data | Deskripsi | Contoh Isi |
    | :--- | :--- | :--- | :--- |
    | `ds` (Date Stamp) | Date/Text | Tanggal pencatatan harga (format YYYY-MM-DD) | `2026-07-10` |
    | `y` (Target Value) | Numeric | Harga rata-rata komoditas pada tanggal tersebut | `14500` |

---

## 💻 3. Langkah demi Langkah Pelatihan di Google Colab

Berikut adalah panduan praktis pengerjaannya di Google Colab:

### Langkah 3.1: Buat Notebook Baru & Install Library
Buka [Google Colab](https://colab.research.google.com/), buat notebook baru, lalu install library Meta Prophet:
```python
!pip install prophet
```

### Langkah 3.2: Unggah Dataset & Muat Data
Unggah file dataset CSV Anda (misalnya `harga_beras_historis.csv`) ke panel file Colab, lalu ketik kode berikut:
```python
import pandas as pd
from prophet import Prophet

# Load dataset
df = pd.read_csv('harga_beras_historis.csv')

# Pastikan kolom ds bertipe datetime
df['ds'] = pd.to_datetime(df['ds'])
print(df.head())
```

### Langkah 3.3: Inisialisasi & Latih Model
Latih model Prophet menggunakan data historis tersebut:
```python
# Inisialisasi model Prophet
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# Tambahkan hari libur nasional Indonesia (Opsional tapi direkomendasikan)
model.add_country_holidays(country_name='ID')

# Latih model
model.fit(df)
print("Model berhasil dilatih!")
```

### Langkah 3.4: Lakukan Prediksi Masa Depan (Forecasting)
Coba lakukan peramalan untuk 30 hari ke depan:
```python
# Buat dataframe kosong untuk 30 hari ke depan
future = model.make_future_dataframe(periods=30)

# Lakukan prediksi
forecast = model.predict(future)

# Tampilkan hasil prediksi komponen (tren, mingguan, tahunan)
model.plot(forecast)
```

### Langkah 3.5: Ekspor Model Terlatih (Hasil Akhir)
Simpan model yang sudah pintar ini ke dalam file `.pkl` agar bisa diunduh:
```python
import pickle

# Simpan model ke file
with open('model_price_beras.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Berkas model_price_beras.pkl siap diunduh!")
```

---

## 💾 4. Cara Penggunaan File Model di FastAPI

1.  Unduh file **`model_price_beras.pkl`** dari Google Colab ke komputer Anda.
2.  Pindahkan file tersebut ke folder proyek Anda di:
    `bakulink-service-ai/models/model_price_beras.pkl`
3.  Di file `main.py` FastAPI, muat model tersebut seperti ini:

```python
import pickle
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# Muat model saat startup
with open("models/model_price_beras.pkl", "rb") as f:
    model_beras = pickle.load(f)

@app.post("/predict-price")
def predict_price(commodity: str, days: int = 7):
    if commodity.lower() == "beras":
        # Buat dataframe tanggal masa depan
        future = model_beras.make_future_dataframe(periods=days)
        forecast = model_beras.predict(future)
        
        # Ambil harga rekomendasi hari terakhir hasil ramalan
        recommended = int(forecast['yhat'].iloc[-1])
        
        return {
            "status": "success",
            "commodity": commodity,
            "recommended_price": recommended
        }
```

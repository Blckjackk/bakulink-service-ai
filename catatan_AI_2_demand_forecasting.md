# Catatan AI 2: Demand Forecasting & Supply Intelligence (Peramalan Permintaan Pasar) 📦

Dokumen ini menjelaskan rancangan arsitektur, landasan rumus matematika, spesifikasi data, serta **panduan praktis langkah demi langkah pelatihan data di Python/Google Colab** untuk model peramalan permintaan pasar (Demand Forecasting) dan rekomendasi pasokan/harga di BakuLink.

---

## 📝 1. Deskripsi AI & Strategi Algoritma

AI Demand Forecasting bertugas memprediksi kuantitas/volume kebutuhan pasar terhadap suatu komoditas pangan untuk periode mendatang (harian/mingguan/bulanan). Fitur ini memberikan **Rekomendasi Stok Gudang** dan **Rekomendasi Harga Jual** bagi Verified Supplier untuk mencegah *stockout* maupun *dead stock*.

### Pendekatan Multi-Metode Berdasarkan Pola Data:
1. **Permintaan Kontinu & Smooth:** Menggunakan **Holt-Winters Generalized** (Mode Stabil/SES jika tidak ada tren, Mode Penuh jika ada tren & musiman).
2. **Permintaan Intermiten / Sporadis (Khas B2B):** Menggunakan **Croston's Method** (memisahkan ukuran order dan interval antar order).
3. **Pola Kompleks Multi-Variabel:** Menggunakan **Random Forest Regressor** atau **SARIMAX** (memperhitungkan variabel harga, musim perayaan/high season, dan tren).

---

## 🧮 2. Landasan Rumus Matematika & Logika Komputasi

### A. Klasifikasi Pola Demand (Syntetos - Boylan)
Sebelum memilih metode forecasting, sistem secara otomatis mengklasifikasikan pola historis setiap komoditas:

1. **Average Demand Interval (ADI):**
   $$\text{ADI} = \frac{N}{K}$$
   * $N$: Total jumlah periode dalam data historis.
   * $K$: Jumlah periode yang memiliki transaksi order ($\text{Demand} > 0$).

2. **Squared Coefficient of Variation ($CV^2$):**
   $$\text{CV}^2 = \left( \frac{\sigma_Y}{\mu_Y} \right)^2$$
   * $\sigma_Y$ & $\mu_Y$: Standar deviasi dan rata-rata kuantitas order (hanya dari periode non-zero).

**Matriks Keputusan Klasifikasi:**

| Kondisi | Klasifikasi | Metode Utama yang Diterapkan |
|---|---|---|
| $\text{ADI} < 1.32 \land \text{CV}^2 < 0.49$ | **Smooth** | Holt-Winters Generalized (Mode Tren/Musiman/SES) |
| $\text{ADI} < 1.32 \land \text{CV}^2 \ge 0.49$ | **Erratic** | Holt-Winters Mode Stabil ($\alpha$ kecil + interval ketidakpastian) |
| $\text{ADI} \ge 1.32 \land \text{CV}^2 < 0.49$ | **Intermittent** | **Croston's Method** |
| $\text{ADI} \ge 1.32 \land \text{CV}^2 \ge 0.49$ | **Lumpy** | **Croston's Method** (interval ketidakpastian dilebarkan) |

---

### B. Formulasi Metode Peramalan

#### 1. Holt-Winters Generalized (Additive)
Dapat dioperasikan dalam **Mode Stabil (setara SES)** dan **Mode Penuh (Tren + Musiman)**:

- **Level ($L_t$):**
  $$L(t) = \alpha \cdot (D(t) - S(t-m)) + (1 - \alpha) \cdot (L(t-1) + T(t-1))$$
- **Tren ($T_t$):**
  $$T(t) = \beta \cdot (L(t) - L(t-1)) + (1 - \beta) \cdot T(t-1)$$
- **Musiman ($S_t$):**
  $$S(t) = \gamma \cdot (D(t) - L(t)) + (1 - \gamma) \cdot S(t-m)$$
- **Persamaan Forecast ($F_{t+h}$):**
  $$F(t+h) = L(t) + h \cdot T(t) + S(t + h - m)$$

* **Mode Stabil (SES):** Jika $\beta = 0$ dan $\gamma = 0$, formula menyederhana menjadi $L(t) = \alpha D(t) + (1-\alpha)L(t-1)$ dan $F(t+h) = L(t)$.
* **Syarat Mode Musiman:** Data historis harus mencakup minimal **2 siklus musiman penuh** (contoh: $\ge 14$ hari untuk musiman mingguan $m=7$).

#### 2. Croston's Method (Permintaan Intermiten B2B)
Memisahkan deret waktu menjadi **Ukuran Order ($Y$)** dan **Interval Antar Order ($X$)**:
- **Smoothing Ukuran Order ($Z_i$):**
  $$Z(i) = \alpha \cdot Y(i) + (1 - \alpha) \cdot Z(i-1)$$
- **Smoothing Interval Order ($V_i$):**
  $$V(i) = \alpha \cdot X(i) + (1 - \alpha) \cdot V(i-1)$$
- **Peramalan Demand Rata-Rata per Periode ($F$):**
  $$F = \frac{Z(i)}{V(i)}$$

#### 3. Weighted Moving Average (WMA)
$$\text{WMA} = \frac{\sum_{i=1}^{n} (W_i \cdot D_i)}{\sum_{i=1}^{n} W_i}$$

---

### C. Optimasi Parameter & Pemicu Notifikasi

1. **Optimasi Parameter ($\alpha, \beta, \gamma$) via Grid Search:**
   Mencari kombinasi $\alpha, \beta, \gamma \in [0.01, 0.99]$ yang meminimalkan Mean Squared Error (MSE):
   $$\text{MSE} = \frac{1}{N} \sum_{t=1}^{N} (D(t) - F(t))^2$$
   * Jika $\beta$ hasil optimasi $< 0.01$, produk dilabeli **Mode Stabil**.
   * Jika $\beta \ge 0.01$, dilabeli **Mode Penuh**.

2. **Ambang Batas Pemicu Notifikasi (Alert Threshold):**
   $$\text{Persentase Perubahan} = \frac{F_{\text{periode depan}} - \bar{D}_{3 \text{ periode terakhir}}}{\bar{D}_{3 \text{ periode terakhir}}} \times 100\%$$
   $$\text{IF } |\text{Persentase Perubahan}| \ge 10\% \implies \text{Kirim Notifikasi Dashboard}$$

---

### D. Formulasi Rekomendasi Stok & Harga

1. **Rekomendasi Target Stok Gudang:**
   $$\text{Target Stok} = F_{\text{periode depan}} \times (1 + \text{Buffer Safety Stock})$$
   * Buffer Safety Stock: $10\%$ untuk pola stasioner/smooth, $15\% - 20\%$ untuk pola musiman/erratic.

2. **Rekomendasi Rentang Harga Jual Supplier:**
   $$\text{Faktor Permintaan} = \frac{\text{Persentase Perubahan}}{100}$$
   $$\text{Harga Bawah} = \text{Harga Rata-Rata Pasar} \times \left(1 + (\text{Faktor Permintaan} \times \text{Elastisitas Markup Min})\right)$$
   $$\text{Harga Atas} = \text{Harga Rata-Rata Pasar} \times \left(1 + (\text{Faktor Permintaan} \times \text{Elastisitas Markup Maks})\right)$$

---

## 📊 3. Spesifikasi Data untuk Training

* **Sumber Data:** Database internal BakuLink (`orders` dan `order_items`).
* **Kebutuhan Minimum:**
  * Minimal **8 minggu data historis** untuk pola Stasioner/Tren.
  * Minimal **2 siklus penuh** untuk mendeteksi Musiman.
* **Struktur Dataset CSV:**

| Kolom | Tipe Data | Deskripsi | Contoh |
|:---|:---|:---|:---|
| `tanggal` | Date | Tanggal pencatatan (YYYY-MM-DD) | `2026-07-10` |
| `komoditas` | Text | Jenis bahan baku | `Tepung Terigu` |
| `total_harga` | Numeric | Rata-rata harga periode tersebut | `12500` |
| `is_high_season` | Binary | 1 jika musim liburan/hari raya, 0 jika biasa | `1` |
| `volume_permintaan` | Numeric | **Target Variable**: Kuantitas penjualan (kg) | `550` |

---

## 💻 4. Panduan Langkah demi Langkah Pelatihan & Eksekusi di Google Colab / Python

Berikut adalah skrip Python lengkap untuk mengklasifikasikan data, melatih model forecasting (Croston, Holt-Winters, Random Forest), dan mengunduh berkas `.pkl`.

### Langkah 4.1: Install & Import Library
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

print("Library berhasil di-import!")
```

### Langkah 4.2: Fungsi Klasifikasi Syntetos-Boylan & Croston's Method
```python
def classify_syntetos_boylan(df_demand):
    """
    df_demand: Series dengan indeks tanggal dan nilai kuantitas order harian/mingguan
    """
    N = len(df_demand)
    non_zero = df_demand[df_demand > 0]
    K = len(non_zero)
    
    if K == 0:
        return "No Data", 0, 0
    
    ADI = N / K
    mean_Y = non_zero.mean()
    std_Y = non_zero.std() if len(non_zero) > 1 else 0
    CV2 = (std_Y / mean_Y) ** 2 if mean_Y > 0 else 0
    
    if ADI < 1.32 and CV2 < 0.49:
        category = "Smooth"
    elif ADI < 1.32 and CV2 >= 0.49:
        category = "Erratic"
    elif ADI >= 1.32 and CV2 < 0.49:
        category = "Intermittent"
    else:
        category = "Lumpy"
        
    return category, round(ADI, 2), round(CV2, 4)

def crostons_method(ts_data, alpha=0.2):
    """
    ts_data: list/array kuantitas order per periode
    """
    Y = [] # Order quantities
    X = [] # Intervals between orders
    
    count_zero = 0
    for val in ts_data:
        count_zero += 1
        if val > 0:
            Y.append(val)
            X.append(count_zero)
            count_zero = 0
            
    if not Y:
        return 0, 0, 0
        
    # Inisialisasi Z dan V
    Z = Y[0]
    V = X[0]
    
    for i in range(1, len(Y)):
        Z = alpha * Y[i] + (1 - alpha) * Z
        V = alpha * X[i] + (1 - alpha) * V
        
    forecast_per_period = Z / V if V > 0 else 0
    return forecast_per_period, Z, V

# Uji Coba Fungsi Croston
sample_orders = [0, 0, 120, 0, 0, 0, 95, 0, 0, 0, 0, 0, 0, 140, 0, 0, 0, 0, 110, 0]
cat, adi, cv2 = classify_syntetos_boylan(pd.Series(sample_orders))
f_croston, z_val, v_val = crostons_method(sample_orders, alpha=0.2)

print(f"Klasifikasi: {cat} (ADI: {adi}, CV2: {cv2})")
print(f"Hasil Croston: Forecast/hari={f_croston:.2f} kg (Est Order={z_val:.1f} kg tiap {v_val:.1f} hari)")
```

### Langkah 4.3: Pelatihan Model Random Forest Regressor & Multi-Variabel
```python
# Load dataset transaksi
# Format CSV: tanggal, komoditas, total_harga, is_high_season, volume_permintaan
df = pd.read_csv('volume_penjualan_historis.csv')

# Filter komoditas tertentu, contoh 'Beras'
df_beras = df[df['komoditas'].str.lower() == 'beras'].copy()

# Buat Fitur & Target
X = df_beras[['total_harga', 'is_high_season']]
y = df_beras['volume_permintaan']

# Fit Model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X, y)

# Evaluasi
preds = model_rf.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds))
print(f"Model Random Forest Beras berhasil dilatih! RMSE: {rmse:.2f} kg")

# Simpan Model ke .pkl
with open('model_demand_beras.pkl', 'wb') as f:
    pickle.dump(model_rf, f)
print("File model_demand_beras.pkl siap diunduh!")
```

---

## 💾 5. Implementasi Endpoint FastAPI (`main.py`)

Di FastAPI `bakulink-service-ai/main.py`, gabungkan algoritma klasifikasi Syntetos-Boylan, Croston, dan Machine Learning:

```python
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="BakuLink AI Demand Service")

# Load ML Model
with open("models/model_demand_beras.pkl", "rb") as f:
    model_demand_beras = pickle.load(f)

class DemandForecastRequest(BaseModel):
    commodity: str
    current_price: float
    is_high_season: int
    recent_sales_history: list[float]  # Histori penjualan 14-30 hari terakhir

@app.post("/predict-demand")
def predict_demand(req: DemandForecastRequest):
    history = np.array(req.recent_sales_history)
    
    # 1. Hitung Syntetos-Boylan Classification
    N = len(history)
    non_zero = history[history > 0]
    K = len(non_zero)
    
    if K > 0 and N > 0:
        adi = N / K
        mean_y = np.mean(non_zero)
        std_y = np.std(non_zero) if K > 1 else 0
        cv2 = (std_y / mean_y) ** 2 if mean_y > 0 else 0
    else:
        adi, cv2 = 1.0, 0.0

    # 2. Pilih Strategi Peramalan
    if adi >= 1.32:
        # Pola Intermittent / Lumpy -> Croston's Method
        Y = []
        X_intervals = []
        c_zero = 0
        for val in history:
            c_zero += 1
            if val > 0:
                Y.append(val)
                X_intervals.append(c_zero)
                c_zero = 0
        
        Z, V = Y[0], X_intervals[0]
        for i in range(1, len(Y)):
            Z = 0.2 * Y[i] + 0.8 * Z
            V = 0.2 * X_intervals[i] + 0.8 * V
        
        forecast_daily = Z / V if V > 0 else 0
        method_used = "Croston's Method (Intermittent)"
        predicted_volume = int(forecast_daily * 7) # Proyeksi 7 hari
    else:
        # Pola Smooth -> Machine Learning Model
        input_data = np.array([[req.current_price, req.is_high_season]])
        predicted_volume = int(model_demand_beras.predict(input_data)[0])
        method_used = "Random Forest Regressor (Smooth)"

    # 3. Hitung Alert Threshold & Rekomendasi Stok/Harga
    recent_3_avg = np.mean(history[-3:]) if len(history) >= 3 else predicted_volume
    pct_change = ((predicted_volume - recent_3_avg) / recent_3_avg) * 100 if recent_3_avg > 0 else 0
    
    trigger_alert = abs(pct_change) >= 10.0
    buffer_stock = 0.15 if cv2 >= 0.49 else 0.10
    target_stock = int(predicted_volume * (1 + buffer_stock))
    
    demand_factor = pct_change / 100.0
    rec_price_min = int(req.current_price * (1 + (demand_factor * 0.05)))
    rec_price_max = int(req.current_price * (1 + (demand_factor * 0.15)))

    return {
        "status": "success",
        "commodity": req.commodity,
        "classification": {"adi": round(adi, 2), "cv2": round(cv2, 4)},
        "method_used": method_used,
        "predicted_demand_7days_kg": predicted_volume,
        "alert_triggered": trigger_alert,
        "pct_change": round(pct_change, 2),
        "inventory_advice": {
            "target_stock_kg": target_stock,
            "safety_buffer_pct": int(buffer_stock * 100)
        },
        "pricing_intelligence": {
            "recommended_price_min": rec_price_min,
            "recommended_price_max": rec_price_max
        }
    }
```

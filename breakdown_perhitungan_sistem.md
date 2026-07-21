# Pemetaan & Breakdown Perhitungan Sistem BakuLink 🧮

Dokumen ini berisi breakdown mendalam dari file **`Perhitungan (1).docx`** serta seluruh sistem perhitungan matematika yang ada pada platform BakuLink. Seluruh formula dipetakan secara terstruktur ke dalam modul AI yang tepat (**AI 1**, **AI 2**, **AI 3**) atau ke dalam **Engine Analytics & Logistik Backend Non-AI**.

---

## 📌 Ringkasan Pemetaan Modul Perhitungan

| No | Modul Perhitungan | Masuk ke Sistem | Peran Utama dalam Platform |
|---|---|---|---|
| 1 | **Price Forecasting & Market Price Adjustment** | **AI 1: Price Forecasting** | Meramal tren harga pangan 7–30 hari ke depan, menjahit data transaksi internal, web scraping PIHPS/Bapanas, dan rasio pasokan supplier. |
| 2 | **Demand Classification, Forecasting & Stock/Price Recommendation** | **AI 2: Demand Forecasting** | Mengklasifikasikan pola permintaan (Syntetos-Boylan), peramalan (Holt-Winters, Croston's, WMA), alert threshold 10%, rekomendasi target stok gudang, & rekomendasi harga jual supplier. |
| 3 | **Supplier & 3PL Selection Engine (TLC, SAW, Bayesian Rating)** | **AI 3: Procurement AI Advisor** | Algoritma pengadaan cerdas untuk menghitung berat volume kargo, Total Landed Cost (TLC), ranking supplier gabungan (SAW 33.33%), Bayesian rating, serta RAG Chatbot. |
| 4 | **Buyer Analytics & Logistics Performance Metrics** | **Backend Analytics Engine (Non-AI)** | Menghitung total belanja pengadaan UMKM, penghematan multisourcing (jarak GPS), Fulfillment Rate (Quantity Match, On-Time, Damage-Free), Lead Time, Donut Chart porsi belanja, dan matriks armada 3PL. |
| 5 | **Supplier Trust Score & Certification Scoring** | **Backend Scoring Model (`SupplierScore`)** | Menghitung skor kepercayaan supplier berdasarkan historis akurasi pengiriman, konsistensi harga, response rate, dan rating. |

---

## 📈 1. Breakdown AI System 1: Price Forecasting (Peramalan Harga Pangan)

Fitur ini bertugas memprediksi harga komoditas pangan pokok untuk 7 hingga 30 hari ke depan agar UMKM dapat membeli di saat termurah dan Supplier dapat menyusun strategi harga.

### A. Grafik Historis (Data Riil Transaksi Internal)
- **Sumber Data:** Tabel `orders` dan `order_items` internal platform BakuLink.
- **Formulasi:**
  $$\text{Rata-Rata Harga Mingguan} = \frac{\sum (\text{Harga Satuan Transaksi})}{\text{Total Transaksi Mingguan}}$$
- Digunakan untuk menggambarkan garis tren masa lalu hingga hari ini.

### B. Grafik Proyeksi (Forecasting 7–30 Hari Ke Depan)
Peramalan harga menggabungkan 3 sumber informasi:
1. **Pola Tren Internal:** Melanjutkan tren persentase kenaikan/penurunan mingguan historis.
2. **Data Eksternal (Web Scraping / API Government):** Menarik data PIHPS (`hargapangan.id`) atau Badan Pangan Nasional untuk menangkap isu kelangkaan nasional.
3. **Faktor Pasokan Supplier (Supply-Demand Factor Adjustment):**
   - AI membaca total stok komoditas dari seluruh supplier lokal di platform.
   - Jika stok menipis saat permintaan tinggi, AI menambahkan kalkulasi lompatan proyeksi harga:
     $$\text{Proyeksi Harga} = \text{Harga Base Forecast} \times (1 + \Delta_{\text{scarcity}})$$
     *(Contoh: Penyesuaian kenaikan $+4.2\%$ jika stok lokal di bawah ambang krisis).*

---

## 📦 2. Breakdown AI System 2: Demand Forecasting (Peramalan Permintaan Pasar - Supplier Side)

Modul ini membantu Verified Supplier memprediksi volume permintaan pasar, mengelola stok gudang, dan menetapkan harga jual optimal.

### A. Klasifikasi Pola Demand (Syntetos - Boylan Classification)
Sebelum menentukan metode forecast, data historis setiap produk dianalisis menggunakan dua indikator:
1. **Average Demand Interval (ADI):**
   $$\text{ADI} = \frac{N}{K}$$
   *di mana $N$ = total periode historis, $K$ = jumlah periode dengan order ($D > 0$).*
2. **Squared Coefficient of Variation ($CV^2$):**
   $$\text{CV}^2 = \left( \frac{\sigma_Y}{\mu_Y} \right)^2$$
   *di mana $\sigma_Y$ dan $\mu_Y$ adalah standar deviasi dan rata-rata kuantitas order (hanya untuk periode ber-demand).*

**Matriks Klasifikasi Syntetos-Boylan:**

| Kondisi | Klasifikasi | Metode Forecasting yang Digunakan |
|---|---|---|
| $\text{ADI} < 1.32$ dan $\text{CV}^2 < 0.49$ | **Smooth** (Kontinu & Stabil) | Lanjut ke Tahap 2 (Holt-Winters / SES) |
| $\text{ADI} < 1.32$ dan $\text{CV}^2 \ge 0.49$ | **Erratic** (Fluktuatif Kontinu) | Holt-Winters mode stabil ($\alpha$ diperkecil + interval ketidakpastian) |
| $\text{ADI} \ge 1.32$ dan $\text{CV}^2 < 0.49$ | **Intermittent** (Order Jarang) | **Croston's Method** |
| $\text{ADI} \ge 1.32$ dan $\text{CV}^2 \ge 0.49$ | **Lumpy** (Jarang & Irreguler) | **Croston's Method** (dengan interval ketidakpastian dilebarkan) |

---

### B. Metode Peramalan (Forecasting Methods)

#### 1. Holt-Winters Generalized (Additive)
Formulasi umum yang mencakup **Mode Stabil (setara SES)** dan **Mode Penuh (Tren + Musiman)**:

- **Komponen Level ($L_t$):**
  $$L(t) = \alpha \cdot (D(t) - S(t-m)) + (1 - \alpha) \cdot (L(t-1) + T(t-1))$$
- **Komponen Tren ($T_t$):**
  $$T(t) = \beta \cdot (L(t) - L(t-1)) + (1 - \beta) \cdot T(t-1)$$
- **Komponen Musiman ($S_t$):**
  $$S(t) = \gamma \cdot (D(t) - L(t)) + (1 - \gamma) \cdot S(t-m)$$
- **Persamaan Peramalan ($F_{t+h}$):**
  $$F(t+h) = L(t) + h \cdot T(t) + S(t + h - m)$$

*Keterangan:*
- **Mode Stabil (SES):** Dipakai jika $\beta = 0$ dan $\gamma = 0$ ($S = 0$), sehingga $L(t) = \alpha D(t) + (1-\alpha)L(t-1)$ dan $F(t+h) = L(t)$.
- **Mode Penuh:** Aktif saat syarat data musiman terpenuhi (minimal 2 siklus musiman penuh, misal $\ge 14$ hari untuk musiman mingguan $m=7$).

#### 2. Croston's Method (Untuk Permintaan Intermiten / B2B Sporadis)
Memisahkan data menjadi deret **Ukuran Order ($Y$)** dan **Interval Antar Order ($X$)**:
- **Smoothing Ukuran Order ($Z$):**
  $$Z(i) = \alpha \cdot Y(i) + (1 - \alpha) \cdot Z(i-1)$$
- **Smoothing Interval Order ($V$):**
  $$V(i) = \alpha \cdot X(i) + (1 - \alpha) \cdot V(i-1)$$
- **Peramalan Permintaan Rata-Rata per Periode ($F$):**
  $$F = \frac{Z(i)}{V(i)}$$

#### 3. Weighted Moving Average (WMA)
Dipakai untuk pola stasioner sederhana:
$$\text{WMA} = \frac{W_n \cdot D_n + W_{n-1} \cdot D_{n-1} + \dots + W_1 \cdot D_1}{W_n + W_{n-1} + \dots + W_1}$$

---

### C. Optimasi Parameter & Pemicu Notifikasi

1. **Optimasi Parameter ($\alpha, \beta, \gamma$):**
   Dijalankan via Grid Search / Minimisasi Error (MSE/RMSE) terhadap data historis:
   $$\text{MSE} = \frac{1}{N} \sum_{t=1}^{N} (D(t) - F(t))^2$$
   - Jika $\beta \approx 0$ (misal $< 0.01$), model dilabeli Mode Stabil (SES).
   - Jika $\beta > 0.01$, model dilabeli Mode Penuh.
   - $\gamma$ hanya diikutkan jika data historis $\ge 2$ siklus musiman penuh.

2. **Ambang Batas Pemicu Notifikasi (Alert Threshold):**
   $$\text{Persentase Perubahan} = \frac{F_{\text{periode depan}} - \bar{D}_{3 \text{ periode terakhir}}}{\bar{D}_{3 \text{ periode terakhir}}} \times 100\%$$
   $$\text{IF } |\text{Persentase Perubahan}| \ge 10\% \implies \text{Kirim Notifikasi Rekomendasi}$$

---

### D. Komponen Rekomendasi Dashboard Supplier

1. **Target Stok Gudang (Inventory Advice):**
   $$\text{Target Stok} = F_{\text{periode depan}} \times (1 + \text{Buffer Safety Stock})$$
   *Buffer Safety Stock disesuaikan nilai $CV$ (10% untuk stabil, 15–20% untuk musiman).*

2. **Rekomendasi Rentang Harga Jual (Pricing Intelligence):**
   $$\text{Faktor Permintaan} = \frac{\text{Persentase Perubahan}}{100}$$
   $$\text{Harga Bawah} = \text{Harga Rata-Rata Pasar} \times \left(1 + (\text{Faktor Permintaan} \times \text{Elastisitas Markup Min})\right)$$
   $$\text{Harga Atas} = \text{Harga Rata-Rata Pasar} \times \left(1 + (\text{Faktor Permintaan} \times \text{Elastisitas Markup Maks})\right)$$

---

## 🤖 3. Breakdown AI System 3: Procurement AI Advisor & Matching Algorithm (Buyer Side)

Modul ini mencakup chatbot interaktif RAG (ChromaDB + Gemini) serta **Algoritma Pemilihan Supplier & 3PL (Total Landed Cost Engine)**.

### A. RAG Knowledge Base Chatbot (ChromaDB + Gemini 1.5 Flash)
- **Fungsi:** Menjawab pertanyaan regulasi HET (Harga Eceran Tertinggi), standar mutu komoditas (SNI), dan panduan rantai pasok.
- **Pencarian Semantik:** Mengubah dokumen regulasi menjadi vektor embedding (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) di ChromaDB.

---

### B. Algoritma Kalkulasi Pemilihan Supplier & 3PL (Backend Engine)

#### 1. Penentuan Berat Kargo (Dimensional Weight Logic)
$$\text{W}_{\text{volume}} = \left(\frac{P \times L \times T}{4000}\right) \times Q$$
$$\text{W}_{\text{final}} = \max(W_{\text{actual}} \times Q, W_{\text{volume}})$$

#### 2. Total Landed Cost (TLC) Mentah
$$\text{Ongkos Kirim } (C_{\text{ongkir}}) = \begin{cases} \text{Berat Min 3PL} \times \text{Tarif/kg}, & \text{jika } W_{\text{final}} < \text{Berat Min} \\ W_{\text{final}} \times \text{Tarif/kg}, & \text{jika } W_{\text{final}} \ge \text{Berat Min} \end{cases}$$
$$\text{TLC} = (\text{Harga Produk} \times Q) + C_{\text{ongkir}} + \text{Biaya Handling/Admin}$$

#### 3. Opsi A — Mode Gabungan (Simple Additive Weighting / SAW)
Menggabungkan TLC, Lead Time, dan Rating dengan bobot rata **33.33% (0.333)**:
- **Normalisasi TLC (Cost):** $R_{\text{TLC}} = \frac{\text{TLC}_{\text{terendah}}}{\text{TLC}_{\text{supplier ini}}}$
- **Normalisasi Lead Time (Cost):** $R_{\text{Time}} = \frac{\text{Lead Time}_{\text{tercepat}}}{\text{Lead Time}_{\text{supplier ini}}}$
- **Normalisasi Rating (Benefit):** $R_{\text{Rating}} = \frac{\text{Rating}_{\text{supplier ini}}}{5.0}$
- **Skor Akhir SAW:**
  $$\text{Skor Akhir} = (R_{\text{TLC}} \times 0.333) + (R_{\text{Time}} \times 0.333) + (R_{\text{Rating}} \times 0.333)$$

#### 4. Opsi B — Mode Terpisah (Aspek Tunggal Mutlak)
- **Fokus Biaya:** $\min(\text{TLC})$ dengan tie-breaker Lead Time tercepat.
- **Fokus Waktu (Lead Time Efektif):**
  $$\text{Lead Time Efektif} = \text{Lead Time Standar} + \text{Rata-Rata Hari Terlambat}$$
- **Fokus Kualitas (Bayesian Average Rating):**
  $$\text{Skor Rating} = \frac{v \cdot R + m \cdot C}{v + m}$$
  *di mana $v$ = jumlah ulasan supplier, $m$ = minimum ulasan kualifikasi (misal 5), $R$ = rating rata-rata supplier, $C$ = rating rata-rata seluruh supplier platform.*

---

## 📊 4. Sistem Perhitungan Analytics & Logistik Non-AI (Buyer Analytics)

Perhitungan ini digunakan pada Dashboard UMKM/Pembeli untuk mengukur efisiensi pengadaan dan performa logistik 3PL.

### A. Analytics Belanja Pengadaan
1. **Total Pembelian:**
   $$\text{Total Pembelian} = \sum (\text{Kuantitas Barang} \times \text{Harga Satuan})$$
2. **Penghematan Biaya Multisourcing (GPS Matchmaking):**
   Membandingkan total biaya supplier yang direkomendasikan AI (jarak dekat via koordinat GPS Latitude & Longitude) vs supplier alternatif berjarak jauh.

### B. Indikator Fulfillment Rate & Performa Logistik
Satu transaksi dinyatakan **Sempurna (Skor = 1)** jika dan hanya jika memenuhi 3 indikator:
1. **Quantity Match:** $\text{QTY ORDERED} = \text{QTY PICKED} = \text{QTY RECEIVED}$.
2. **On-Time Delivery:** $\text{ACTUAL\_DELIVERY\_TIME} \le \text{ESTIMATED\_DELIVERY\_TIME}$.
3. **Damage-Free Delivery:** $\text{CONFIRMATION\_STATUS} = \text{"Diterima dengan Baik"}$ & tidak ada `DISPUTE_LOG`.

$$\text{Fulfillment Rate} = \frac{\text{Jumlah Transaksi Sempurna}}{\text{Total Transaksi}} \times 100\%$$

### C. Rata-Rata Lead Time Pengiriman
$$\text{Lead Time} = \frac{1}{N} \sum_{i=1}^{N} (T2_i - T1_i)$$
*di mana $T1$ = timestamp Picked Up 3PL, $T2$ = timestamp Proof of Delivery (POD).*

### D. Donut Chart Porsi Belanja Komoditas
$$\text{Porsi Kategori A} = \frac{\text{Total Belanja Kategori A}}{\text{Total Seluruh Belanja}} \times 100\%$$

### E. Aturan Pemilihan Jenis Kendaraan 3PL & Biaya Handling
- **Motor 3PL (Tarif Flat/km):** Berat $\le 50 \text{ kg}$.
- **Mobil Pick-up 3PL (Tarif Flat/km):** Berat $51 \text{ kg} - 1.000 \text{ kg}$.
- **Truk Engkel 3PL (Tarif Flat/km):** Berat $1.01 \text{ ton} - 2.5 \text{ ton}$.
- **Biaya Bongkar Muat (Tiered):**
  - Mandiri = $\text{Rp } 0$.
  - 3PL Volume $\le 100 \text{ kg}$ = Tarif Tier 1.
  - 3PL Volume $\le 500 \text{ kg}$ = Tarif Tier 2.
  - 3PL Volume $> 500 \text{ kg}$ = Tarif Tier 3.
- **Biaya Asuransi & Admin:**
  $$\text{Biaya Asuransi \& Admin} = (\text{Persentase Premi} \times \text{Harga Bahan Baku}) + \text{Admin Flat Platform}$$

---

## 🏆 5. Sistem Perhitungan Skor Kepercayaan Supplier (`SupplierScore`)

Di backend Laravel (`App\Models\SupplierScore`), tingkat kepercayaan supplier dihitung berdasarkan indikator kinerja:

1. **Delivery Accuracy Pct:**
   $$\text{Delivery Accuracy} = \frac{\text{Completed Orders} - \text{Late Deliveries}}{\text{Completed Orders}} \times 100\%$$
2. **Price Consistency Pct:** Mengukur seberapa stabil harga yang dipasang supplier dibanding rata-rata fluktuasi pasar.
3. **Response Rate Pct:** Tingkat kecepatan merespon pesanan dan obrolan pengguna.
4. **AI Trust Score:** Skor agregat 0–100 yang menjadi dasar lencana *Verified Supplier*.

---
*Dokumen ini merupakan acuan resmi pembagian komputasi matematika di dalam ekosistem platform BakuLink.*

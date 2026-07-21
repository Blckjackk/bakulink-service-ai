# Catatan AI 3: Procurement AI Advisor & Matching Engine (Asisten Pengadaan Pintar) 🤖

Dokumen ini menjelaskan rancangan, spesifikasi data, matematika algoritma komputasi pemilihan supplier/3PL, dan langkah demi langkah pembuatan asisten pengadaan pintar berbasis RAG (ChromaDB + Gemini) di BakuLink.

---

## 📝 1. Deskripsi AI & Cakupan Fitur

Procurement AI Advisor memiliki **dua fungsi utama**:
1. **Chatbot Konsultan Pengadaan (RAG System):** Berinteraksi secara kontekstual untuk menjawab kebijakan perdagangan, regulasi Harga Eceran Tertinggi (HET), standar mutu komoditas (SNI), serta panduan manajemen persediaan.
2. **Algoritma Pemilihan Supplier & 3PL (Compute Engine):** Menghitung berat kargo volume, Total Landed Cost (TLC), ranking Simple Additive Weighting (SAW 33.33%), dan Bayesian Rating untuk menentukan rekomendasi pemasok dan logistik terbaik bagi UMKM.

---

## 🧮 2. Landasan Rumus Matematika Algoritma Pemilihan Supplier & 3PL

### A. Penentuan Berat Kargo (Dimensional Weight Logic)
AI menghitung berat aktual vs berat volume, lalu mengambil nilai terbesar sebagai dasar pengalian tarif 3PL:
$$W_{\text{volume}} = \left( \frac{P \times L \times T}{4000} \right) \times Q$$
$$W_{\text{final}} = \max(W_{\text{actual}} \times Q, W_{\text{volume}})$$
*(4000 = standar pembagi kargo darat/laut, P, L, T dalam cm).*

---

### B. Total Landed Cost (TLC) Mentah
Menghitung total biaya riil untuk setiap pasangan Supplier + 3PL:
$$C_{\text{ongkir}} = \begin{cases} \text{Berat Minimum 3PL} \times \text{Tarif/kg}, & \text{jika } W_{\text{final}} < \text{Berat Minimum 3PL} \\ W_{\text{final}} \times \text{Tarif/kg}, & \text{jika } W_{\text{final}} \ge \text{Berat Minimum 3PL} \end{cases}$$

$$\text{TLC} = (\text{Harga Produk} \times Q) + C_{\text{ongkir}} + \text{Biaya Handling/Admin}$$

---

### C. Opsi A — Mode Gabungan (Simple Additive Weighting / SAW)
Dipicu saat pembeli memilih **"Opsi Ideal (Rata-Rata)"** dengan bobot seimbang $33.33\%$ ($0.333$) per aspek:

1. **Normalisasi TLC (Cost - Semakin Kecil Semakin Baik):**
   $$R_{\text{TLC}} = \frac{\text{TLC}_{\text{Terendah}}}{\text{TLC}_{\text{Supplier Ini}}}$$
2. **Normalisasi Lead Time (Cost - Semakin Kecil Semakin Baik):**
   $$R_{\text{Time}} = \frac{\text{Lead Time}_{\text{Tercepat}}}{\text{Lead Time}_{\text{Supplier Ini}}}$$
3. **Normalisasi Rating (Benefit - Semakin Besar Semakin Baik):**
   $$R_{\text{Rating}} = \frac{\text{Rating}_{\text{Supplier Ini}}}{5.0}$$
4. **Skor Akhir SAW:**
   $$\text{Skor Akhir} = (R_{\text{TLC}} \times 0.333) + (R_{\text{Time}} \times 0.333) + (R_{\text{Rating}} \times 0.333)$$

---

### D. Opsi B — Mode Terpisah (Aspek Tunggal Mutlak)
1. **Fokus Biaya (TLC Termurah):**
   * Filter/Sort: $\min(\text{TLC})$. Tie-breaker: Lead Time tercepat.
2. **Fokus Waktu (Lead Time Tercepat):**
   * Memperhitungkan penalty riwayat keterlambatan supplier:
     $$\text{Lead Time Efektif} = \text{Lead Time Standar} + \text{Rata-Rata Hari Terlambat}$$
   * Sort: $\min(\text{Lead Time Efektif})$.
3. **Fokus Kualitas (Bayesian Average Rating):**
   * Mencegah bias supplier baru bintang 5 dari 1 ulasan dikalkulasikan lebih tinggi dari supplier bintang 4.9 dari 1.000 ulasan:
     $$\text{Skor Rating} = \frac{v \cdot R + m \cdot C}{v + m}$$
     * $v$: Jumlah ulasan supplier tersebut.
     * $m$: Batas minimum ulasan kualifikasi (misal: 5).
     * $R$: Rata-rata rating supplier tersebut (1-5).
     * $C$: Rata-rata rating seluruh supplier di platform.

---

## 📊 3. Spesifikasi Data untuk Indexing (ChromaDB)

* **Sumber Pengetahuan:**
  1. PDF/Teks Regulasi Kementerian Perdagangan & Bapanas tentang HET.
  2. Artikel panduan rantai pasok & manajemen stok UMKM (FIFO).
  3. Standar mutu komoditas BSN (Beras Medium vs. Premium, SNI Minyak Goreng).
* **Format:** Kumpulan dokumen `.txt` / `.pdf` yang di-chunk dan di-embed ke vektor.

---

## 💻 4. Langkah demi Langkah Pembuatan Vector DB & RAG di Google Colab

### Langkah 4.1: Install Library
```python
!pip install chromadb sentence-transformers
```

### Langkah 4.2: Inisialisasi ChromaDB & Indeks Dokumen
```python
import chromadb
from chromadb.utils import embedding_functions

# Database Persistent
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Model Embedding Multilingual
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

collection = chroma_client.create_collection(
    name="bakulink_knowledge", 
    embedding_function=emb_fn
)

documents = [
    {
        "id": "het-beras-2026",
        "text": "Berdasarkan Peraturan Badan Pangan Nasional Nomor 7 Tahun 2026, Harga Eceran Tertinggi (HET) Beras Medium untuk wilayah Jawa, Lampung, dan Sumatra Selatan ditetapkan sebesar Rp 13.900 per kilogram. Untuk Beras Premium ditetapkan sebesar Rp 15.400 per kilogram."
    },
    {
        "id": "manajemen-stok-umkm",
        "text": "Tips manajemen stok UMKM: Gunakan metode FIFO (First In First Out). Batasi stok cabai segar maksimal untuk kebutuhan 3 hari produksi."
    }
]

collection.add(
    documents=[doc["text"] for doc in documents],
    ids=[doc["id"] for doc in documents]
)

print("ChromaDB Knowledge Base berhasil dibuat!")
```

---

## 💾 5. Integration di FastAPI (`main.py`)

```python
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Inisialisasi ChromaDB
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="bakulink_knowledge", embedding_function=emb_fn)

# Model Data SAW Request
class SupplierMatchRequest(BaseModel):
    length_cm: float
    width_cm: float
    height_cm: float
    actual_weight_kg: float
    quantity: int
    suppliers: list[dict] # [{id, price, lead_time, rating, review_count, min_weight_3pl, tariff_per_kg}]

@app.post("/rank-suppliers-saw")
def rank_suppliers_saw(req: SupplierMatchRequest):
    # 1. Dimensional Weight
    vol_weight = ((req.length_cm * req.width_cm * req.height_cm) / 4000.0) * req.quantity
    final_weight = max(req.actual_weight_kg * req.quantity, vol_weight)
    
    results = []
    for s in req.suppliers:
        # TLC
        chargeable_w = max(final_weight, s.get('min_weight_3pl', 1.0))
        c_ongkir = chargeable_w * s['tariff_per_kg']
        tlc = (s['price'] * req.quantity) + c_ongkir
        
        # Bayesian Rating (m=5, C=4.5)
        v = s.get('review_count', 0)
        R = s.get('rating', 0)
        bayesian_r = (v * R + 5 * 4.5) / (v + 5)
        
        results.append({
            "supplier_id": s['id'],
            "tlc": tlc,
            "lead_time": s['lead_time'],
            "bayesian_rating": bayesian_r
        })
    
    # 2. Normalisasi SAW
    min_tlc = min(r['tlc'] for r in results)
    min_time = min(r['lead_time'] for r in results)
    
    for r in results:
        r_tlc = min_tlc / r['tlc']
        r_time = min_time / r['lead_time']
        r_rating = r['bayesian_rating'] / 5.0
        
        r['final_saw_score'] = round((r_tlc * 0.333) + (r_time * 0.333) + (r_rating * 0.333), 4)
        
    # Sort
    results.sort(key=lambda x: x['final_saw_score'], reverse=True)
    return {"status": "success", "final_weight_kg": final_weight, "rankings": results}
```

import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Any

app = FastAPI(
    title="BakuLink AI Microservice",
    description="Service AI untuk Price Forecasting, Demand Forecasting, & Procurement AI Advisor BakuLink",
    version="2.0.0"
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
# ChromaDB & Google Gemini LLM RAG Setup
# ----------------------------------------------------------------------
rag_collection = None
gemini_model = None

def init_gemini():
    """
    Inisialisasi Google Gemini API jika GEMINI_API_KEY atau GOOGLE_API_KEY tersedia.
    """
    global gemini_model
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            print("[Gemini LLM] Google Gemini API configured successfully!")
        except Exception as e:
            print(f"[Gemini LLM] Failed to configure Gemini API: {e}")
            gemini_model = None
    else:
        print("[Gemini LLM] GEMINI_API_KEY not found in environment. Using RAG rule fallback.")

def init_rag():
    """
    Inisialisasi ChromaDB collection untuk RAG.
    """
    global rag_collection
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        rag_collection = chroma_client.get_collection(
            name="bakulink_knowledge",
            embedding_function=emb_fn
        )
        print("[RAG] ChromaDB collection loaded successfully.")
    except Exception as e:
        print(f"[RAG] ChromaDB load note: {e}")
        rag_collection = None

# Jalankan init RAG & Gemini saat startup
try:
    init_rag()
    init_gemini()
except Exception:
    pass


# ----------------------------------------------------------------------
# Models Schema Definition
# ----------------------------------------------------------------------

class PricePredictRequest(BaseModel):
    commodity: str = "Tepung Terigu"
    days: int = 7
    total_local_stock_kg: float = 500.0

class DemandPredictRequest(BaseModel):
    commodity: str = "Tepung Terigu"
    category_pattern: str = "rendah"
    unit_price: float = 11000.0
    is_high_season: int = 1
    day_of_week: int = 2
    month: int = 7
    recent_sales_history: list[float] = Field(
        default_factory=lambda: [500.0, 480.0, 520.0, 510.0, 490.0, 505.0, 515.0]
    )

class ChatMessage(BaseModel):
    role: str = "user"   # "user" | "assistant"
    content: str = ""

class ChatAdvisorRequest(BaseModel):
    user_message: str = "Supplier telur terbaik di Bandung"
    chat_history: List[ChatMessage] = Field(default_factory=list)
    # Context supplier dari database Laravel (opsional)
    supplier_context: Optional[List[dict]] = None

class SupplierItem(BaseModel):
    id: str
    name: str
    price: float
    lead_time_hours: float
    rating: float
    review_count: int = 10
    tariff_per_kg: float = 1500.0
    min_weight_3pl: float = 1.0
    # Tambahan untuk custom rank
    late_days_avg: float = 0.0  # Rata-rata hari terlambat (untuk Lead Time Efektif)

class RankSuppliersSAWRequest(BaseModel):
    length_cm: float = 30.0
    width_cm: float = 20.0
    height_cm: float = 15.0
    actual_weight_kg: float = 5.0
    quantity: int = 10
    suppliers: List[SupplierItem]

class RankSuppliersCustomRequest(BaseModel):
    """
    Mode ranking aspek tunggal:
    - cheapest: sort by min(TLC)
    - fastest: sort by min(Lead Time Efektif)
    - quality: sort by max(Bayesian Rating)
    """
    mode: str = "cheapest"  # "cheapest" | "fastest" | "quality"
    length_cm: float = 30.0
    width_cm: float = 20.0
    height_cm: float = 15.0
    actual_weight_kg: float = 5.0
    quantity: int = 10
    suppliers: List[SupplierItem]


# ----------------------------------------------------------------------
# Helper: Bayesian Average Rating
# m=5, C=4.5 (platform average)
# ----------------------------------------------------------------------
def bayesian_rating(v: float, R: float, m: float = 5.0, C: float = 4.5) -> float:
    """
    Skor = (v*R + m*C) / (v + m)
    v: jumlah ulasan, R: rating supplier, m: min ulasan kualifikasi, C: avg platform
    """
    return (v * R + m * C) / (v + m)


# ----------------------------------------------------------------------
# Helper: Dimensional Weight & TLC
# ----------------------------------------------------------------------
def compute_weights_and_tlc(
    length_cm: float, width_cm: float, height_cm: float,
    actual_weight_kg: float, quantity: int,
    suppliers: List[SupplierItem]
) -> tuple[float, list[dict]]:
    """
    Returns (final_weight_kg, list_of_supplier_tlc_dicts)
    """
    vol_weight = ((length_cm * width_cm * height_cm) / 4000.0) * quantity
    final_weight = max(actual_weight_kg * quantity, vol_weight)

    m_const = 5.0
    c_const = 4.5
    results = []

    for s in suppliers:
        chargeable_w = max(final_weight, s.min_weight_3pl)
        c_ongkir = chargeable_w * s.tariff_per_kg
        tlc = (s.price * quantity) + c_ongkir

        v = float(s.review_count)
        R = float(s.rating)
        bayes_r = bayesian_rating(v, R, m_const, c_const)

        # Lead Time Efektif = standar + rata-rata hari terlambat
        effective_lead_time = s.lead_time_hours + (s.late_days_avg * 24)

        results.append({
            "id": s.id,
            "name": s.name,
            "tlc": tlc,
            "lead_time": s.lead_time_hours,
            "effective_lead_time": effective_lead_time,
            "bayesian_rating": round(bayes_r, 4),
            "original_rating": s.rating,
            "review_count": s.review_count,
        })

    return final_weight, results


# ----------------------------------------------------------------------
# Intent Detection Helper for Chat Advisor
# ----------------------------------------------------------------------
COMMODITY_MAP = {
    "beras": ["beras", "nasi", "padi", "rice"],
    "telur": ["telur", "telor", "egg"],
    "minyak": ["minyak", "minyak goreng", "minyak sawit", "oil"],
    "gula": ["gula", "gula pasir", "gula merah", "sugar"],
    "tepung": ["tepung", "terigu", "maizena", "flour"],
    "bumbu": ["bumbu", "rempah", "bawang", "cabai", "jahe"],
    "daging": ["daging", "sapi", "ayam", "kambing", "meat"],
    "susu": ["susu", "dairy", "milk"],
    "sayuran": ["sayuran", "sayur", "kentang", "wortel"],
}

def detect_intent(msg: str) -> dict:
    """
    Mendeteksi intent dari pesan user.
    Returns: {type: 'COMPARISON_TABLE' | 'SINGLE_SUPPLIER_CARD' | 'TEXT_RAG', commodity: str|None}
    """
    m = msg.lower()

    # Deteksi komoditas
    detected_commodity = None
    for cat, keywords in COMMODITY_MAP.items():
        if any(kw in m for kw in keywords):
            detected_commodity = cat
            break

    # Intent: COMPARISON_TABLE
    if any(kw in m for kw in ["bandingkan", "perbandingan", "komparasi", "compare", "3 supplier", "beberapa supplier"]):
        return {"type": "COMPARISON_TABLE", "commodity": detected_commodity}

    # Intent: SINGLE_SUPPLIER_CARD (cari supplier terbaik/termurah/terdekat)
    if detected_commodity and any(kw in m for kw in [
        "terbaik", "terpercaya", "nomor 1", "paling bagus", "rekomendasi",
        "termurah", "terdekat", "supplier", "cari", "carikan"
    ]):
        return {"type": "SINGLE_SUPPLIER_CARD", "commodity": detected_commodity}

    if detected_commodity:
        return {"type": "SINGLE_SUPPLIER_CARD", "commodity": detected_commodity}

    # Default: TEXT_RAG (kebijakan, regulasi, pertanyaan umum)
    return {"type": "TEXT_RAG", "commodity": None}


SUPPLIER_DATA_MOCK = {
    "telur": {
        "name": "Peternakan Telur Segar Bandung",
        "verified": True,
        "role": "Produsen Lokal · Bandung",
        "ai_score": 94,
        "tags": ["Kualitas Premium", "Pengiriman Cepat", "MOQ Rendah", "STOK SIAP"],
        "avg_price": "Rp 24.000 /kg",
        "rating": 4.8,
        "reviews_count": 445,
        "lead_time": "12 jam pengiriman",
        "top_products": ["Telur Ayam Ras", "Telur Ayam Kampung", "Telur Bebek"],
        "maps_url": "https://www.google.com/maps/dir/?api=1&destination=-6.9174,107.6191",
        "slug": "peternakan-telur-segar-bandung",
    },
    "beras": {
        "name": "Sentosa Food Supplier",
        "verified": True,
        "role": "Distributor Utama · Bandung",
        "ai_score": 96,
        "tags": ["Stok Melimpah", "Harga Grosir", "Beras Premium", "STOK SIAP"],
        "avg_price": "Rp 13.500 /kg",
        "rating": 4.9,
        "reviews_count": 812,
        "lead_time": "24 jam pengiriman",
        "top_products": ["Beras Premium Grade A", "Beras Medium Jabar", "Beras Pandan Wangi"],
        "maps_url": "https://www.google.com/maps/dir/?api=1&destination=-6.9023,107.6052",
        "slug": "sentosa-food-supplier",
    },
    "minyak": {
        "name": "UD Karya Sawit Mandiri",
        "verified": True,
        "role": "Agen Resmi · Bekasi",
        "ai_score": 88,
        "tags": ["Harga Kompetitif", "Volume Besar", "Agen Resmi"],
        "avg_price": "Rp 15.200 /liter",
        "rating": 4.6,
        "reviews_count": 290,
        "lead_time": "36 jam pengiriman",
        "top_products": ["Minyak Sawit Kemasan 1L", "Minyak Goreng Curah 5L", "Minyak Premium"],
        "maps_url": "https://www.google.com/maps/dir/?api=1&destination=-6.2381,106.9754",
        "slug": "ud-karya-sawit-mandiri",
    },
    "gula": {
        "name": "PT Gula Kristal Nusantara",
        "verified": True,
        "role": "Pabrik Industri · Cirebon",
        "ai_score": 91,
        "tags": ["Pabrik Langsung", "MOQ Besar", "Harga Industri"],
        "avg_price": "Rp 14.800 /kg",
        "rating": 4.7,
        "reviews_count": 178,
        "lead_time": "48 jam pengiriman",
        "top_products": ["Gula Pasir Putih 25kg", "Gula Merah Aren", "Gula Industri"],
        "maps_url": "https://www.google.com/maps/dir/?api=1&destination=-6.7324,108.5523",
        "slug": "pt-gula-kristal-nusantara",
    },
    "tepung": {
        "name": "CV Berkah Terigu Sejahtera",
        "verified": True,
        "role": "Distributor Terigu · Semarang",
        "ai_score": 89,
        "tags": ["Pengiriman Luas", "Kualitas SNI", "MOQ Rendah"],
        "avg_price": "Rp 9.800 /kg",
        "rating": 4.7,
        "reviews_count": 321,
        "lead_time": "24 jam pengiriman",
        "top_products": ["Tepung Segitiga Biru 25kg", "Tepung Cakra Kembar", "Tepung Beras"],
        "maps_url": "https://www.google.com/maps/dir/?api=1&destination=-6.9932,110.4203",
        "slug": "cv-berkah-terigu-sejahtera",
    },
}

COMPARISON_DATA_MOCK = {
    "beras": [
        {"name": "CV Berkah Pangan (Bandung)", "price": "Rp 12.800 /kg", "lead_time": "24 jam", "rating": 4.8, "saw_score": 95, "status": "Termurah & Terpercaya"},
        {"name": "PT Agro Makmur (Jakarta)", "price": "Rp 12.300 /kg", "lead_time": "36 jam", "rating": 4.6, "saw_score": 92, "status": "MOQ Skala Besar"},
        {"name": "UD Sumber Rezeki (Cimahi)", "price": "Rp 13.200 /kg", "lead_time": "12 jam", "rating": 4.5, "saw_score": 88, "status": "Pengiriman Tercepat"},
    ],
    "minyak": [
        {"name": "UD Karya Sawit Mandiri (Bekasi)", "price": "Rp 15.200 /liter", "lead_time": "36 jam", "rating": 4.6, "saw_score": 91, "status": "Agen Resmi"},
        {"name": "PT Palmindo Sejati (Jakarta)", "price": "Rp 14.900 /liter", "lead_time": "48 jam", "rating": 4.4, "saw_score": 87, "status": "Harga Termurah"},
        {"name": "CV Nusantara Oil (Tangerang)", "price": "Rp 15.500 /liter", "lead_time": "24 jam", "rating": 4.7, "saw_score": 89, "status": "Pengiriman Cepat"},
    ],
    "default": [
        {"name": "CV Berkah Pangan (Bandung)", "price": "Rp 12.800 /kg", "lead_time": "24 jam", "rating": 4.8, "saw_score": 95, "status": "Pilihan Terbaik AI"},
        {"name": "PT Agro Makmur (Jakarta)", "price": "Rp 12.300 /kg", "lead_time": "36 jam", "rating": 4.6, "saw_score": 92, "status": "Harga Kompetitif"},
        {"name": "UD Sumber Rezeki (Cimahi)", "price": "Rp 13.200 /kg", "lead_time": "12 jam", "rating": 4.5, "saw_score": 88, "status": "Pengiriman Tercepat"},
    ],
}

RAG_RESPONSES_FALLBACK = {
    "het": "Berdasarkan Peraturan Badan Pangan Nasional 2026, HET Beras Medium untuk wilayah Jawa adalah Rp 13.900/kg, sedangkan Beras Premium Rp 15.400/kg. Pastikan supplier yang Anda pilih menjual di bawah batas HET untuk menghindari sanksi.",
    "fifo": "Metode FIFO (First In First Out) sangat disarankan untuk manajemen stok bahan pangan. Barang yang masuk lebih awal harus dikeluarkan lebih dahulu untuk mencegah kadaluwarsa dan kerugian stok.",
    "sni": "Standar Nasional Indonesia (SNI) untuk Beras mengharuskan kadar air maksimal 14%, butir kepala minimum 85% untuk kelas premium. Pastikan supplier memiliki sertifikat SNI yang valid.",
    "default": "Berdasarkan regulasi Badan Pangan Nasional dan panduan rantai pasok BakuLink: disarankan memilih supplier dengan lencana Verified BakuLink untuk menjamin akurasi kuantitas, kualitas sesuai standar SNI, serta garansi penggantian untuk kerusakan selama pengiriman. Gunakan fitur TLC Calculator untuk membandingkan total biaya pengadaan secara akurat.",
}

def get_rag_response(query: str, chat_history: list = None) -> str:
    """
    Mengambil konteks dari ChromaDB Vector DB dan menghasilkan respons LLM alami via Google Gemini API.
    Jika Gemini API Key belum ada atau error, fallback ke teks RAG / rule-based.
    """
    global rag_collection, gemini_model
    q = query.lower()

    # 1. Query ChromaDB Vector Collection untuk mendapatkan chunk konteks terelasi
    retrieved_docs = []
    if rag_collection is not None:
        try:
            results = rag_collection.query(query_texts=[query], n_results=3)
            docs = results.get("documents", [[]])[0]
            if docs:
                retrieved_docs = docs
        except Exception as e:
            print(f"[RAG] Error querying ChromaDB: {e}")

    rag_context = "\n---\n".join(retrieved_docs) if retrieved_docs else ""

    # 2. Panggil Google Gemini LLM jika API Key terpasang
    if gemini_model is not None:
        try:
            prompt = f"""Anda adalah BakuLink AI Procurement Advisor, konsultan pengadaan cerdas untuk UMKM bahan pangan di Indonesia.
Tugas Anda adalah memberikan saran pengadaan, regulasi pasar (HET/SNI), serta panduan rantai pasok secara ramah, profesional, dan solutif.

[DOKUMEN KNOWLEDGE BASE (RAG)]:
{rag_context if rag_context else "Gunakan pengetahuan umum rantai pasok pangan dan regulasi Badan Pangan Nasional Indonesia."}

[PERTANYAAN UMKM]:
{query}

Jawablah langsung secara kontekstual, singkat, jelas, berwawasan bisnis UMKM, dan gunakan format Markdown (seperti bold atau bullet points) agar mudah dibaca."""

            response = gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[Gemini LLM] Error calling Gemini API: {e}")

    # 3. Fallback jika Gemini API Key belum dipasang
    if retrieved_docs:
        return f"Berdasarkan Knowledge Base BakuLink:\n\n{retrieved_docs[0]}"

    if any(kw in q for kw in ["het", "harga eceran", "harga tertinggi", "batas harga"]):
        return RAG_RESPONSES_FALLBACK["het"]
    if any(kw in q for kw in ["fifo", "stok", "manajemen persediaan", "inventory"]):
        return RAG_RESPONSES_FALLBACK["fifo"]
    if any(kw in q for kw in ["sni", "standar", "kualitas", "mutu", "sertifikat"]):
        return RAG_RESPONSES_FALLBACK["sni"]

    return RAG_RESPONSES_FALLBACK["default"]


# ======================================================================
# ENDPOINTS
# ======================================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "BakuLink AI Microservice v2",
        "endpoints": [
            "/predict-price",
            "/predict-demand",
            "/api/chat-advisor",
            "/rank-suppliers-saw",
            "/rank-suppliers-custom",
        ]
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
# Helper: Find Demand Model File
# ----------------------------------------------------------------------
def locate_demand_model(pattern_name: str):
    pattern_clean = pattern_name.lower().strip()
    possible_dirs = [
        os.path.join("models", "Model Demand Forecasting"),
        os.path.join("models", "model demand forecasting"),
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

    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            X_input = pd.DataFrame(
                [[req.unit_price, req.is_high_season, req.day_of_week, req.month]],
                columns=['unit_price', 'is_high_season', 'day_of_week', 'month']
            )
            raw_pred = model.predict(X_input)
            daily_pred = float(raw_pred[0])
            method_used = f"Random Forest ({pattern.capitalize()})"
        except Exception:
            daily_pred = None

    if daily_pred is None:
        daily_pred = float(np.mean(req.recent_sales_history)) if req.recent_sales_history else 500.0
        method_used = f"Simple Moving Average (Fallback - {pattern.capitalize()})"

    predicted_7days_kg = int(round(daily_pred * 7))
    predicted_daily_kg = round(daily_pred, 1)
    recent_avg = float(np.mean(req.recent_sales_history)) if req.recent_sales_history else daily_pred

    pct_change = round(((daily_pred - recent_avg) / recent_avg) * 100, 1) if recent_avg > 0 else 0.0
    alert_triggered = abs(pct_change) >= 10.0

    safety_buffer_pct = 15 if pattern == "musiman" else (20 if pattern == "fluktuatif" else 10)
    target_stock_kg = int(round(predicted_7days_kg * (1 + safety_buffer_pct / 100.0)))

    demand_factor = pct_change / 100.0
    rec_min = int(round(req.unit_price * (1 + max(0, demand_factor * 0.05))))
    rec_max = int(round(req.unit_price * (1 + max(0, demand_factor * 0.15))))

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


# ----------------------------------------------------------------------
# ENDPOINT: POST /api/chat-advisor
# Procurement AI Advisor dengan Intent Detection & Structured Response
# ----------------------------------------------------------------------
@app.post("/api/chat-advisor")
def chat_advisor(req: ChatAdvisorRequest):
    intent = detect_intent(req.user_message)
    intent_type = intent["type"]
    commodity = intent.get("commodity")

    # ── Intent: COMPARISON_TABLE ────────────────────────────────────────
    if intent_type == "COMPARISON_TABLE":
        comp_key = commodity if commodity in COMPARISON_DATA_MOCK else "default"
        commodity_label = commodity.capitalize() if commodity else "Bahan Pangan"

        # Jika ada supplier_context dari Laravel DB, gunakan itu
        if req.supplier_context and len(req.supplier_context) >= 2:
            suppliers_data = []
            for s in req.supplier_context[:3]:
                suppliers_data.append({
                    "name": s.get("name", "Supplier BakuLink"),
                    "price": f"Rp {int(s.get('price', 0)):,} /kg".replace(",", "."),
                    "lead_time": f"{s.get('lead_time_hours', 24)} jam",
                    "rating": float(s.get("rating", 4.5)),
                    "saw_score": int(s.get("saw_score", 85)),
                    "status": s.get("status", "Supplier Terverifikasi"),
                })
        else:
            suppliers_data = COMPARISON_DATA_MOCK[comp_key]

        return {
            "status": "success",
            "type": "COMPARISON_TABLE",
            "ai_summary": f"Berikut adalah tabel perbandingan 3 supplier {commodity_label} berdasarkan Total Landed Cost (TLC), Lead Time Pengiriman, dan Rating Reputasi BakuLink:",
            "suppliers": suppliers_data
        }

    # ── Intent: SINGLE_SUPPLIER_CARD ────────────────────────────────────
    if intent_type == "SINGLE_SUPPLIER_CARD":
        commodity_label = commodity.capitalize() if commodity else "Bahan Pangan"

        # Jika ada supplier_context dari Laravel DB, gunakan top-1
        if req.supplier_context and len(req.supplier_context) >= 1:
            s = req.supplier_context[0]
            supplier_data = {
                "name": s.get("name", "Supplier BakuLink"),
                "verified": bool(s.get("verified", True)),
                "role": s.get("role", "Supplier Terverifikasi · BakuLink"),
                "ai_score": int(s.get("saw_score", s.get("ai_score", 90))),
                "tags": s.get("tags", ["Supplier Terverifikasi", "STOK SIAP"]),
                "avg_price": f"Rp {int(s.get('price', 0)):,} /kg".replace(",", "."),
                "rating": float(s.get("rating", 4.5)),
                "reviews_count": int(s.get("review_count", 50)),
                "lead_time": f"{s.get('lead_time_hours', 24)} jam pengiriman",
                "top_products": s.get("products", [commodity_label]),
                "maps_url": s.get("maps_url", ""),
                "slug": s.get("slug", ""),
            }
        else:
            # Mock data per komoditas
            mock_key = commodity if commodity in SUPPLIER_DATA_MOCK else list(SUPPLIER_DATA_MOCK.keys())[0]
            supplier_data = SUPPLIER_DATA_MOCK[mock_key]

        return {
            "status": "success",
            "type": "SINGLE_SUPPLIER_CARD",
            "ai_summary": f"Berdasarkan AI Trust Score (SAW) dan analisis reputasi BakuLink, {supplier_data['name']} adalah pilihan supplier {commodity_label} terbaik saat ini.",
            "data": supplier_data
        }

    # ── Intent: TEXT_RAG (Pertanyaan Umum / Regulasi) ───────────────────
    rag_answer = get_rag_response(req.user_message)
    return {
        "status": "success",
        "type": "TEXT_RAG",
        "ai_summary": rag_answer,
        "data": None
    }


# ----------------------------------------------------------------------
# ENDPOINT: POST /rank-suppliers-saw (SAW Compute Engine — Mode Gabungan)
# ----------------------------------------------------------------------
@app.post("/rank-suppliers-saw")
def rank_suppliers_saw(req: RankSuppliersSAWRequest):
    final_weight, raw_results = compute_weights_and_tlc(
        req.length_cm, req.width_cm, req.height_cm,
        req.actual_weight_kg, req.quantity, req.suppliers
    )

    if not raw_results:
        return {"status": "success", "final_weight_kg": final_weight, "rankings": []}

    # SAW Normalization (33.33% weights each)
    min_tlc = min(r['tlc'] for r in raw_results)
    min_time = min(r['effective_lead_time'] for r in raw_results)
    max_rating = max(r['bayesian_rating'] for r in raw_results)

    for r in raw_results:
        r_tlc = min_tlc / r['tlc'] if r['tlc'] > 0 else 1.0
        r_time = min_time / r['effective_lead_time'] if r['effective_lead_time'] > 0 else 1.0
        r_rating = r['bayesian_rating'] / 5.0

        r['final_saw_score'] = round(((r_tlc * 0.333) + (r_time * 0.333) + (r_rating * 0.334)) * 100, 1)

    raw_results.sort(key=lambda x: x['final_saw_score'], reverse=True)

    return {
        "status": "success",
        "final_weight_kg": round(final_weight, 2),
        "rankings": raw_results
    }


# ----------------------------------------------------------------------
# ENDPOINT: POST /rank-suppliers-custom (Aspek Tunggal Mutlak)
# ----------------------------------------------------------------------
@app.post("/rank-suppliers-custom")
def rank_suppliers_custom(req: RankSuppliersCustomRequest):
    """
    Mode ranking aspek tunggal:
    - cheapest: TLC terendah (tie-break: lead time tercepat)
    - fastest: Lead Time Efektif tercepat (standar + avg keterlambatan)
    - quality: Bayesian Average Rating tertinggi
    """
    final_weight, raw_results = compute_weights_and_tlc(
        req.length_cm, req.width_cm, req.height_cm,
        req.actual_weight_kg, req.quantity, req.suppliers
    )

    if not raw_results:
        return {"status": "success", "mode": req.mode, "final_weight_kg": final_weight, "rankings": []}

    mode = req.mode.lower()

    if mode == "cheapest":
        sorted_results = sorted(raw_results, key=lambda x: (x['tlc'], x['effective_lead_time']))
        sort_label = "Total Landed Cost (TLC) Termurah"
    elif mode == "fastest":
        sorted_results = sorted(raw_results, key=lambda x: x['effective_lead_time'])
        sort_label = "Lead Time Efektif Tercepat"
    elif mode == "quality":
        sorted_results = sorted(raw_results, key=lambda x: x['bayesian_rating'], reverse=True)
        sort_label = "Bayesian Rating Tertinggi"
    else:
        raise HTTPException(status_code=400, detail=f"Mode '{mode}' tidak valid. Gunakan: cheapest, fastest, quality.")

    # Tambahkan rank
    for i, r in enumerate(sorted_results):
        r['rank'] = i + 1

    return {
        "status": "success",
        "mode": mode,
        "sort_label": sort_label,
        "final_weight_kg": round(final_weight, 2),
        "rankings": sorted_results
    }
# Catatan AI 3: Procurement AI Advisor (Asisten Chatbot Pengadaan) 🤖

Dokumen ini menjelaskan rancangan, spesifikasi data, dan langkah demi langkah pembuatan asisten pengadaan pintar berbasis RAG (Retrieval-Augmented Generation) menggunakan ChromaDB di BakuLink.

---

## 📝 1. Deskripsi AI
Procurement AI Advisor adalah chatbot interaktif di dalam aplikasi BakuLink. Chatbot ini bertindak sebagai konsultan bisnis pintar bagi UMKM dan supplier untuk menanyakan kebijakan perdagangan, regulasi harga pangan pemerintah (HET), manajemen persediaan barang, serta tips negosiasi bisnis.

*   **Tipe AI:** RAG (Retrieval-Augmented Generation) Chatbot.
*   **LLM Engine:** **Google Gemini Flash 1.5 API** (Gratis & Sangat Cepat untuk Lomba) / OpenAI GPT-4o-mini.
*   **Vector Store:** **ChromaDB** (Database Vektor Lokal).

---

## 📊 2. Spesifikasi Data untuk Indexing (ChromaDB)
AI Advisor membutuhkan basis dokumen pengetahuan (Knowledge Base) dalam format dokumen teks agar bisa menjawab secara akurat sesuai regulasi di Indonesia.

*   **Sumber Dokumen Pengetahuan:**
    1.  PDF/Teks Regulasi Kementerian Perdagangan & Badan Pangan Nasional tentang Harga Eceran Tertinggi (HET) Beras, Minyak Goreng, dll.
    2.  Artikel/Panduan tentang manajemen rantai pasok kuliner & UMKM.
    3.  Pola standar mutu beras (Beras Premium vs. Medium).
*   **Format Data:** Kumpulan file teks `.txt` atau dokumen `.pdf` yang nantinya dipecah menjadi bagian-bagian kecil (chunking) dan dikonversi menjadi vektor numerik (*embeddings*).

---

## 💻 3. Langkah demi Langkah Pembuatan Vector DB di Google Colab

Kita akan memproses dokumen mentah menjadi basis data vektor ChromaDB menggunakan Google Colab.

### Langkah 3.1: Install Library Pendukung
Buka Google Colab baru, dan install library berikut:
```python
!pip install chromadb sentence-transformers langchain-textsplitters
```

### Langkah 3.2: Persiapan Dokumen Pengetahuan
Tulis dokumen pengetahuan Anda ke dalam teks (atau unggah file PDF regulasi):
```python
documents = [
    {
        "id": "het-beras-2026",
        "text": "Berdasarkan Peraturan Badan Pangan Nasional Nomor 7 Tahun 2026, Harga Eceran Tertinggi (HET) Beras Medium untuk wilayah Jawa, Lampung, dan Sumatra Selatan ditetapkan sebesar Rp 13.900 per kilogram. Untuk Beras Premium ditetapkan sebesar Rp 15.400 per kilogram."
    },
    {
        "id": "manajemen-stok-umkm",
        "text": "Tips manajemen stok untuk UMKM kuliner: Selalu gunakan metode FIFO (First In First Out) terutama untuk komoditas cepat busuk seperti cabai dan bawang. Batasi stok cabai segar maksimal untuk kebutuhan 3 hari produksi."
    },
    {
        "id": "standar-mutu-minyak",
        "text": "Standar mutu Minyak Goreng Curah dan Kemasan yang aman dikonsumsi wajib memenuhi kriteria SNI, memiliki kadar air maksimal 0,1% dan kandungan asam lemak bebas maksimal 0,3%."
    }
]
```

### Langkah 3.3: Embed & Simpan Dokumen ke ChromaDB
Kita akan membuat database vektor lokal di Google Colab dan menyimpannya ke dalam sebuah direktori bernama `chroma_db`:
```python
import chromadb
from chromadb.utils import embedding_functions

# Inisialisasi ChromaDB client dengan penyimpanan persistent (ke folder lokal)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Gunakan model embedding gratis dari HuggingFace (Mendukung multibahasa/Indonesia)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Buat koleksi baru di database
collection = chroma_client.create_collection(
    name="bakulink_knowledge", 
    embedding_function=emb_fn
)

# Masukkan dokumen ke dalam Vector Store
collection.add(
    documents=[doc["text"] for doc in documents],
    ids=[doc["id"] for doc in documents],
    metadatas=[{"source": "regulasionline"} for _ in documents]
)

print("Vector Database ChromaDB berhasil dibuat!")
```

### Langkah 3.4: Uji Coba Pencarian Semantik (Semantic Search)
Mari kita tes apakah database bisa mencari dokumen yang relevan secara makna kata:
```python
# Cari dokumen terkait HET beras
results = collection.query(
    query_texts=["berapa harga eceran tertinggi beras medium di jawa?"],
    n_results=1
)

print("Dokumen Relevan Ditemukan:")
print(results['documents'][0][0])
```

### Langkah 3.5: Ekspor/Zip Folder ChromaDB (Hasil Akhir)
Kompres folder database yang sudah terisi menjadi berkas `.zip` agar bisa diunduh:
```python
!zip -r chroma_db.zip ./chroma_db
```
*Unduh file `chroma_db.zip` ke komputer Anda.*

---

## 💾 4. Cara Penggunaan chroma_db di FastAPI

1.  Ekstrak file `chroma_db.zip` hasil unduhan dari Colab.
2.  Pindahkan isinya ke dalam folder proyek Anda di:
    `bakulink-service-ai/chroma_db/`
3.  Di file `main.py` FastAPI, Anda bisa menyambungkan ChromaDB dengan API Google Gemini untuk menjawab pertanyaan:

```python
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
import google.generativeai as genai

app = FastAPI()

# Konfigurasi Gemini API Key
genai.configure(api_key="API_KEY_GEMINI_ANDA")
llm_model = genai.GenerativeModel('gemini-1.5-flash')

# Muat ChromaDB yang sudah di-extract
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="bakulink_knowledge", embedding_function=emb_fn)

@app.post("/advisor")
def ask_advisor(question: str):
    # 1. Cari dokumen relevan di ChromaDB
    search_results = collection.query(query_texts=[question], n_results=1)
    context = search_results['documents'][0][0] if search_results['documents'] else ""
    
    # 2. Susun prompt RAG untuk Gemini
    prompt = f"""
    Kamu adalah Procurement AI Advisor bernama BakuLink Advisor.
    Jawablah pertanyaan user secara profesional dan ramah menggunakan Konteks Referensi di bawah ini.
    Jika tidak ada di referensi, jawablah dengan pengetahuan umum terbaikmu namun beri tahu jika itu di luar regulasi resmi BakuLink.
    
    Konteks Referensi:
    {context}
    
    Pertanyaan: {question}
    Jawaban:
    """
    
    # 3. Minta jawaban dari Gemini LLM
    response = llm_model.generate_content(prompt)
    
    return {
        "status": "success",
        "answer": response.text,
        "source_context": context
    }
```
*Sekarang chatbot AI Anda siap memandu para UMKM dengan basis data regulasi yang kredibel dan real-time!*

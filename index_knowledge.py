import os
import chromadb
from chromadb.utils import embedding_functions

def build_knowledge_base():
    """
    Script untuk membuat dan mengisi ChromaDB Vector Database dengan dokumen
    pengetahuan pengadaan BakuLink (Regulasi HET, Standar SNI, Panduan FIFO, TLC, MOQ).
    """
    db_path = "chroma_db"
    print(f"[Indexing] Memulai pembuatan Vector Database di '{db_path}'...")

    # Persistent client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Embedding model multilingual
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name="bakulink_knowledge",
        embedding_function=emb_fn
    )

    # Dokumen Pengetahuan Pengadaan BakuLink
    documents = [
        {
            "id": "het-beras-2026",
            "text": "Berdasarkan Peraturan Badan Pangan Nasional Nomor 7 Tahun 2026, Harga Eceran Tertinggi (HET) Beras Medium untuk wilayah Jawa, Lampung, dan Sumatra Selatan ditetapkan sebesar Rp 13.900 per kilogram. Untuk Beras Premium ditetapkan sebesar Rp 15.400 per kilogram. UMKM disarankan membeli bahan baku beras dari supplier terverifikasi yang mematuhi batas HET."
        },
        {
            "id": "manajemen-stok-fifo",
            "text": "Panduan Manajemen Persediaan UMKM Kuliner: Gunakan metode FIFO (First In First Out). Bahan baku komoditas segar seperti cabai, telur, dan daging harus digunakan dalam rentang 3-5 hari. Batasi penumpukan stok beras dan tepung maksimal untuk kebutuhan 30 hari produksi guna mencegah kerusakan dari kelembapan."
        },
        {
            "id": "standar-sni-pangan",
            "text": "Standar Nasional Indonesia (SNI) Komoditas Pangan: Beras Premium kelas 1 wajib memiliki kadar air maksimal 14%, derajat sosoh 100%, dan butir kepala minimal 95%. Minyak goreng sawit kemasan wajib memenuhi SNI 7709:2019 terfortifikasi Vitamin A. Tepung terigu konsumsi wajib memenuhi SNI 3751:2018."
        },
        {
            "id": "total-landed-cost-tlc",
            "text": "Total Landed Cost (TLC) BakuLink adalah total biaya riil pengadaan bahan baku yang dihitung dari: (Harga Produk x Kuantitas) + Biaya Ongkir Cargo 3PL (berdasarkan Berat Dimensional Volume) + Biaya Penanganan Admin (1.5%). Pengadaan melalui perbandingan TLC terbukti menghemat hingga 15% biaya operasional UMKM."
        },
        {
            "id": "moq-dan-saw-engine",
            "text": "Strategi MOQ (Minimum Order Quantity) & SAW Engine: BakuLink AI menggunakan Simple Additive Weighting (SAW) dengan bobot 33.3% TLC, 33.3% Lead Time Pengiriman, dan 33.4% Bayesian Rating untuk merekomendasikan supplier optimal. Pembeli skala kecil disarankan memilih supplier dengan MOQ rendah (di bawah 25kg)."
        },
        {
            "id": "kebijakan-garansi-bakulink",
            "text": "Garansi Pengadaan BakuLink: Semua transaksi yang dilakukan via platform BakuLink mendapatkan perlindungan Garansi Mutu & Kuantitas. Jika komoditas yang diterima mengalami busuk, rusak saat pengiriman logistik 3PL, atau selisih timbangan, pembeli berhak mengajukan klaim pengembalian dana 100% dalam 1x24 jam setelah penyerahan barang."
        }
    ]

    # Insert / Upsert ke ChromaDB
    collection.upsert(
        documents=[doc["text"] for doc in documents],
        ids=[doc["id"] for doc in documents]
    )

    print(f"✅ ChromaDB Knowledge Base berhasil diisi dengan {len(documents)} dokumen RAG!")
    print("Dokumen terindeks: " + ", ".join([doc["id"] for doc in documents]))

if __name__ == "__main__":
    build_knowledge_base()

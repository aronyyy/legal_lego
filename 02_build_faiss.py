"""
02_build_faiss.py
=================
NYAYA-SETU — Stage 2: Embedding + FAISS Index (CPU-only)
Reads chunks_master.parquet → embeds with MiniLM → writes FAISS index + metadata parquet.

⚠️  Databricks Free Edition has NO GPU.
    This script uses CPU-only sentence-transformers + faiss-cpu.
    On ~42k chunks expect ~8–12 min on CPU. Reduce BATCH_SIZE if OOM.

Run:  python scripts/02_build_faiss.py
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR      = os.getenv("OUT_DIR",      "/dbfs/FileStore/nyaya_setu/output")
MASTER_PATH  = os.getenv("MASTER_PATH",  f"{OUT_DIR}/chunks_master.parquet")
INDEX_PATH   = os.getenv("INDEX_PATH",   f"{OUT_DIR}/faiss_precedents.index")
META_PATH    = os.getenv("META_PATH",    f"{OUT_DIR}/faiss_metadata.parquet")

# MiniLM-L6 is 80MB, runs well on CPU, 384-dim embeddings
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM    = 384
# CPU batch: keep low to avoid OOM on 15GB RAM Databricks Free
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "64"))

os.makedirs(OUT_DIR, exist_ok=True)


# ── LOAD CHUNKS ───────────────────────────────────────────────────────────────
print("=" * 60)
print("📥  Loading chunks_master.parquet …")
print("=" * 60)

df = pd.read_parquet(MASTER_PATH)
df["embed_text"] = df["embed_text"].fillna("").astype(str)
df = df[df["embed_text"].str.strip().str.len() > 10].reset_index(drop=True)
df["faiss_id"] = range(len(df))

print(f"   Chunks loaded  : {len(df):,}")
print(f"   Chunk types    :\n{df['chunk_type'].value_counts().to_string()}\n")

texts = df["embed_text"].tolist()
n     = len(texts)


# ── LOAD MODEL (CPU) ─────────────────────────────────────────────────────────
print(f"🔄  Loading embedding model on CPU: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device="cpu")
model.max_seq_length = 256
print("✅  Model loaded\n")


# ── EMBED (CPU, chunked to avoid OOM) ────────────────────────────────────────
print(f"🚀  Encoding {n:,} chunks on CPU (batch={BATCH_SIZE}) …")
print("    This takes ~8–15 min on Databricks Free. Grab a chai ☕")

t_start = time.time()

vectors = model.encode(
    texts,
    batch_size          = BATCH_SIZE,
    show_progress_bar   = True,
    normalize_embeddings= True,
    convert_to_numpy    = True,
)
vectors = vectors.astype("float32")

elapsed = time.time() - t_start
print(f"\n⏱️  Encoding done: {elapsed:.1f}s  ({n/max(elapsed,1):.0f} chunks/sec)\n")

# Sanity check
norms = np.linalg.norm(vectors, axis=1)
print(f"   Matrix shape : {vectors.shape}")
print(f"   Memory       : {vectors.nbytes / 1e6:.1f} MB")
print(f"   Norm range   : {norms.min():.4f} – {norms.max():.4f}  (should be ≈1.0)\n")
del model; gc.collect()


# ── BUILD FAISS INDEX (CPU) ───────────────────────────────────────────────────
print("🏗️   Building FAISS IndexFlatIP (CPU) …")
index_flat = faiss.IndexFlatIP(EMBED_DIM)
index      = faiss.IndexIDMap(index_flat)
ids        = df["faiss_id"].values.astype("int64")
index.add_with_ids(vectors, ids)
print(f"✅  Index built.  Total vectors: {index.ntotal:,}\n")
del vectors; gc.collect()


# ── SMOKE TEST ────────────────────────────────────────────────────────────────
print("🧪  Smoke-test query …")

# Re-load model just for the test vector (small — one sentence)
embedder     = SentenceTransformer(MODEL_NAME, device="cpu")
test_query   = "standard of proof in motor vehicle accident compensation claim"
q_vec        = embedder.encode([test_query], normalize_embeddings=True).astype("float32")
D, I         = index.search(q_vec, k=5)

print(f"   Query: \"{test_query}\"")
for rank, (score, idx) in enumerate(zip(D[0], I[0])):
    row = df[df["faiss_id"] == idx]
    if row.empty:
        continue
    row = row.iloc[0]
    print(f"   [{rank+1}] score={score:.4f}  |  "
          f"{str(row.get('case_name',''))[:50]}  |  "
          f"para={row.get('held_para','?')}")
del embedder, q_vec; gc.collect()


# ── SAVE FAISS INDEX ──────────────────────────────────────────────────────────
print(f"\n💾  Saving FAISS index → {INDEX_PATH}")
faiss.write_index(index, INDEX_PATH)
print(f"   Size: {os.path.getsize(INDEX_PATH)/1e6:.1f} MB")


# ── SAVE METADATA ─────────────────────────────────────────────────────────────
print(f"💾  Saving metadata → {META_PATH}")
keep_cols = [c for c in [
    "faiss_id", "chunk_id", "chunk_type", "held_para",
    "case_name", "citation", "year", "source_file",
    "legal_domain", "acts_involved", "judges", "date",
    "chunk_text", "embed_text", "word_count",
] if c in df.columns]

df[keep_cols].to_parquet(META_PATH, index=False)
print(f"   Rows : {len(df):,}")
print(f"   Size : {os.path.getsize(META_PATH)/1e6:.1f} MB")

print("\n" + "=" * 60)
print("✅  Stage 2 complete — Run 03_build_bns_ipc_map.py next")
print("=" * 60)

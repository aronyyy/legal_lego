"""
04_langchain_pipeline.py  (also copied as scripts/langchain_pipeline.py)
=========================================================================
NYAYA-SETU — Dual-RAG LangChain Pipeline (Groq / CPU-only)

Importable by app.py AND runnable standalone:
    python scripts/04_langchain_pipeline.py

Design rules for Databricks Apps:
  - NO module-level side-effects (no prints/IO at import time).
  - GROQ_API_KEY is read lazily inside build_chain(), not at import.
  - All paths from env vars so app.yaml / App config controls them.
  - Works with pre-built files: just point the 3 env vars at your files.
"""

from __future__ import annotations

import os
import textwrap
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
# Default = DBFS path for Databricks.
# Override with env vars or pass explicit paths into load_pipeline_artifacts().
OUT_DIR          = os.getenv("OUT_DIR",    "/dbfs/FileStore/nyaya_setu/output")
FAISS_INDEX_PATH = os.getenv("INDEX_PATH", f"{OUT_DIR}/faiss_precedents.index")
FAISS_META_PATH  = os.getenv("META_PATH",  f"{OUT_DIR}/faiss_metadata.parquet")
BNS_CSV_PATH     = os.getenv("BNS_CSV",    f"{OUT_DIR}/bns_ipc_mapping.csv")

MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
GROQ_TEMP       = 0.1
GROQ_MAX_TOKENS = 2048


# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_pipeline_artifacts(
    index_path: str = FAISS_INDEX_PATH,
    meta_path:  str = FAISS_META_PATH,
    bns_path:   str = BNS_CSV_PATH,
) -> tuple:
    """
    Load the 3 pre-built output files.
    Accepts either DBFS paths (/dbfs/...) or local paths — both work identically.
    Returns (faiss_index, df_meta, df_bns).
    """
    print(f"📥  FAISS index  ← {index_path}")
    faiss_index = faiss.read_index(index_path)

    print(f"📥  Metadata     ← {meta_path}")
    df_meta = pd.read_parquet(meta_path)

    print(f"📥  BNS-IPC map  ← {bns_path}")
    df_bns = pd.read_csv(bns_path)

    print(f"✅  Loaded: {faiss_index.ntotal:,} vectors | "
          f"{len(df_meta):,} meta rows | {len(df_bns):,} BNS sections")
    return faiss_index, df_meta, df_bns


def load_embedder(device: str = "cpu") -> SentenceTransformer:
    """Load MiniLM on CPU. ~80 MB on first run, cached by HuggingFace after that."""
    print(f"🔢  Loading embedder on {device.upper()} …")
    embedder = SentenceTransformer(MODEL_NAME, device=device)
    embedder.max_seq_length = 256
    print("✅  Embedder ready")
    return embedder


# ── HELPER ────────────────────────────────────────────────────────────────────

def clean_case_name(raw_name, citation: str = "", year=None) -> str:
    if pd.notna(raw_name) and str(raw_name).strip() not in ("None", "", "nan"):
        return str(raw_name).strip()
    if citation and str(citation).strip() not in ("None", "", "nan"):
        return str(citation).strip()
    try:
        yr = str(int(float(year))) if year and not pd.isna(year) else "Year Unknown"
    except Exception:
        yr = "Year Unknown"
    return f"Supreme Court Judgment ({yr})"


# ── CHAIN BUILDER ─────────────────────────────────────────────────────────────

def build_chain(faiss_index, df_meta, df_bns, embedder):
    """
    Build the LangChain LCEL pipeline.
    Reads GROQ_API_KEY from env at call time (safe to import without the key set).
    Returns a chain — call: response = chain.invoke("case facts as a string")
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "  Databricks App → App config → Environment variables\n"
            "  Local testing  → export GROQ_API_KEY=gsk_xxx\n"
            "  Streamlit UI   → enter key in the sidebar"
        )

    # Pre-embed all BNS rows once (cheap, ~50 rows)
    bns_vectors = embedder.encode(
        df_bns["chunk_text"].tolist(),
        convert_to_numpy    =True,
        normalize_embeddings=True,
        show_progress_bar   =False,
    ).astype("float32")

    # ── DUAL-RAG RETRIEVER ────────────────────────────────────────────────────
    def retrieve_dual_context(lawyer_query: str) -> str:
        # Phase A — BNS statute via cosine similarity on the mapping CSV
        q_vec = embedder.encode(
            [lawyer_query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        sims        = np.dot(bns_vectors, q_vec.T).flatten()
        best_idx    = int(np.argmax(sims))
        bns_row     = df_bns.iloc[best_idx]
        bns_statute = bns_row["chunk_text"]
        bns_section = bns_row.get("BNS_Section", "?")
        ipc_section = bns_row.get("IPC_Section", "?")
        confidence  = bns_row.get("Confidence_Score", "?")
        status      = bns_row.get("Status", "?")

        # Phase B — enriched FAISS search over Supreme Court precedents
        enriched = f"{lawyer_query} {bns_statute}"
        q_enr    = embedder.encode(
            [enriched], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        D, I = faiss_index.search(q_enr, k=4)

        seen, unique = set(), []
        for score, idx in zip(D[0], I[0]):
            rows = df_meta[df_meta["faiss_id"] == idx]
            if rows.empty:
                continue
            row = rows.iloc[0]
            src = str(row.get("source_file", idx))
            if src in seen:
                continue
            seen.add(src)
            unique.append((score, row))
            if len(unique) == 3:
                break

        statute_block = (
            "=== GOVERNING STATUTE ===\n"
            f"BNS Section {bns_section}  <->  IPC Section {ipc_section}  "
            f"[Confidence: {confidence}% | {status}]\n\n"
            f"{bns_statute}\n"
        )
        precedent_block = "=== SUPREME COURT PRECEDENTS ===\n"
        for score, row in unique:
            name    = clean_case_name(row.get("case_name"), row.get("citation", ""), row.get("year"))
            yr      = str(row.get("year", ""))
            domain  = str(row.get("legal_domain", "Criminal"))
            snippet = str(row.get("chunk_text", ""))[:600].strip()
            precedent_block += (
                f"\n[CASE: {name} ({yr})] [Domain: {domain}] [Similarity: {score:.3f}]\n"
                f"{snippet}\n" + "-" * 60 + "\n"
            )

        return statute_block + "\n" + precedent_block

    # ── PROMPT ────────────────────────────────────────────────────────────────
    SYSTEM_PROMPT = (
        "You are Nyaya-Setu, an elite Indian Legal AI Co-Counsel specialising in "
        "criminal defence under the Bharatiya Nyaya Sanhita (BNS) and its predecessor "
        "Indian Penal Code (IPC).\n\n"
        "ABSOLUTE RULES:\n"
        "1. ONLY cite cases from [CASE: ...] tags. NEVER invent case names.\n"
        "2. Generic case names: write 'As held in [Year] Supreme Court precedent'.\n"
        "3. Use the exact five-section structure below.\n"
        "4. Name the BNS Section and its IPC equivalent in your opening paragraph.\n"
        "5. Flag mappings with Confidence < 60% — advise independent verification.\n"
        "6. End with a TACTICAL SUMMARY (2-3 sentences, readable aloud to the judge).\n\n"
        "RESPONSE STRUCTURE:\n"
        "**I. CHARGE & STATUTORY FRAMEWORK**\n"
        "**II. GOVERNING PRINCIPLES FROM PRECEDENT**\n"
        "**III. DEFENCE STRATEGY (bullet points)**\n"
        "**IV. ANTICIPATED PROSECUTION ARGUMENTS & REBUTTALS**\n"
        "**V. TACTICAL SUMMARY**"
    )

    USER_TEMPLATE = (
        "=== RETRIEVED LEGAL CONTEXT ===\n{context}\n\n"
        "=== CASE FACTS ===\n{case_facts}\n\n"
        "Draft a complete, court-ready strategic legal defence covering all five sections."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_TEMPLATE),
    ])

    llm = ChatGroq(
        model      =GROQ_MODEL,
        temperature=GROQ_TEMP,
        max_tokens =GROQ_MAX_TOKENS,
        api_key    =api_key,
    )

    return (
        {
            "context"   : RunnableLambda(retrieve_dual_context),
            "case_facts": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# ── RUNNER ────────────────────────────────────────────────────────────────────

def run_case(chain, facts: str, label: str = "TEST CASE") -> str:
    d = "=" * 70
    print(f"\n{d}\n  {label}\n{d}")
    print(f"\nFACTS:\n{textwrap.fill(facts.strip(), 70)}\n\nRESPONSE:\n")
    response = chain.invoke(facts.strip())
    print(response)
    print(f"\n{d}\n")
    return response


# ── STANDALONE ENTRY POINT ────────────────────────────────────────────────────
if __name__ == "__main__":
    fi, dm, db = load_pipeline_artifacts()
    emb        = load_embedder()
    ch         = build_chain(fi, dm, db, emb)
    run_case(ch,
        "My client is charged under BNS Section 117 for grievous hurt after defending "
        "himself from an armed attack. He claims pure self-defence.",
        "Smoke Test — Self-Defence"
    )

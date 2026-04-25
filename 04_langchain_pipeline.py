"""
04_langchain_pipeline.py
========================
NYAYA-SETU — Stage 4: Dual-RAG LangChain Pipeline (Groq / CPU-only)
Loads FAISS index + BNS-IPC map → builds retrieval chain → runs test cases.

On Databricks:
  - GROQ_API_KEY must be set as a Databricks Secret (or env var for local dev).
  - No GPU needed: MiniLM runs on CPU, Groq is cloud-hosted.

Run standalone:  GROQ_API_KEY=gsk_xxx python scripts/04_langchain_pipeline.py
"""

import os
import gc
import textwrap
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR           = os.getenv("OUT_DIR",         "/dbfs/FileStore/nyaya_setu/output")
FAISS_INDEX_PATH  = os.getenv("INDEX_PATH",      f"{OUT_DIR}/faiss_precedents.index")
FAISS_META_PATH   = os.getenv("META_PATH",       f"{OUT_DIR}/faiss_metadata.parquet")
BNS_CSV_PATH      = os.getenv("BNS_CSV",         f"{OUT_DIR}/bns_ipc_mapping.csv")

MODEL_NAME        = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL        = "llama-3.3-70b-versatile"
GROQ_TEMP         = 0.1
GROQ_MAX_TOKENS   = 2048

# ── API KEY ───────────────────────────────────────────────────────────────────
# In Databricks: store via `dbutils.secrets.put(scope, key, value)`
# Here we read from env var (set by app.py or Databricks job config)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. "
        "On Databricks: set it in the App environment variables. "
        "Locally: export GROQ_API_KEY=gsk_xxx"
    )


# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────────
def load_pipeline_artifacts():
    """Load FAISS index, metadata, and BNS-IPC CSV. Returns (index, df_meta, df_bns)."""
    print("📥  Loading FAISS index …")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    print("📥  Loading FAISS metadata …")
    df_meta = pd.read_parquet(FAISS_META_PATH)

    print("📥  Loading BNS→IPC mapping …")
    df_bns = pd.read_csv(BNS_CSV_PATH)

    return faiss_index, df_meta, df_bns


# ── EMBEDDING MODEL ───────────────────────────────────────────────────────────
def load_embedder():
    """Load CPU MiniLM embedder and pre-embed BNS rows."""
    print("🔢  Loading sentence-transformer on CPU …")
    embedder = SentenceTransformer(MODEL_NAME, device="cpu")
    return embedder


# ── HELPERS ───────────────────────────────────────────────────────────────────
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


# ── PIPELINE BUILDER ──────────────────────────────────────────────────────────
def build_chain(faiss_index, df_meta, df_bns, embedder):
    """
    Build and return the LangChain LCEL pipeline.
    Call chain.invoke(case_facts_string) to get a response.
    """

    # Pre-embed all BNS rows once
    bns_vectors = embedder.encode(
        df_bns["chunk_text"].tolist(),
        convert_to_numpy   = True,
        normalize_embeddings = True,
        show_progress_bar  = False,
    ).astype("float32")

    # ── RETRIEVER ─────────────────────────────────────────────────────────────
    def retrieve_dual_context(lawyer_query: str) -> str:
        # Phase A: best-matching BNS statute
        q_vec = embedder.encode(
            [lawyer_query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        sims         = np.dot(bns_vectors, q_vec.T).flatten()
        best_idx     = int(np.argmax(sims))
        bns_row      = df_bns.iloc[best_idx]
        bns_statute  = bns_row["chunk_text"]
        bns_section  = bns_row.get("BNS_Section", "?")
        ipc_section  = bns_row.get("IPC_Section", "?")
        confidence   = bns_row.get("Confidence_Score", "?")
        status       = bns_row.get("Status", "?")

        # Phase B: enriched FAISS search over Supreme Court precedents
        enriched = f"{lawyer_query} {bns_statute}"
        q_enr    = embedder.encode(
            [enriched], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        D, I = faiss_index.search(q_enr, k=4)

        seen, unique = set(), []
        for score, idx in zip(D[0], I[0]):
            row = df_meta[df_meta["faiss_id"] == idx]
            if row.empty:
                continue
            row = row.iloc[0]
            src = row.get("source_file", str(idx))
            if src in seen:
                continue
            seen.add(src)
            unique.append((score, row))
            if len(unique) == 3:
                break

        # Format context block
        statute_block = (
            f"=== 📜 GOVERNING STATUTE ===\n"
            f"BNS Section {bns_section}  ←→  IPC Section {ipc_section}  "
            f"[Confidence: {confidence}% | {status}]\n\n"
            f"{bns_statute}\n"
        )
        precedent_block = "=== 📚 SUPREME COURT PRECEDENTS ===\n"
        for score, row in unique:
            name    = clean_case_name(row.get("case_name"), row.get("citation", ""), row.get("year"))
            year    = str(row.get("year", ""))
            domain  = str(row.get("legal_domain", "Criminal"))
            snippet = str(row.get("chunk_text", ""))[:600].strip()
            precedent_block += (
                f"\n[CASE: {name} ({year})] [Domain: {domain}] [Similarity: {score:.3f}]\n"
                f"{snippet}\n"
                + "─" * 60 + "\n"
            )

        return statute_block + "\n" + precedent_block

    # ── PROMPT ────────────────────────────────────────────────────────────────
    SYSTEM_PROMPT = """You are Nyaya-Setu, an elite Indian Legal AI Co-Counsel specialising in \
criminal defence under the Bharatiya Nyaya Sanhita (BNS) and its predecessor Indian Penal Code (IPC).

ABSOLUTE RULES:
1. ONLY cite cases explicitly provided in the [CASE: ...] tags of the context. NEVER invent case names.
2. When a case name is generic like "Supreme Court Judgment", write it as \
"As held in [Year] Supreme Court precedent".
3. Structure every response with the exact headings below.
4. Address ALL key legal doctrines relevant to the charge.
5. Identify the BNS Section number and its IPC equivalent in your opening paragraph.
6. Flag any low-confidence mappings (< 60%) and advise the lawyer to verify independently.
7. Finish with a TACTICAL SUMMARY of 2–3 sentences the lawyer can read aloud to the judge.

RESPONSE STRUCTURE:
**I. CHARGE & STATUTORY FRAMEWORK**
**II. GOVERNING PRINCIPLES FROM PRECEDENT**
**III. DEFENCE STRATEGY (bullet points)**
**IV. ANTICIPATED PROSECUTION ARGUMENTS & REBUTTALS**
**V. TACTICAL SUMMARY**"""

    USER_TEMPLATE = """=== RETRIEVED LEGAL CONTEXT ===
{context}

=== CASE FACTS PROVIDED BY THE LAWYER ===
{case_facts}

Draft a complete, court-ready strategic legal defence covering all five required sections."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_TEMPLATE),
    ])

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = ChatGroq(
        model       = GROQ_MODEL,
        temperature = GROQ_TEMP,
        max_tokens  = GROQ_MAX_TOKENS,
        api_key     = GROQ_API_KEY,
    )

    # ── CHAIN ─────────────────────────────────────────────────────────────────
    chain = (
        {
            "context"   : RunnableLambda(retrieve_dual_context),
            "case_facts": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── RUNNER ────────────────────────────────────────────────────────────────────
def run_case(chain, facts: str, label: str = "TEST CASE"):
    divider = "═" * 70
    print(f"\n{divider}")
    print(f"  🏛  {label}")
    print(divider)
    print(f"\n📋  FACTS:\n{textwrap.fill(facts, width=70)}\n")
    print("⚖️   NYAYA-SETU RESPONSE:\n")
    response = chain.invoke(facts)
    print(response)
    print(f"\n{divider}\n")
    return response


# ── MAIN (smoke-test when run directly) ──────────────────────────────────────
if __name__ == "__main__":
    faiss_index, df_meta, df_bns = load_pipeline_artifacts()
    embedder                      = load_embedder()
    chain                         = build_chain(faiss_index, df_meta, df_bns, embedder)

    print("\n" + "=" * 60)
    print("✅  NYAYA-SETU v2.0 ENGINE READY")
    print(f"    Model  : {GROQ_MODEL} (Groq, free tier)")
    print("    Device : CPU  (no GPU needed)")
    print("=" * 60 + "\n")

    # Test case
    facts_1 = """
    My client, a 34-year-old shopkeeper, was closing his shop at night when the complainant
    and two accomplices attacked him with iron rods. In the struggle, my client grabbed
    a nearby wooden plank and struck one of them on the head, causing grievous hurt.
    The police have charged him under BNS Section 117 (voluntarily causing grievous hurt)
    and argue his force was disproportionate. He claims pure self-defence.
    """
    run_case(chain, facts_1, "CASE 1 — Self-Defence (BNS 34 / IPC 99)")

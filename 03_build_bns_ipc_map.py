"""
03_build_bns_ipc_map.py
=======================
NYAYA-SETU — Stage 3: IPC ↔ BNS Autonomous Mapping Engine
Downloads IPC & BNS PDFs → parses sections → TF-IDF cosine similarity → CSV.

CPU-only. No GPU required. scikit-learn TF-IDF is efficient on CPU.

Run:  python scripts/03_build_bns_ipc_map.py
"""

import os
import re
import io
import time
import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import PyPDF2
    from PyPDF2 import PdfReader as _PyPDF2Reader
    USE_PYPDF2 = True
except ImportError:
    USE_PYPDF2 = False

try:
    import fitz  # pymupdf — preferred, more reliable
    USE_FITZ = True
except ImportError:
    USE_FITZ = False

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR    = os.getenv("OUT_DIR",    "/dbfs/FileStore/nyaya_setu/output")
OUTPUT_CSV = os.getenv("BNS_CSV",    f"{OUT_DIR}/bns_ipc_mapping.csv")

# PDF download URLs
URL_IPC = "https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf"
URL_BNS = "https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf"

# Local fallback paths (upload PDFs to DBFS if download is blocked)
LOCAL_IPC = os.getenv("LOCAL_IPC", f"{OUT_DIR}/ipc.pdf")
LOCAL_BNS = os.getenv("LOCAL_BNS", f"{OUT_DIR}/bns.pdf")

os.makedirs(OUT_DIR, exist_ok=True)

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}


# ── PDF EXTRACTION ────────────────────────────────────────────────────────────
def _read_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from raw PDF bytes using pymupdf (preferred) or PyPDF2."""
    if USE_FITZ:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text
    if USE_PYPDF2:
        reader = _PyPDF2Reader(io.BytesIO(pdf_bytes))
        return "\n".join(
            (page.extract_text() or "") for page in reader.pages
        )
    raise RuntimeError("Neither pymupdf nor PyPDF2 is installed. Add one to requirements.txt")


def get_pdf_text(url: str, local_fallback: str) -> str:
    """Download PDF from URL; fall back to a local file if download fails."""
    # Try local cache first to avoid re-downloading
    if os.path.exists(local_fallback):
        print(f"📂  Using cached local file: {local_fallback}")
        with open(local_fallback, "rb") as f:
            return _read_pdf_bytes(f.read())

    print(f"🌐  Downloading: {url[:60]}…")
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        resp.raise_for_status()
        pdf_bytes = resp.content
        # Cache to DBFS for future runs
        with open(local_fallback, "wb") as f:
            f.write(pdf_bytes)
        print(f"✅  Downloaded ({len(pdf_bytes)/1e6:.1f} MB) — cached to {local_fallback}")
        return _read_pdf_bytes(pdf_bytes)
    except Exception as e:
        print(f"⚠️  Download failed ({e})")
        if os.path.exists(local_fallback):
            print(f"   Falling back to: {local_fallback}")
            with open(local_fallback, "rb") as f:
                return _read_pdf_bytes(f.read())
        raise FileNotFoundError(
            f"Cannot get PDF. Please upload manually to: {local_fallback}\n"
            f"Then re-run this script."
        ) from e


# ── SECTION PARSING ───────────────────────────────────────────────────────────
_SECTION_RE = re.compile(
    r'\n(\d+[A-Z]?)\.\s+([^\n]+)(.*?)(?=\n\d+[A-Z]?\.\s+|$)',
    re.DOTALL
)

def parse_law_text(text: str, code_name: str) -> pd.DataFrame:
    """Regex-parse raw PDF text into individual sections."""
    print(f"🔪  Parsing {code_name} …")
    matches  = _SECTION_RE.findall(text)
    sections = []
    for sec_num, title, content in matches:
        content = content.strip().replace('\n', ' ')
        if len(content) > 20:
            sections.append({
                "Section": sec_num.strip(),
                "Title":   title.strip(),
                "Text":    content,
            })
    print(f"✅  {len(sections):,} sections extracted from {code_name}")
    return pd.DataFrame(sections)


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("PHASE 1: PDF INGESTION & PARSING")
    print("=" * 55)

    ipc_text = get_pdf_text(URL_IPC, LOCAL_IPC)
    bns_text = get_pdf_text(URL_BNS, LOCAL_BNS)

    if not ipc_text.strip() or not bns_text.strip():
        raise ValueError("Empty PDF text. Check network / local files.")

    df_ipc = parse_law_text(ipc_text, "IPC 1860")
    df_bns = parse_law_text(bns_text, "BNS 2023")

    if df_ipc.empty or df_bns.empty:
        raise ValueError("Parsing produced empty dataframes. The PDFs may be scanned images "
                         "(not text-selectable). OCR is required for those.")

    # ── PHASE 2: TF-IDF MAPPING ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PHASE 2: TF-IDF SEMANTIC MAPPING (CPU)")
    print("=" * 55)

    print("🧠  Vectorising IPC + BNS (1–3 ngrams, stop-words removed)…")
    t0         = time.time()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))

    ipc_matrix = vectorizer.fit_transform(df_ipc["Text"])
    bns_matrix = vectorizer.transform(df_bns["Text"])

    print(f"   IPC matrix : {ipc_matrix.shape}")
    print(f"   BNS matrix : {bns_matrix.shape}")

    print("   Computing cosine similarity …")
    sim_matrix = cosine_similarity(bns_matrix, ipc_matrix)
    print(f"⏱️   Done in {time.time()-t0:.1f}s\n")

    # ── PHASE 3: ALIGNMENT ────────────────────────────────────────────────────
    print("=" * 55)
    print("PHASE 3: LINKING COUNTERPARTS")
    print("=" * 55)

    results             = []
    low_confidence_cnt  = 0

    for bns_idx in range(len(df_bns)):
        bns_row   = df_bns.iloc[bns_idx]
        best_ipc  = int(np.argmax(sim_matrix[bns_idx]))
        score     = float(sim_matrix[bns_idx][best_ipc])
        ipc_row   = df_ipc.iloc[best_ipc]

        status = "HIGH CONFIDENCE" if score >= 0.40 else "NEEDS REVIEW"
        if score < 0.40:
            low_confidence_cnt += 1

        results.append({
            "BNS_Section":       bns_row["Section"],
            "IPC_Section":       ipc_row["Section"],
            "Offence":           bns_row["Title"],
            "Confidence_Score":  round(score * 100, 1),
            "Status":            status,
            "BNS_Text":          bns_row["Text"],
        })

    df_mapping = pd.DataFrame(results)

    # Create the "Bridge Chunk" used by the LLM in Stage 4
    df_mapping["chunk_text"] = df_mapping.apply(
        lambda r: (
            f"BNS Section {r['BNS_Section']} - {r['Offence']} "
            f"(Formerly IPC Section {r['IPC_Section']}): {r['BNS_Text']}"
        ),
        axis=1,
    )

    df_mapping.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅  MAPPING COMPLETE!")
    print(f"   Total laws mapped          : {len(df_mapping):,}")
    print(f"   Low confidence (< 40%)     : {low_confidence_cnt:,}")
    print(f"   Saved → {OUTPUT_CSV}")

    print("\n   Sample mappings:")
    print(df_mapping[["BNS_Section", "IPC_Section", "Offence", "Confidence_Score"]]
          .head(10).to_string(index=False))

    print("\n✅  Stage 3 complete — Run 04_langchain_pipeline.py next (or launch app.py)")

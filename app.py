"""
app.py
======
NYAYA-SETU — Streamlit Frontend (Databricks Apps)

Databricks Apps runs this file directly.
All heavy artifacts (FAISS index, metadata, BNS CSV) must already exist
in DBFS before launching the app. Run the 4 pipeline scripts first.

Environment variables expected (set in app.yaml or Databricks App config):
  GROQ_API_KEY   — your Groq API key
  OUT_DIR        — DBFS path where pipeline outputs live
                   (default: /dbfs/FileStore/nyaya_setu/output)
"""

import os
import sys
import streamlit as st

# ── PAGE CONFIG (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title = "Nyaya-Setu ⚖️",
    page_icon  = "⚖️",
    layout     = "wide",
)

# ── INJECT scripts/ folder INTO PATH ─────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
SCRIPTS  = os.path.join(APP_DIR, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
OUT_DIR          = os.getenv("OUT_DIR",    "/dbfs/FileStore/nyaya_setu/output")
FAISS_INDEX_PATH = os.getenv("INDEX_PATH", f"{OUT_DIR}/faiss_precedents.index")
FAISS_META_PATH  = os.getenv("META_PATH",  f"{OUT_DIR}/faiss_metadata.parquet")
BNS_CSV_PATH     = os.getenv("BNS_CSV",    f"{OUT_DIR}/bns_ipc_mapping.csv")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=80)
    st.title("Nyaya-Setu ⚖️")
    st.caption("AI Legal Co-Counsel · BNS / IPC")
    st.divider()

    api_key_input = st.text_input(
        "Groq API Key",
        value       = GROQ_API_KEY,
        type        = "password",
        help        = "Get a free key at console.groq.com",
        placeholder = "gsk_…",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    st.divider()
    st.markdown("**Data paths**")
    st.code(f"FAISS : {FAISS_INDEX_PATH}\nMeta  : {FAISS_META_PATH}\nBNS   : {BNS_CSV_PATH}", language="")
    st.divider()
    st.markdown(
        "**How to set up**\n\n"
        "1. Run `01_chunk_pdfs.py`\n"
        "2. Run `02_build_faiss.py`\n"
        "3. Run `03_build_bns_ipc_map.py`\n"
        "4. Launch this app 🚀"
    )


# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("⚖️ Nyaya-Setu — Indian Legal AI Co-Counsel")
st.markdown(
    "Powered by **Llama 3.3 70B** (Groq) · **MiniLM** embeddings · "
    "**42k** Supreme Court precedents · **BNS ↔ IPC** mapping"
)
st.divider()


# ── ARTIFACT CHECK ────────────────────────────────────────────────────────────
def check_artifacts() -> tuple[bool, list]:
    missing = []
    for label, path in [
        ("FAISS Index",  FAISS_INDEX_PATH),
        ("Metadata",     FAISS_META_PATH),
        ("BNS-IPC Map",  BNS_CSV_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"**{label}** not found at `{path}`")
    return len(missing) == 0, missing


artifacts_ok, missing_list = check_artifacts()

if not artifacts_ok:
    st.error("⚠️  Required data files are missing. Run the pipeline scripts first.")
    for m in missing_list:
        st.markdown(f"- {m}")
    st.stop()


# ── LOAD PIPELINE (cached so it only runs once per session) ───────────────────
@st.cache_resource(show_spinner="Loading AI pipeline — first load ~2 min on CPU…")
def get_pipeline():
    if not os.environ.get("GROQ_API_KEY"):
        return None, "GROQ_API_KEY not set."
    try:
        # Import from scripts/
        from scripts.langchain_pipeline_module import (   # noqa: E402
            load_pipeline_artifacts, load_embedder, build_chain
        )
        faiss_index, df_meta, df_bns = load_pipeline_artifacts()
        embedder                      = load_embedder()
        chain                         = build_chain(faiss_index, df_meta, df_bns, embedder)
        return chain, None
    except Exception as e:
        return None, str(e)


# Lazy init: only load when user provides a key
if not os.environ.get("GROQ_API_KEY"):
    st.info("👈  Enter your **Groq API Key** in the sidebar to start.")
    st.stop()

chain, load_error = get_pipeline()
if load_error:
    st.error(f"Pipeline load failed: {load_error}")
    st.stop()


# ── EXAMPLE CASES ─────────────────────────────────────────────────────────────
EXAMPLES = {
    "Self-Defence (BNS 34 / IPC 99)": (
        "My client, a 34-year-old shopkeeper, was closing his shop at night when the complainant "
        "and two accomplices attacked him with iron rods. In the struggle, my client grabbed a nearby "
        "wooden plank and struck one of them on the head, causing grievous hurt. Police charged him "
        "under BNS Section 117 arguing disproportionate force. He claims pure self-defence."
    ),
    "Unlawful Assembly (BNS 189)": (
        "My clients are five farmers who gathered at the district collector's office to protest land "
        "acquisition. Police dispersed them and all five were arrested under BNS Section 189 (unlawful "
        "assembly) and BNS Section 193 (rioting). The protest was peaceful; no property was damaged. "
        "Build the strongest defence challenging the unlawful assembly charge."
    ),
    "Criminal Breach of Trust (BNS 316)": (
        "My client was a finance manager entrusted with client deposits worth ₹2.4 crore. The company "
        "collapsed and clients filed BNS Section 316 (criminal breach of trust) against him personally. "
        "He had no independent authority over funds, followed board directions, and was himself "
        "defrauded by the promoters. The FIR does not allege personal enrichment."
    ),
}


# ── INPUT AREA ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📋 Example Cases")
    example_choice = st.selectbox("Load example:", ["— custom —"] + list(EXAMPLES.keys()))

with col1:
    st.subheader("📝 Case Facts")
    default_text = EXAMPLES.get(example_choice, "") if example_choice != "— custom —" else ""
    case_facts   = st.text_area(
        "Describe the FIR, charges, and your client's position:",
        value  = default_text,
        height = 250,
        placeholder = (
            "e.g. My client is charged under BNS Section ___ for ___. "
            "The facts are: … The defence is: …"
        ),
    )

st.divider()

run_btn = st.button("⚖️  Generate Legal Defence", type="primary", use_container_width=True)


# ── RUN PIPELINE ──────────────────────────────────────────────────────────────
if run_btn:
    if not case_facts.strip():
        st.warning("Please enter case facts before running.")
    else:
        with st.spinner("🔍 Retrieving precedents & drafting defence… (~15–30s on first query)"):
            try:
                response = chain.invoke(case_facts.strip())
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        st.divider()
        st.subheader("⚖️ Nyaya-Setu Legal Analysis")

        # Render each section as an expander for readability
        sections = {
            "I. CHARGE & STATUTORY FRAMEWORK":           "📜",
            "II. GOVERNING PRINCIPLES FROM PRECEDENT":   "📚",
            "III. DEFENCE STRATEGY":                     "🛡️",
            "IV. ANTICIPATED PROSECUTION ARGUMENTS":     "⚔️",
            "V. TACTICAL SUMMARY":                       "🎯",
        }

        # Try structured display; fall back to plain markdown
        found_any = False
        remaining = response

        import re
        for heading, icon in sections.items():
            pattern = re.compile(
                rf'\*\*{re.escape(heading)}[^*]*\*\*(.*?)(?=\*\*[IVX]+\.|$)',
                re.DOTALL | re.IGNORECASE
            )
            m = pattern.search(response)
            if m:
                found_any = True
                content   = m.group(1).strip()
                with st.expander(f"{icon} {heading}", expanded=True):
                    st.markdown(content)

        if not found_any:
            # Fallback: raw markdown
            st.markdown(response)

        # Download button
        st.divider()
        st.download_button(
            label    = "📥  Download Full Analysis (.txt)",
            data     = response,
            file_name= "nyaya_setu_analysis.txt",
            mime     = "text/plain",
        )


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️  Nyaya-Setu is an AI research tool for legal professionals. "
    "It does not constitute legal advice. Always verify citations independently. "
    "Low-confidence BNS↔IPC mappings (< 60%) must be cross-checked with official gazette."
)

# Nyaya-Setu ⚖️
**Indian Legal AI Co-Counsel · BNS / IPC · CPU-only · Databricks Apps**

> Powered by Llama 3.3 70B (Groq) + MiniLM embeddings + 42k Supreme Court precedents

---

## Repository Layout

```
nyaya_setu/
├── app.py                        # Streamlit frontend (Databricks App entry point)
├── app.yaml                      # Databricks Apps deployment manifest
├── requirements.txt              # All Python dependencies (CPU-only)
├── README.md
└── scripts/
    ├── __init__.py
    ├── 01_chunk_pdfs.py          # Stage 1: PDF → chunks_master.parquet
    ├── 02_build_faiss.py         # Stage 2: chunks → FAISS index + metadata
    ├── 03_build_bns_ipc_map.py   # Stage 3: IPC + BNS PDFs → bns_ipc_mapping.csv
    ├── 04_langchain_pipeline.py  # Stage 4: LangChain + Groq chain (standalone test)
    └── langchain_pipeline.py     # Same as 04_, imported by app.py
```

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Databricks workspace | Free Edition works (CPU-only) |
| Groq API Key | Free at [console.groq.com](https://console.groq.com/keys) |
| Supreme Court PDFs | Upload to DBFS (see Step 2) |
| GitHub account | To link repo with Databricks |

---

## Step-by-Step Setup

### Step 1 — Create & Push the GitHub Repo

```bash
git init nyaya_setu
cd nyaya_setu
# Copy all files from this repo
git add .
git commit -m "Initial Nyaya-Setu commit"
git remote add origin https://github.com/YOUR_USERNAME/nyaya-setu.git
git push -u origin main
```

### Step 2 — Link GitHub Repo to Databricks

1. In Databricks: **Workspace → Git Folders → Add Git Folder**
2. Paste your GitHub repo URL
3. Set branch: `main`
4. Click **Create Git Folder**

Your files now appear under `/Workspace/Users/you@email.com/nyaya-setu/`

### Step 3 — Upload PDFs to DBFS

Option A — Databricks UI:
- Go to **Data → DBFS → FileStore**
- Create folder: `nyaya_setu/pdfs/`
- Upload your Supreme Court PDFs maintaining `year=YYYY/english/english/` structure

Option B — Databricks CLI:
```bash
databricks fs mkdirs dbfs:/FileStore/nyaya_setu/pdfs/year=2015/english/english/
databricks fs cp local_pdfs/ dbfs:/FileStore/nyaya_setu/pdfs/ --recursive
```

### Step 4 — Install Dependencies in a Cluster

1. Create a **CPU cluster** (Standard_DS3_v2 or similar, no GPU needed)
2. Go to **Cluster → Libraries → Install New → PyPI**
3. Install from `requirements.txt` — or paste requirements one by one
   
   **Quick install in a notebook cell:**
   ```python
   %pip install pymupdf PyPDF2 pandas pyarrow numpy sentence-transformers \
                faiss-cpu scikit-learn langchain langchain-core langchain-groq \
                groq requests streamlit tqdm
   ```

### Step 5 — Run the 4 Pipeline Scripts (in order)

Open a Databricks notebook and run each script in a cell:

```python
# Cell 1 — Chunk PDFs (adjust PDF_GLOB if needed)
import os
os.environ["PDF_GLOB"] = "/dbfs/FileStore/nyaya_setu/pdfs/year=*/english/**/*.pdf"
os.environ["OUT_DIR"]  = "/dbfs/FileStore/nyaya_setu/output"
%run /Workspace/Users/you@email.com/nyaya-setu/scripts/01_chunk_pdfs.py
```

```python
# Cell 2 — Build FAISS index (~8-15 min on CPU)
%run /Workspace/Users/you@email.com/nyaya-setu/scripts/02_build_faiss.py
```

```python
# Cell 3 — Build BNS-IPC map (downloads PDFs from govt URLs)
%run /Workspace/Users/you@email.com/nyaya-setu/scripts/03_build_bns_ipc_map.py
```

```python
# Cell 4 — Test the LangChain pipeline
import os
os.environ["GROQ_API_KEY"] = "gsk_YOUR_KEY_HERE"  # remove after testing
%run /Workspace/Users/you@email.com/nyaya-setu/scripts/04_langchain_pipeline.py
```

Expected output file sizes:
- `chunks_master.parquet`   — ~150–400 MB
- `faiss_precedents.index`  — ~60–200 MB  
- `faiss_metadata.parquet`  — ~200–500 MB
- `bns_ipc_mapping.csv`     — ~1–5 MB

### Step 6 — Deploy as Databricks App

1. Go to **Apps** in the left sidebar → **Create App**
2. Select **Custom App** (not a template)
3. Set:
   - **Name**: `nyaya-setu`
   - **Source**: Git folder → your linked repo
   - **Entry file**: `app.py`
4. Add **Environment Variables**:
   - `OUT_DIR` = `/dbfs/FileStore/nyaya_setu/output`
   - `INDEX_PATH` = `/dbfs/FileStore/nyaya_setu/output/faiss_precedents.index`
   - `META_PATH` = `/dbfs/FileStore/nyaya_setu/output/faiss_metadata.parquet`
   - `BNS_CSV` = `/dbfs/FileStore/nyaya_setu/output/bns_ipc_mapping.csv`
5. Add **Secret**:
   - Key: `GROQ_API_KEY`, Value: `gsk_YOUR_KEY`
6. Click **Deploy** 🚀

The app will be available at `https://YOUR_WORKSPACE.azuredatabricks.net/apps/nyaya-setu`

### Step 7 — Store Groq Key as Databricks Secret (Production)

```bash
# Using Databricks CLI
databricks secrets create-scope nyaya-setu
databricks secrets put-secret nyaya-setu groq-api-key --string-value "gsk_YOUR_KEY"
```

Then in `app.yaml`, the `secretRef` block references `groq-api-key` automatically.

---

## Resource Constraints (Databricks Free Edition)

| Resource | Constraint | How we handle it |
|----------|-----------|-----------------|
| CPU only, no GPU | Can't run large local models | Use Groq API (cloud LLM, free tier) |
| ~15 GB RAM | Can't load huge datasets | Parquet streaming, small batch sizes |
| No `venv` module | Can't create virtual envs | Install directly via `%pip` |
| Limited DBFS | Large index files | Clean checkpoints after run |
| No persistent processes | App restarts are cold starts | `@st.cache_resource` caches pipeline |

**Why Groq instead of a local model?**  
A quantised 7B model on CPU takes ~4 minutes per response. Groq serves  
Llama 3.3 70B at ~280 tokens/sec — faster and far smarter.

---

## Troubleshooting

**"FAISS index not found"**  
→ Run Steps 4–5 first. Check DBFS path with `dbutils.fs.ls("dbfs:/FileStore/nyaya_setu/output/")`.

**"No PDFs found" in Stage 1**  
→ Check `PDF_GLOB`. DBFS paths in Python use `/dbfs/` prefix, not `dbfs:/`.

**Stage 3 PDF download fails**  
→ Government URLs sometimes block automated requests. Upload PDFs manually to  
`/dbfs/FileStore/nyaya_setu/output/ipc.pdf` and `/dbfs/FileStore/nyaya_setu/output/bns.pdf`.  
The script will auto-detect and use local files.

**OOM during Stage 2 embedding**  
→ Reduce `BATCH_SIZE`: `os.environ["BATCH_SIZE"] = "32"` before running Stage 2.

**App loads but pipeline fails**  
→ Make sure `GROQ_API_KEY` is set in App environment variables, not just in the notebook.

---

## Architecture

```
PDFs (DBFS)
    │
    ▼
01_chunk_pdfs.py ──→ chunks_master.parquet
    │
    ▼
02_build_faiss.py ──→ faiss_precedents.index + faiss_metadata.parquet
    │
03_build_bns_ipc_map.py ──→ bns_ipc_mapping.csv
    │
    ▼
app.py (Streamlit)
    │
    ├── MiniLM embedder (CPU) ──→ query vector
    ├── FAISS search ──→ top-3 SC precedents  
    ├── BNS cosine search ──→ governing statute
    └── Groq LLM (cloud) ──→ legal defence draft
```

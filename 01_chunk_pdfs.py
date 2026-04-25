"""
01_chunk_pdfs.py
================
NYAYA-SETU — Stage 1: PDF Chunking Pipeline
Reads Supreme Court PDFs → cleans → chunks → saves master parquet to DBFS.

Run in Databricks:  %run ./scripts/01_chunk_pdfs.py
Or standalone:      python scripts/01_chunk_pdfs.py
"""

import os
import re
import gc
import glob
import time
import fitz           # pymupdf
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
# On Databricks, put your PDFs in a Unity Catalog volume or DBFS path
PDF_GLOB   = os.getenv("PDF_GLOB",   "/dbfs/FileStore/nyaya_setu/pdfs/year=*/english/**/*.pdf")
OUT_DIR    = os.getenv("OUT_DIR",    "/dbfs/FileStore/nyaya_setu/output")
MASTER_OUT = f"{OUT_DIR}/chunks_master.parquet"
CKPT_DIR   = f"{OUT_DIR}/checkpoints"

SAVE_EVERY    = 100
CHUNK_WORDS   = 400
OVERLAP_WORDS = 50
MIN_WORDS     = 40

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── CLEANING REGEXES ─────────────────────────────────────────────────────────
RE_MARGIN_LINE        = re.compile(r'(?m)^\s*[A-H]\s*$')
RE_MARGIN_INLINE      = re.compile(r'(?<!\w)([A-H])(?!\w)')
RE_PAGE_HDR           = re.compile(r'\[\d{4}\]\s+\d+\s+S[\s.]*(?:C[\s.]*)?R[\s.]*[^\n]*')
RE_SCR_LABEL          = re.compile(r'(?m)^SUPREME COURT REPORTS[^\n]*$')
RE_PAGE_NUM           = re.compile(r'(?m)^\s*\d{2,4}\s*$')
RE_RUNNING_CASE_HDR   = re.compile(
    r'^[ \t]*[A-Z][A-Z\s&\.v,/\[\]()]+\s+\d{3,4}[ \t]*$'
    r'|^[ \t]*&[A-Z\s\.]+\[[A-Z\s\.,]+J\.?\][ \t]*$',
    re.MULTILINE
)
RE_FOOTNOTE_INLINE    = re.compile(r'\d+\s*\(\d{4}\)\s+\d+\s+S[CJ]{1,2}C\s+\d+')
RE_PARA_REF           = re.compile(
    r'\[Paras?\s*[\d][^\]\)]{0,30}[\)\]]'
    r'|\[\d{3,4}[-–][A-H](?:[-–][A-H])?'
    r'(?:;\s*\d{3,4}[-–][A-H](?:[-–][A-H])?)*[\)\]]'
    r'|\[\d{3,4}[-–][A-H][-–][A-H][\)\]]',
    re.IGNORECASE
)
RE_XXX                = re.compile(r'x{3,}', re.IGNORECASE)

# ── ZONE DETECTION ───────────────────────────────────────────────────────────
RE_JUDGES       = re.compile(r'\[\s*([A-Z][A-Z\s\.,]+?(?:J\.?\s*J\.?|C\.?\s*J\.?|CJI\.?))\s*\]')
RE_CITATION     = re.compile(r'\[(\d{4})\]\s+(\d+)\s+S[\s.]*(?:C[\s.]*)?R[\s.]*\s*(\d+)', re.IGNORECASE)
RE_DATE         = re.compile(
    r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|'
    r'SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},?\s+\d{4}',
    re.IGNORECASE
)
RE_APPEAL       = re.compile(
    r'(Civil|Criminal|Writ)\s+(?:Appeal|Petition)\s+'
    r'(?:D\.?No\.?|No\.?|Nos?\\.?)\s*[\d\s,]+of\s+\d{4}',
    re.IGNORECASE
)
RE_COURT_ACTION = re.compile(
    r'(?:Allowing|Dismissing|Disposing\s+of|Partly\s+allowing|Allowing\s+in\s+part)\s+'
    r'(?:the\s+)?(?:appeal|petition|application|appeals)'
    r'[s,\s]*(?:the\s+)?(?:[Cc]ourt|Court)',
    re.IGNORECASE
)
RE_BODY_START   = re.compile(
    r'The\s+Judgment\s+of\s+the\s+Court\s+was\s+delivered\s+by'
    r'|J\s*U\s*D\s*G\s*M\s*E\s*N\s*T\s*\n'
    r'|\bORDER\b\s*\n',
    re.IGNORECASE
)
RE_CASELAW_REF  = re.compile(
    r'Case\s*(?:Law\s*)?Reference|CIVIL\s+APPELLATE\s+JURISDICTION'
    r'|CRIMINAL\s+APPELLATE\s+JURISDICTION|ORIGINAL\s+JURISDICTION',
    re.IGNORECASE
)
RE_COUNSEL      = re.compile(
    r'(?:for\s+(?:the\s+)?(?:Appellant|Respondent|Petitioner)|'
    r'(?:Sr\.|Senior)\s+Adv|Advocate|ASG\b|Additional\s+Solicitor)',
    re.IGNORECASE
)

# ── HELD SPLITTING ───────────────────────────────────────────────────────────
RE_DECIMAL_LABEL       = re.compile(r'(?m)^\s*(\d+\.\d+)\s+')
RE_PLAIN_LABEL         = re.compile(r'(?m)^\s*(\d{1,2})\.\s+(?=[A-Z(])')
RE_ROMAN_LABEL         = re.compile(r'(?m)^\s*\(([ivxlIVXL]+)\)\s+')
RE_HELD_KEYWORD_STRICT = re.compile(r'\bHELD\s*:\s*')
RE_HELD_KEYWORD_LOOSE  = re.compile(r'\bHELD\s*:\s*', re.IGNORECASE)

MAX_GARBAGE_RATIO = 0.10


# ── HELPERS ──────────────────────────────────────────────────────────────────
def garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    junk = sum(1 for c in text if ord(c) > 127 or c in r'·•□■▪▫~`^©®\|{}')
    return junk / max(len(text), 1)


def is_garbage_chunk(text: str) -> bool:
    if garbage_ratio(text) > MAX_GARBAGE_RATIO:
        return True
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return True
    counsel_lines = sum(1 for l in lines if RE_COUNSEL.search(l))
    if len(lines) > 3 and counsel_lines / len(lines) > 0.4:
        return True
    word_count   = len(text.split())
    symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if word_count > 0 and symbol_count / max(word_count, 1) > 4.0:
        return True
    return False


def extract_pdf_text(path: str):
    try:
        doc   = fitz.open(path)
        pages = [pg.get_text("text") for pg in doc]
        doc.close()
        return "\n".join(pages)
    except Exception:
        return None


def clean_text(raw: str) -> str:
    t = RE_MARGIN_LINE.sub('\n', raw)
    t = RE_RUNNING_CASE_HDR.sub(' ', t)
    t = RE_PAGE_HDR.sub(' ', t)
    t = RE_SCR_LABEL.sub(' ', t)
    t = RE_PAGE_NUM.sub('\n', t)
    t = RE_FOOTNOTE_INLINE.sub(' ', t)
    t = RE_PARA_REF.sub(' ', t)
    t = RE_XXX.sub(' ', t)
    t = RE_MARGIN_INLINE.sub(' ', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def split_by_words(text: str, max_w: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list:
    words = text.split()
    if len(words) <= max_w:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_w, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_w - overlap
    return chunks


def _clean_act_name(name: str) -> str:
    return re.sub(r'^[\s\nA-H]+', '', name).strip()


def parse_metadata(raw_text: str, filepath: str) -> dict:
    header = raw_text[:4000]
    meta = {
        "source_file":   Path(filepath).name,
        "year":          None,
        "citation":      None,
        "case_name":     None,
        "judges":        None,
        "date":          None,
        "appeal_no":     None,
        "acts_involved": [],
        "legal_domain":  None,
        "outcome":       None,
    }

    ym = re.search(r'year=(\d{4})', filepath)
    if ym:
        meta["year"] = int(ym.group(1))

    cit = RE_CITATION.search(header)
    if cit:
        meta["citation"] = f"[{cit.group(1)}] {cit.group(2)} S.C.R. {cit.group(3)}"
        if not meta["year"]:
            meta["year"] = int(cit.group(1))

    dt = RE_DATE.search(header)
    if dt:
        meta["date"] = dt.group(0).strip()

    judges_found = RE_JUDGES.findall(header)
    if judges_found:
        meta["judges"] = max(judges_found, key=len).strip()

    ap = RE_APPEAL.search(header[:3000])
    if ap:
        meta["appeal_no"] = ap.group(0).strip()

    clean_hdr  = RE_PAGE_HDR.sub('', header)
    clean_hdr  = RE_SCR_LABEL.sub('', clean_hdr)
    lines      = [l.strip() for l in clean_hdr.split('\n') if l.strip()]
    name_lines = []
    for i, line in enumerate(lines[:30]):
        if re.search(r'\bv(?:s)?\.?\s+', line, re.IGNORECASE) and 5 < len(line) < 200:
            name_lines = lines[max(0, i - 2): i + 2]
            break

    candidate = " ".join(name_lines).strip()[:220] if name_lines else (lines[0][:220] if lines else "")
    if candidate and (len(candidate.strip()) <= 2 or re.fullmatch(r'[A-H\s]*', candidate.strip())):
        candidate = ""
    meta["case_name"] = candidate or None

    raw_acts = re.findall(
        r'[A-Z][A-Za-z\s]+(?:Act|Code|Rules|Ordinance|Regulation),?\s*\d{4}',
        header[:3000]
    )
    acts = [_clean_act_name(a) for a in raw_acts if len(_clean_act_name(a)) > 6]
    meta["acts_involved"] = list(dict.fromkeys(acts))[:8]

    sample     = header[:3000].upper()
    domain_map = {
        "Criminal":       ["IPC", "CRPC", "NDPS", "ARMS ACT", "NARCOTIC", "CRIMINAL", "PENAL CODE"],
        "Constitutional": ["CONSTITUTION", "ARTICLE ", "FUNDAMENTAL RIGHTS", "WRIT", "HABEAS CORPUS"],
        "Service":        ["SERVICE LAW", "ARMY", "DISCHARGE", "SERVICE RULES", "DISCIPLINARY"],
        "Civil":          ["MOTOR VEHICLES", "CIVIL APPEAL", "COMPENSATION", "CONTRACT", "TORT"],
        "Revenue":        ["LAND ACQUISITION", "REVENUE", "MINES", "MINERAL"],
        "Family":         ["HINDU MARRIAGE", "DIVORCE", "MAINTENANCE", "CUSTODY", "SUCCESSION"],
        "Industrial":     ["INDUSTRIAL DISPUTES", "WORKMEN", "EMPLOYEES", "LABOUR"],
        "Environmental":  ["ENVIRONMENT", "FOREST", "WILDLIFE", "POLLUTION"],
    }
    for domain, kws in domain_map.items():
        if any(k in sample for k in kws):
            meta["legal_domain"] = domain
            break
    meta["legal_domain"] = meta["legal_domain"] or "General"

    outcome_m = RE_COURT_ACTION.search(raw_text[:5000])
    if outcome_m:
        meta["outcome"] = outcome_m.group(0).strip()[:80]

    return meta


def extract_issue_summary(text: str) -> str:
    judge_m  = RE_JUDGES.search(text[:5000])
    start    = judge_m.end() if judge_m else 0
    action_m = RE_COURT_ACTION.search(text[start: start + 3000])
    end      = start + action_m.start() if action_m else start + 2500
    snippet  = text[start:end]

    issue_lines = []
    for line in snippet.split('\n'):
        line = line.strip()
        if len(line) < 20:
            continue
        if ('–' in line or ' - ' in line
                or bool(re.search(r'\b(?:Held|Act|Code|Section|Rule)\b', line, re.IGNORECASE))
                or ':' in line):
            issue_lines.append(line)
    return "\n".join(issue_lines[:35])


def isolate_held_block(text: str):
    held_m = RE_HELD_KEYWORD_STRICT.search(text) or RE_HELD_KEYWORD_LOOSE.search(text)
    if not held_m:
        for pat in [RE_DECIMAL_LABEL, RE_PLAIN_LABEL]:
            m = pat.search(text[:10000])
            if m:
                held_m = m
                break
    if not held_m:
        return "", "absent"

    held_text = text[held_m.start():]
    body_m    = RE_BODY_START.search(held_text)
    if body_m:
        held_text = held_text[:body_m.start()]
    caselaw_m = RE_CASELAW_REF.search(held_text)
    if caselaw_m:
        held_text = held_text[:caselaw_m.start()]

    if len(held_text.split()) < MIN_WORDS:
        return "", "absent"

    decimal_count = len(RE_DECIMAL_LABEL.findall(held_text))
    plain_count   = len(RE_PLAIN_LABEL.findall(held_text))
    roman_count   = len(RE_ROMAN_LABEL.findall(held_text))

    if decimal_count >= 2:
        return held_text, "decimal"
    elif plain_count >= 2:
        return held_text, "plain"
    elif roman_count >= 2:
        return held_text, "roman"
    return held_text, "single"


def split_held_block(held_text: str, style: str) -> list:
    patterns = {
        "decimal": (r'(?m)(?=^\s*\d+\.\d+\s+)', RE_DECIMAL_LABEL),
        "plain":   (r'(?m)(?=^\s*\d{1,2}\.\s+[A-Z(])', RE_PLAIN_LABEL),
        "roman":   (r'(?m)(?=^\s*\([ivxlIVXL]+\)\s+)', RE_ROMAN_LABEL),
    }
    if style not in patterns:
        para = re.sub(r'^HELD\s*:\s*', '', held_text, flags=re.IGNORECASE).strip()
        return [{"label": "single", "para_text": para}]

    split_pat, label_re = patterns[style]
    segments = re.split(split_pat, held_text)
    results  = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        lm = label_re.match(seg)
        if lm:
            label     = lm.group(1)
            para_body = seg[lm.end():].strip()
        else:
            label     = "intro"
            para_body = re.sub(r'^HELD\s*:\s*', '', seg, flags=re.IGNORECASE).strip()
        if len(para_body.split()) >= MIN_WORDS and not is_garbage_chunk(para_body):
            results.append({"label": label, "para_text": para_body})
    return results


def process_one_pdf(pdf_path: str) -> list:
    raw = extract_pdf_text(pdf_path)
    if not raw:
        return []

    meta = parse_metadata(raw, pdf_path)
    text = clean_text(raw)
    del raw; gc.collect()

    if len(text.split()) < MIN_WORDS:
        del text; gc.collect()
        return []

    issue_text = extract_issue_summary(text)
    held_text, style = isolate_held_block(text)
    held_segs = split_held_block(held_text, style) if held_text else []

    base_id = Path(pdf_path).stem
    chunks  = []

    if len(issue_text.split()) >= MIN_WORDS and not is_garbage_chunk(issue_text):
        chunks.append({
            **meta,
            "chunk_id":   f"{base_id}_issues",
            "chunk_type": "issue_summary",
            "held_para":  None,
            "chunk_text": issue_text,
            "word_count": len(issue_text.split()),
            "embed_text": " ".join(issue_text.split()[:180]),
        })

    for seg in held_segs:
        sub_chunks = split_by_words(seg["para_text"], CHUNK_WORDS, OVERLAP_WORDS)
        for idx, sub in enumerate(sub_chunks):
            wc = len(sub.split())
            if wc < MIN_WORDS or is_garbage_chunk(sub):
                continue
            suffix = f"_p{idx}" if len(sub_chunks) > 1 else ""
            chunks.append({
                **meta,
                "chunk_id":   f"{base_id}_held_{seg['label']}{suffix}",
                "chunk_type": "held_principle",
                "held_para":  seg["label"],
                "chunk_text": sub,
                "word_count": wc,
                "embed_text": " ".join(sub.split()[:180]),
            })

    del text; gc.collect()
    return chunks


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("🔍 SCANNING FOR PDFs...")
    print("=" * 65)

    all_pdfs = sorted(glob.glob(PDF_GLOB, recursive=True))
    print(f"   Found: {len(all_pdfs):,} PDFs\n")

    if not all_pdfs:
        print("⚠️  No PDFs found. Check PDF_GLOB env var or path.")
        raise SystemExit(1)

    all_chunks, failed = [], []
    ckpt_idx = 0
    t_start  = time.time()

    pbar = tqdm(all_pdfs, unit="pdf", dynamic_ncols=True)
    for i, pdf_path in enumerate(pbar):
        pbar.set_postfix({"chunks": len(all_chunks), "failed": len(failed)}, refresh=False)
        try:
            all_chunks.extend(process_one_pdf(pdf_path))
        except Exception as e:
            failed.append({"file": Path(pdf_path).name, "error": str(e)})

        if (i + 1) % SAVE_EVERY == 0 and all_chunks:
            ckpt_path = f"{CKPT_DIR}/ckpt_{ckpt_idx:04d}.parquet"
            pd.DataFrame(all_chunks).to_parquet(ckpt_path, index=False)
            tqdm.write(f"   💾 Checkpoint {ckpt_idx} → {ckpt_path}  ({len(all_chunks):,} chunks)")
            ckpt_idx += 1

    elapsed = time.time() - t_start
    print(f"\n⏱️  Done in {elapsed:.1f}s  ({len(all_pdfs)/max(elapsed,1):.1f} PDFs/sec)\n")

    if all_chunks:
        df_master = pd.DataFrame(all_chunks)
        if "acts_involved" in df_master.columns:
            df_master["acts_involved"] = df_master["acts_involved"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else str(x)
            )
        df_master.to_parquet(MASTER_OUT, index=False)
        print(f"💾 Master parquet saved → {MASTER_OUT}")
        print(f"   Size: {os.path.getsize(MASTER_OUT)/1e6:.1f} MB")
        print(f"   Total chunks: {len(df_master):,}")
    else:
        print("⚠️  No chunks extracted.")

    if failed:
        print(f"\n⚠️  Failed PDFs (first 10):")
        for f in failed[:10]:
            print(f"   {f['file']} → {f['error']}")

    import shutil
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    print("\n✅  Stage 1 complete — Run 02_build_faiss.py next")

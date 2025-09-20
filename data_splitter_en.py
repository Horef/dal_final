# data_splitter_en.py
# English-friendly, 2-column-aware splitter for MiniRAG
# Robust against text skipping, duplicate chunks, and over-eager table detection.

from __future__ import annotations
import re, json, os, shutil
from typing import List, Dict, Optional, Callable

# ---------- Heuristics ----------
_PAGE_RE = re.compile(r"^===\s*Page\s+(\d+)\s*===\s*$")

# IMPORTANT: Make table detection conservative.
# Old pattern over-fired on ordinary lines that just had multiple spaces.
# New: only tabs or literal pipes (|) count as table-ish.
_TABLEISH_RE = re.compile(r"(?:\t.+\t)|(?:.+\|.+\|.+)")

_LISTISH_RE = re.compile(
    r"""^\s*(?:[-\u2022\*\u25CF]
          |\(\s*[ivxlcdmIVXLCDM]+\s*\)
          |[ivxlcdmIVXLCDM]+\s*[.)]\s+
          |\(\s*[a-zA-Z]\s*\)
          |[a-zA-Z]\s*[.)]\s+
          |\d+\s*[.)]\s+
         )""",
    re.VERBOSE,
)

# Split on sentence terminators, but we still do paragraph-first logic.
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?…])\s+")

_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+){0,5}\s+\S")
_APPENDIX_HEADING_RE = re.compile(r"^\s*Appendix\s+[A-Z][\)\.: -]\s*\S", re.IGNORECASE)
_ROMAN_HEADING_RE = re.compile(r"^\s*[IVXLCDM]+\s+[A-Z][^\n]{1,80}$")
_ALL_CAPS_LINE_RE = re.compile(r"^[A-Z][A-Z0-9 &/\-,'()]{2,80}$")
_COMMON_TERMS = (
    "Undergraduate Studies", "Introduction", "Regulations", "Procedures", "Curriculum",
    "Semesters", "Admission", "Eligibility", "Credits", "Grades", "Exams", "Appeals",
    "Degree", "Graduation", "Distinction", "Specializations", "Tracks",
    "Prerequisites", "Syllabi", "Requirements", "Assessment", "Evaluation",
    "Admission Requirements", "Admission Tracks", "Study Tracks", "Degree Tracks",
    "Course of Study",
)
_CONTACT_RE = re.compile(r"(@|mailto:|tel:|\bhttps?://|\bwww\.)", re.IGNORECASE)

def _normalize_spaces_keep_tabs(s: str) -> str:
    # Keep tabs for table detection; normalize other whitespace.
    return re.sub(r"[^\S\t]+", " ", s).strip()

def _looks_all_caps(s: str) -> bool:
    return bool(_ALL_CAPS_LINE_RE.match(s)) and len(s) <= 80

def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if _NUMBERED_HEADING_RE.match(s): return True
    if _APPENDIX_HEADING_RE.match(s): return True
    if _ROMAN_HEADING_RE.match(s): return True
    if _looks_all_caps(s): return True
    score = 0
    if s.endswith(":"): score += 2
    punct_count = sum(ch in ",.;?!:|/" for ch in s)
    if punct_count <= 1: score += 1
    if any(t.lower() in s.lower() for t in _COMMON_TERMS): score += 2
    if _LISTISH_RE.match(s): return False
    return score >= 3

def _split_paragraph(s: str, soft: int, hard: int) -> List[str]:
    """Split long paragraphs into sub-paras, trying to respect sentences first."""
    s = s.strip()
    if len(s) <= hard:
        return [s]
    parts = _SENT_SPLIT_RE.split(s)
    out: List[str] = []
    buf: List[str] = []
    blen = 0

    def flush_buf():
        nonlocal out, buf, blen
        if buf:
            out.append(" ".join(buf).strip())
            buf, blen = [], 0

    for seg in (p.strip() for p in parts if p.strip()):
        if len(seg) > hard:
            # Very long "sentence": fall back to word packing.
            words, wbuf, wlen = seg.split(), [], 0
            for w in words:
                add = (1 if wlen else 0) + len(w)
                if wlen + add > soft:
                    if wbuf: out.append(" ".join(wbuf))
                    wbuf, wlen = [w], len(w)
                else:
                    wbuf.append(w); wlen += add
            if wbuf: out.append(" ".join(wbuf))
            continue
        add = (1 if blen else 0) + len(seg)
        if blen + add > soft:
            flush_buf()
        buf.append(seg); blen += add
    flush_buf()

    # Merge tiny trailing fragments back if it fits.
    merged: List[str] = []
    for p in out:
        if merged and len(p) < 80 and len(merged[-1]) + 1 + len(p) <= hard:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged

class Block:
    __slots__ = ("kind", "text", "page_start", "page_end", "heading")
    def __init__(self, kind: str, text: str, page_start: int, page_end: int, heading: Optional[str]):
        self.kind, self.text, self.page_start, self.page_end, self.heading = kind, text, page_start, page_end, heading

def _iter_blocks(raw_text: str) -> List[Block]:
    """
    Parse lines into blocks (heading / list / table / para).
    We do NOT drop headers/footers here (your extractor can, if asked).
    Also fix common PDF hyphenation across line breaks.
    """
    def _dehyphenate(prev: str, cur: str) -> Optional[str]:
        # join "Techn-" + "ion" -> "Technion"; prefer if next starts lowercase or mid-word
        if prev.endswith("-") and (cur and cur[0].islower()):
            return prev[:-1] + cur
        return None

    lines = raw_text.splitlines()
    cur_page = 1
    blocks: List[Block] = []
    buf: List[str] = []
    buf_kind: Optional[str] = None
    buf_page_start = 1
    section_heading: Optional[str] = None

    def flush():
        nonlocal buf, buf_kind, buf_page_start
        if not buf:
            return
        # Merge lines inside a block, but avoid losing structure.
        joined = []
        for ln in buf:
            if joined:
                fused = _dehyphenate(joined[-1], ln)
                if fused is not None:
                    joined[-1] = fused
                    continue
                # Inline-join when the previous line doesn't look like a hard break.
                if not joined[-1].endswith((".", "?", "!", ":", ";")) and not _LISTISH_RE.match(ln):
                    joined[-1] = joined[-1].rstrip() + " " + ln.lstrip()
                    continue
            joined.append(ln)
        txt = "\n".join(joined).strip("\n")
        if txt.strip():
            blocks.append(Block(buf_kind or "para", txt, buf_page_start, cur_page, section_heading))
        buf, buf_kind = [], None

    for ln in lines:
        if _PAGE_RE.match(ln):
            flush()
            try:
                cur_page = int(_PAGE_RE.match(ln).group(1))
            except:
                pass
            continue

        line = _normalize_spaces_keep_tabs(ln)
        if not line:
            flush()
            continue

        # Contacts/URLs keep as standalone paragraphs
        if _CONTACT_RE.search(line):
            flush()
            blocks.append(Block("para", line, cur_page, cur_page, section_heading))
            continue

        if _is_heading(line):
            flush()
            blocks.append(Block("heading", line, cur_page, cur_page, None))
            section_heading = line.rstrip(":")
            continue

        is_table = bool(_TABLEISH_RE.search(line))
        is_list  = bool(_LISTISH_RE.match(line))
        kind = "table" if is_table else "list" if is_list else "para"
        if buf_kind != kind:
            flush()
            buf_kind, buf_page_start = kind, cur_page
        buf.append(line)
    flush()

    # Propagate last seen heading down to blocks
    last_heading: Optional[str] = None
    for b in blocks:
        if b.kind == "heading":
            last_heading, b.heading = b.text.rstrip(":"), None
        else:
            b.heading = last_heading
    return blocks

# ---------- Chunking ----------
def _chunkize_blocks(
    blocks: List[Block],
    target_chars=1100,
    overlap_chars=150,
    min_chars=200,
    max_chars=2200,
    keep_table_as_whole=True
) -> List[Dict]:

    # Keep overlap modest: ≤ 25% of target (prevents heavy duplication).
    max_overlap = max(0, int(0.25 * max(1, target_chars)))
    if overlap_chars > max_overlap:
        overlap_chars = max_overlap

    chunks: List[Dict] = []
    buf: List[str] = []
    cur_heading: Optional[str] = None
    cur_page_start: Optional[int] = None
    cur_page_end: Optional[int] = None
    idx = 1

    # Overlap is a pending tail attached exactly once when fresh content arrives.
    pending_tail: str = ""

    def cur_len() -> int:
        return sum(len(s) + 1 for s in buf)

    def _word_safe_tail(text: str, desired: int) -> str:
        """
        Choose an overlap tail that starts on a WORD BOUNDARY by scanning LEFT,
        not right. The previous version scanned forward and could drop short
        words like 'as' entirely ("…as well" -> "… well").
        """
        if desired <= 0 or len(text) <= desired:
            return text
        start = max(0, len(text) - desired)
        # Move left to the nearest whitespace boundary to avoid mid-word cut.
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        tail = text[start:].lstrip()
        # Trim a partial word at the very end, if any
        end = len(tail) - 1
        while end >= 0 and not tail[end].isspace():
            end -= 1
        if 0 <= end < len(tail) - 1:
            tail = tail[:end + 1].rstrip()
        return tail

    def _attach_pending_tail_if_any():
        nonlocal pending_tail, buf
        if pending_tail:
            if not buf:
                buf = [pending_tail]
            else:
                if buf[0] and not (pending_tail.endswith((" ", "\n", "\t")) or buf[0][0].isspace()):
                    buf[0] = pending_tail + " " + buf[0]
                else:
                    buf[0] = pending_tail + buf[0]
            pending_tail = ""

    def flush(force=False, allow_overlap=True):
        nonlocal buf, cur_heading, cur_page_start, cur_page_end, idx, pending_tail, chunks

        text = "\n".join(buf).strip()
        if not text:
            return
        L = len(text)

        # respect min size unless forced
        if not force and L < min_chars:
            return

        # If forced but tiny, merge into previous instead of emitting a micro-chunk
        if force and L < min_chars:
            if chunks:
                prev = chunks[-1]
                sep = "\n" if prev["text"] and not prev["text"].endswith("\n") else ""
                prev["text"] = f"{prev['text']}{sep}{text}"
                prev["char_count"] = len(prev["text"])
                if cur_page_end:
                    prev["page_end"] = max(prev.get("page_end") or cur_page_end, cur_page_end)
                safe_ov = max(0, min(overlap_chars, max(0, target_chars - 20)))
                pending_tail = _word_safe_tail(prev["text"], safe_ov) if (allow_overlap and safe_ov > 0) else ""
                buf.clear(); cur_heading = None; cur_page_start = None; cur_page_end = None
                return
            # no previous → keep waiting for growth
            return

        # Emit proper chunk
        chunks.append({
            "id": f"chunk_{idx:04d}",
            "section": (cur_heading or "General"),
            "page_start": cur_page_start,
            "page_end": cur_page_end,
            "char_count": L,
            "text": text,
        })
        idx += 1

        if allow_overlap:
            safe_ov = max(0, min(overlap_chars, max(0, target_chars - 20)))
            pending_tail = _word_safe_tail(text, safe_ov) if safe_ov > 0 else ""
        else:
            pending_tail = ""

        buf = []
        cur_heading = None
        cur_page_start = None
        cur_page_end = None

    # Build chunks
    for b in blocks:
        if b.kind == "heading":
            if buf:
                # Heading is a hard boundary: close WITHOUT overlap carry-over
                flush(force=True, allow_overlap=False)
            cur_heading, cur_page_start, cur_page_end = b.text.rstrip(":"), b.page_start, b.page_end
            continue

        if cur_heading is None: cur_heading = b.heading or "General"
        if cur_page_start is None: cur_page_start = b.page_start
        cur_page_end = b.page_end

        _attach_pending_tail_if_any()

        if b.kind == "table" and keep_table_as_whole:
            if cur_len() + len(b.text) + 1 > max_chars:
                flush(True)
                _attach_pending_tail_if_any()
                if cur_page_start is None: cur_page_start = b.page_start
                cur_page_end = b.page_end
            buf.append(b.text)
            if cur_len() >= target_chars or len(b.text) >= target_chars // 2:
                flush(True)
            continue

        if b.kind == "table" and not keep_table_as_whole:
            rows = [r for r in b.text.splitlines() if r.strip()]
            for row in rows:
                if cur_len() + len(row) + 1 > max_chars:
                    flush(True)
                    _attach_pending_tail_if_any()
                    if cur_page_start is None: cur_page_start = b.page_start
                    cur_page_end = b.page_end
                buf.append(row)
                if cur_len() >= target_chars:
                    flush()
                    _attach_pending_tail_if_any()
                    if cur_page_start is None: cur_page_start = b.page_start
                    cur_page_end = b.page_end
            continue

        for sub in _split_paragraph(b.text, soft=target_chars, hard=max_chars):
            if cur_len() + len(sub) + 1 > max_chars:
                flush(True)
                _attach_pending_tail_if_any()
                if cur_page_start is None: cur_page_start = b.page_start
                cur_page_end = b.page_end
            buf.append(sub)
            if cur_len() >= target_chars:
                flush()
                _attach_pending_tail_if_any()
                if cur_page_start is None: cur_page_start = b.page_start
                cur_page_end = b.page_end

    if buf:
        flush(True)

    # Renumber defensively
    for j, ch in enumerate(chunks, 1):
        ch["id"] = f"chunk_{j:04d}"
        ch["char_count"] = len(ch["text"])
    return chunks

def _limit_chunks(chunks: List[Dict], max_chunks: int) -> List[Dict]:
    if max_chunks is None or max_chunks < 1 or len(chunks) <= max_chunks:
        return chunks
    merged = chunks[:]
    i = 0
    while len(merged) > max_chunks and i < len(merged) - 1:
        a, b = merged[i], merged[i + 1]
        a["text"] = a["text"].rstrip() + "\n" + b["text"].lstrip()
        a["char_count"] = len(a["text"])
        a["page_end"] = b.get("page_end", a["page_end"])
        del merged[i + 1]
        i = min(i + 1, len(merged) - 2) if len(merged) > max_chunks else i
    for j, ch in enumerate(merged, 1):
        ch["id"] = f"chunk_{j:04d}"
        ch["char_count"] = len(ch["text"])
    return merged

# ---------- Extraction (fallback) ----------
def _extract_pdf_best_effort(pdf_path: str, rtl: bool=False, two_cols: bool=True, max_pages: Optional[int]=None) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        out, n = [], len(doc)
        if max_pages is not None: n = min(n, max_pages)
        for pno in range(n):
            page = doc[pno]; out.append(f"=== Page {pno+1} ===")
            if two_cols:
                rect = page.rect; midx = rect.x0 + rect.width/2.0
                left = fitz.Rect(rect.x0, rect.y0, midx, rect.y1)
                right = fitz.Rect(midx, rect.y0, rect.x1, rect.y1)
                lt = page.get_text("text", clip=left) or ""
                rt = page.get_text("text", clip=right) or ""
                out.extend((lt.splitlines() + rt.splitlines()) if not rtl else (rt.splitlines() + lt.splitlines()))
            else:
                out.extend((page.get_text("text") or "").splitlines())
        return "\n".join([ln.rstrip() for ln in out])
    except Exception:
        pass
    try:
        import pdfplumber
        out = []
        with pdfplumber.open(pdf_path) as pdf:
            n = len(pdf.pages)
            if max_pages is not None: n = min(n, max_pages)
            for pno in range(n):
                page = pdf.pages[pno]; out.append(f"=== Page {pno+1} ===")
                if two_cols:
                    w, h, midx = float(page.width), float(page.height), float(page.width)/2.0
                    left, right = page.within_bbox((0,0,midx,h)), page.within_bbox((midx,0,w,h))
                    lt = (left.extract_text() or "").splitlines()
                    rt = (right.extract_text() or "").splitlines()
                    out.extend((lt + rt) if not rtl else (rt + lt))
                else:
                    out.extend((page.extract_text() or "").splitlines())
        return "\n".join([ln.rstrip() for ln in out])
    except Exception as e:
        raise RuntimeError(f"Could not extract PDF text. Install PyMuPDF or pdfplumber. Original error: {e}")

# ---------- Public API ----------
def build_chunks_from_txt(
    txt: str,
    target_chars=1100,
    overlap_chars=150,
    min_chars=200,
    max_chars=2200,
    keep_table_as_whole=True,
    max_chunks: Optional[int]=None,
) -> List[Dict]:
    blocks = _iter_blocks(txt)
    chunks = _chunkize_blocks(blocks, target_chars, overlap_chars, min_chars, max_chars, keep_table_as_whole)
    if max_chunks:
        chunks = _limit_chunks(chunks, max_chunks=max_chunks)
    return chunks

def build_chunks_from_pdf(
    pdf_path: str,
    *,
    extract_fn: Optional[Callable[..., str]] = None,
    rtl: bool=False,
    two_cols: bool=True,
    max_pages: Optional[int]=None,
    target_chars=1100,
    overlap_chars=150,
    min_chars=200,
    max_chars=2200,
    keep_table_as_whole=True,
    max_chunks: Optional[int]=None,
) -> List[Dict]:
    if extract_fn is None:
        try:
            from text_from_pdf import extract_text_from_pdf as _proj_extract
            txt = _proj_extract(pdf_path, two_cols=two_cols, rtl=rtl, max_pages=max_pages)
        except Exception:
            txt = _extract_pdf_best_effort(pdf_path, rtl=rtl, two_cols=two_cols, max_pages=max_pages)
    else:
        txt = extract_fn(pdf_path, two_cols=two_cols, rtl=rtl, max_pages=max_pages)
    return build_chunks_from_txt(
        txt, target_chars=target_chars, overlap_chars=overlap_chars, min_chars=min_chars,
        max_chars=max_chars, keep_table_as_whole=keep_table_as_whole, max_chunks=max_chunks
    )

def write_chunks_jsonl(chunks: List[Dict], outfile: str) -> None:
    with open(outfile, "w", encoding="utf-8") as f:
        for ch in chunks: f.write(json.dumps(ch, ensure_ascii=False) + "\n")

def write_chunks_txt(chunks: List[Dict], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for ch in chunks:
        with open(os.path.join(outdir, f"{ch['id']}.txt"), "w", encoding="utf-8") as f:
            f.write(ch["text"].rstrip() + "\n")

def write_chunks(chunks: List[Dict], outdir: str, basename: str="chunks", clean: bool=True, write_json: bool=False) -> None:
    if clean and os.path.isdir(outdir): shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    if write_json:
        with open(os.path.join(outdir, f"{basename}.jsonl"), "w", encoding="utf-8") as f:
            for ch in chunks: f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    write_chunks_txt(chunks, outdir)

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="English data splitter for MiniRAG (2-column aware, catalogue-friendly)")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("--two-cols", action="store_true", default=True)
    ap.add_argument("--rtl", action="store_true", default=False)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--min", dest="min_chars", type=int, default=150)
    ap.add_argument("--max", dest="max_chars", type=int, default=800)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--outdir", default="chunks_out")
    ap.add_argument("--basename", default="chunks")
    ap.add_argument("--write-json", action="store_true", default=False)
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()

    try:
        from text_from_pdf import extract_text_from_pdf as _proj_extract
        chunks = build_chunks_from_pdf(
            args.pdf, extract_fn=_proj_extract, two_cols=args.two_cols, rtl=args.rtl,
            target_chars=args.target, overlap_chars=args.overlap, min_chars=args.min_chars, max_chars=args.max_chars,
            max_pages=args.max_pages, max_chunks=args.max_chunks
        )
    except Exception:
        chunks = build_chunks_from_pdf(
            args.pdf, extract_fn=None, two_cols=args.two_cols, rtl=args.rtl,
            target_chars=args.target, overlap_chars=args.overlap, min_chars=args.min_chars, max_chars=args.max_chars,
            max_pages=args.max_pages, max_chunks=args.max_chunks
        )

    write_chunks(chunks, args.outdir, basename=args.basename, clean=True, write_json=args.write_json)
    print(f"Wrote {len(chunks)} chunks to {args.outdir} (cleaned). JSON in outdir: {bool(args.write_json)}")


# data_splitter_en.py

from __future__ import annotations
import re, json, os
from typing import List, Dict, Optional, Callable

# --------------------
# Regexes & heuristics
# --------------------

_PAGE_RE = re.compile(r"^===\s*Page\s+(\d+)\s*===\s*$")

# Tables: tabs, pipes, or multi-space alignment (3+ spaces) → looks columnar
_TABLEISH_RE = re.compile(r"(?:[^\t]*\t[^\t]*\t[^\t]*)|(?:\S+\s*\|\s*\S+)|(?:\S+(?:\s{3,}\S+){2,})")

# Lists: bullets, roman numerals, alpha/numeric outlines
_LISTISH_RE = re.compile(
    r"""^\s*(?:[-\u2022\*\u25CF]               # bullets
          |\(\s*[ivxlcdmIVXLCDM]+\s*\)        # (iv) / (IV)
          |[ivxlcdmIVXLCDM]+\s*[.)]\s+        # IV.  iv)
          |\(\s*[a-zA-Z]\s*\)                 # (a)
          |[a-zA-Z]\s*[.)]\s+                 # a.   A)
          |\d+\s*[.)]\s+                      # 1.   2)
         )""",
    re.VERBOSE,
)

# English sentence-ish split (keep colons within sentences)
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?…])\s+")

# Headings
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+){0,5}\s+\S")  # "1", "1.2", "1.2.3.4" + space + text
_APPENDIX_HEADING_RE = re.compile(r"^\s*Appendix\s+[A-Z][\)\.: -]\s*\S", re.IGNORECASE)  # "Appendix A: ..."
_ROMAN_HEADING_RE = re.compile(r"^\s*[IVXLCDM]+\s+[A-Z][^\n]{1,80}$")  # "III REGULATIONS"
_ALL_CAPS_LINE_RE = re.compile(r"^[A-Z][A-Z0-9 &/\-,'()]{2,80}$")      # short all-caps headings


# Contacts/URLs
_CONTACT_RE = re.compile(r"(@|mailto:|tel:|\bhttps?://|\bwww\.)", re.IGNORECASE)

def _normalize_spaces_keep_tabs(s: str) -> str:
    # Collapse all non-tab whitespace to single spaces; keep tabs so table-ish detection works.
    s = re.sub(r"[^\S\t]+", " ", s)
    return s.strip()

def _looks_all_caps(s: str) -> bool:
    return bool(_ALL_CAPS_LINE_RE.match(s)) and len(s) <= 80

def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 2:
        return False
    if len(s) > 120:      # too long to be a heading
        return False
    if _NUMBERED_HEADING_RE.match(s): return True
    if _APPENDIX_HEADING_RE.match(s): return True
    if _ROMAN_HEADING_RE.match(s): return True
    if _looks_all_caps(s): return True
    score = 0
    if s.endswith(":"): score += 2
    punct_count = sum(ch in ",.;?!:|/" for ch in s)
    if punct_count <= 1: score += 1
    if any(term.lower() in s.lower() for term in _COMMON_TERMS): score += 2
    if _LISTISH_RE.match(s): return False
    return score >= 3

def _split_paragraph(s: str, soft: int, hard: int) -> List[str]:
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

    for seg in parts:
        seg = seg.strip()
        if not seg:
            continue

        if len(seg) > hard:
            # word-fallback
            words = seg.split()
            wbuf: List[str] = []
            wlen = 0
            for w in words:
                add = (1 if wlen else 0) + len(w)
                if wlen + add > soft:
                    if wbuf:
                        out.append(" ".join(wbuf))
                    wbuf, wlen = [w], len(w)
                else:
                    wbuf.append(w); wlen += add
            if wbuf:
                out.append(" ".join(wbuf))
            continue

        add = (1 if blen else 0) + len(seg)
        if blen + add > soft:
            flush_buf()
        buf.append(seg); blen += add

    flush_buf()

    # merge tiny tails
    merged: List[str] = []
    for p in out:
        if merged and len(p) < 80 and len(merged[-1]) + 1 + len(p) <= hard:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged

# --------------------
# Block model
# --------------------

class Block:
    __slots__ = ("kind", "text", "page_start", "page_end", "heading")
    def __init__(self, kind: str, text: str, page_start: int, page_end: int, heading: Optional[str]):
        self.kind, self.text, self.page_start, self.page_end, self.heading = kind, text, page_start, page_end, heading

# --------------------
# Header/footer cleaner
# --------------------

def _discover_repeating_lines(lines: List[str], min_repeats=3, max_len=100) -> set:
    """Find lines that repeat many times (likely running headers/footers)."""
    from collections import Counter
    c = Counter([ln.strip() for ln in lines if ln.strip() and not _PAGE_RE.match(ln)])
    return {ln for ln, cnt in c.items() if cnt >= min_repeats and len(ln) <= max_len}

# --------------------
# Block iterator
# --------------------

def _iter_blocks(raw_text: str) -> List[Block]:
    lines = raw_text.splitlines()

    # Identify & drop line-level headers/footers that repeat across pages
    to_drop = _discover_repeating_lines(lines, min_repeats=4, max_len=80)

    cur_page = 1
    blocks: List[Block] = []
    buf: List[str] = []
    buf_kind: Optional[str] = None
    buf_page_start = 1
    section_heading: Optional[str] = None

    def flush():
        nonlocal buf, buf_kind, buf_page_start
        if not buf: return
        txt = "\n".join(buf).strip("\n")
        if txt.strip():
            blocks.append(Block(buf_kind or "para", txt, buf_page_start, cur_page, section_heading))
        buf, buf_kind = [], None

    for ln in lines:
        if _PAGE_RE.match(ln):
            flush()
            try: cur_page = int(_PAGE_RE.match(ln).group(1))
            except: pass
            continue
        line = _normalize_spaces_keep_tabs(ln)
        if not line or line in to_drop:
            if not line: flush()
            continue

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

    # propagate last heading down
    last_heading: Optional[str] = None
    for b in blocks:
        if b.kind == "heading":
            last_heading, b.heading = b.text.rstrip(":"), None
        else:
            b.heading = last_heading
    return blocks

# --------------------
# Chunking
# --------------------

def _chunkize_blocks(
    blocks: List[Block],
    target_chars=1200,
    overlap_chars=150,
    min_chars=200,
    max_chars=2200,
    keep_table_as_whole=True
) -> List[Dict]:

    chunks: List[Dict] = []
    buf: List[str] = []
    cur_heading: Optional[str] = None
    cur_page_start: Optional[int] = None
    cur_page_end: Optional[int] = None
    idx = 1

    def cur_len() -> int: return sum(len(s)+1 for s in buf)

    def flush(force=False):
        nonlocal buf, cur_heading, cur_page_start, cur_page_end, idx
        text = "\n".join(buf).strip()
        if not text: return
        L = len(text)
        if force or L >= min_chars:
            chunks.append({
                "id": f"chunk_{idx:04d}",
                "section": (cur_heading or "General"),
                "page_start": cur_page_start,
                "page_end": cur_page_end,
                "char_count": L,
                "text": text,
            })
            idx += 1
            if overlap_chars > 0 and len(text) > overlap_chars:
                buf, cur_page_start = [text[-overlap_chars:]], cur_page_end
            else:
                buf, cur_heading, cur_page_start, cur_page_end = [], None, None, None

    for b in blocks:
        if b.kind == "heading":
            flush(True)
            cur_heading, cur_page_start, cur_page_end = b.text.rstrip(":"), b.page_start, b.page_end
            continue

        if cur_heading is None: cur_heading = b.heading or "General"
        if cur_page_start is None: cur_page_start = b.page_start
        cur_page_end = b.page_end

        if b.kind == "table" and keep_table_as_whole:
            if cur_len() + len(b.text) + 1 > max_chars:
                flush(True)
            buf.append(b.text)
            if cur_len() >= target_chars or len(b.text) >= target_chars//2:
                flush(True)
            continue

        if b.kind == "table" and not keep_table_as_whole:
            for row in [r for r in b.text.splitlines() if r.strip()]:
                if cur_len() + len(row) + 1 > max_chars:
                    flush(True)
                buf.append(row)
                if cur_len() >= target_chars:
                    flush()
            continue

        # non-table: split long paras before appending
        for sub in _split_paragraph(b.text, soft=target_chars, hard=max_chars):
            if cur_len() + len(sub) + 1 > max_chars:
                flush(True)
            buf.append(sub)
            if cur_len() >= target_chars:
                flush()

    flush(True)
    return chunks

# --------------------
# PDF extraction (best-effort, optional)
# --------------------

def _extract_pdf_best_effort(pdf_path: str, rtl: bool=False, two_cols: bool=True, max_pages: Optional[int]=None) -> str:
    """
    Try PyMuPDF first; if not available, try pdfplumber.
    Produce text with page markers and 2-column ordering (left→right).
    """
    # Try PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        out_lines: List[str] = []
        n_pages = len(doc)
        if max_pages is not None:
            n_pages = min(n_pages, max_pages)
        for pno in range(n_pages):
            page = doc[pno]
            out_lines.append(f"=== Page {pno+1} ===")
            if two_cols:
                rect = page.rect
                midx = rect.x0 + rect.width / 2.0
                left  = fitz.Rect(rect.x0, rect.y0, midx, rect.y1)
                right = fitz.Rect(midx, rect.y0, rect.x1, rect.y1)
                # Extract left then right as plain text (keeps line breaks)
                left_text  = page.get_text("text", clip=left) or ""
                right_text = page.get_text("text", clip=right) or ""
                # Normalize RTL if needed (not typical for English docs)
                if rtl:
                    # naive RTL join: right then left
                    out_lines.extend([ln.rstrip() for ln in right_text.splitlines()])
                    out_lines.extend([ln.rstrip() for ln in left_text.splitlines()])
                else:
                    out_lines.extend([ln.rstrip() for ln in left_text.splitlines()])
                    out_lines.extend([ln.rstrip() for ln in right_text.splitlines()])
            else:
                txt = page.get_text("text") or ""
                out_lines.extend([ln.rstrip() for ln in txt.splitlines()])
        return "\n".join(out_lines)
    except Exception:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        out_lines: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            if max_pages is not None:
                n_pages = min(n_pages, max_pages)
            for pno in range(n_pages):
                page = pdf.pages[pno]
                out_lines.append(f"=== Page {pno+1} ===")
                if two_cols:
                    w = float(page.width); h = float(page.height)
                    midx = w / 2.0
                    left  = page.within_bbox((0, 0, midx, h))
                    right = page.within_bbox((midx, 0, w, h))
                    if rtl:
                        # right then left
                        rt = (right.extract_text() or "").splitlines()
                        lt = (left.extract_text()  or "").splitlines()
                        out_lines.extend([ln.rstrip() for ln in rt])
                        out_lines.extend([ln.rstrip() for ln in lt])
                    else:
                        lt = (left.extract_text()  or "").splitlines()
                        rt = (right.extract_text() or "").splitlines()
                        out_lines.extend([ln.rstrip() for ln in lt])
                        out_lines.extend([ln.rstrip() for ln in rt])
                else:
                    txt = page.extract_text() or ""
                    out_lines.extend([ln.rstrip() for ln in txt.splitlines()])
        return "\n".join(out_lines)
    except Exception as e:
        raise RuntimeError(f"Could not extract PDF text. Install PyMuPDF or pdfplumber. Original error: {e}")

# --------------------
# Public API
# --------------------

def build_chunks_from_txt(
    txt: str,
    target_chars=1200,
    overlap_chars=150,
    min_chars=200,
    max_chars=2200,
    keep_table_as_whole=True
) -> List[Dict]:
    blocks = _iter_blocks(txt)
    return _chunkize_blocks(blocks, target_chars, overlap_chars, min_chars, max_chars, keep_table_as_whole)

def build_chunks_from_pdf(
    pdf_path: str,
    *,
    extract_fn: Optional[Callable[..., str]] = None,
    rtl: bool=False,
    two_cols: bool=True,
    max_pages: Optional[int]=None,
    target_chars=1200,
    overlap_chars=150,
    min_chars=200,
    max_chars=2200,
    keep_table_as_whole=True
) -> List[Dict]:
    """
    If extract_fn is None, uses _extract_pdf_best_effort (PyMuPDF -> pdfplumber).
    Your extract_fn should accept: (pdf_path, rtl=False, two_cols=True, max_pages=None) and return a raw text string.
    """
    if extract_fn is None:
        txt = _extract_pdf_best_effort(pdf_path, rtl=rtl, two_cols=two_cols, max_pages=max_pages)
    else:
        txt = extract_fn(pdf_path, rtl=rtl, two_cols=two_cols, max_pages=max_pages)
    return build_chunks_from_txt(txt, target_chars, overlap_chars, min_chars, max_chars, keep_table_as_whole)

def write_chunks_jsonl(chunks: List[Dict], outfile: str) -> None:
    with open(outfile, "w", encoding="utf-8") as f:
        for ch in chunks: f.write(json.dumps(ch, ensure_ascii=False) + "\n")

def write_chunks_txt(chunks: List[Dict], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for ch in chunks:
        with open(os.path.join(outdir, f"{ch['id']}.txt"), "w", encoding="utf-8") as f:
            f.write(ch["text"].rstrip() + "\n")

# ---------------
# CLI (optional)
# ---------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="English data splitter for MiniRAG (2-column aware)")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("--two-cols", action="store_true", default=True, help="Treat pages as two columns")
    ap.add_argument("--target", type=int, default=1200, help="Target characters per chunk")
    ap.add_argument("--overlap", type=int, default=150, help="Overlap characters between chunks")
    ap.add_argument("--min", dest="min_chars", type=int, default=200, help="Minimum chunk size to flush")
    ap.add_argument("--max", dest="max_chars", type=int, default=2200, help="Hard cap on chunk size")
    ap.add_argument("-o", "--out", default="chunks.jsonl", help="Output JSONL file")
    args = ap.parse_args()

    chunks = build_chunks_from_pdf(
        args.pdf,
        two_cols=args.two_cols,
        target_chars=args.target,
        overlap_chars=args.overlap,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    write_chunks_jsonl(chunks, args.out)
    print(f"Wrote {len(chunks)} chunks to {args.out}")

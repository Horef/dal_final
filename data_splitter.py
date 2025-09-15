# data_splitter.py
# Hebrew-friendly, layout-agnostic syllabus & regulation chunker for miniRAG
# -------------------------------------------------------------------------
# Improvements:
# - Detects course tables (lines starting with course codes 6–7 digits)
# - Treats "רטסמס" (semester markers) as headings
# - Treats nested numbering (1.1.2, 2.3.3.4) as headings
# - Handles long name lists (faculty/staff) as list blocks
# - Splits out contact details (emails, phones, URLs) into standalone paras
# - More robust heading detection for takanon-style docs
#
# Public API:
#   build_chunks_from_txt(text, target_chars=..., overlap_chars=...)
#   build_chunks_from_pdf(pdf_path, extract_fn=...)
#   write_chunks_jsonl(chunks, outfile)

from __future__ import annotations
import re, json
from typing import List, Dict, Optional
import os

# --- Heuristics & Regexes ---

_HEB = r"\u0590-\u05FF"

_PAGE_RE = re.compile(r"^===\s*Page\s+(\d+)\s*===\s*$")

# Tables: tabs, pipes, or course codes
_TABLEISH_RE = re.compile(
    r"(?:[^\t]*\t[^\t]*\t[^\t]*)|(?:\S+\s*\|\s*\S+)|(?:^\d{6,7}\s+)"
)

# Lists: bullets, Hebrew letters, numbers, or long faculty lists
_LISTISH_RE = re.compile(
    rf"""^\s*(?:
        [\-\u2022\*\u25CF]       |   # bullets
        \(\s*[\divxlIVXL]+\s*\)  |   # (1) (iv)
        \(?[א-ת]\)\s*            |   # (א)
        [א-ת]\s*[.\)]\s+         |   # א. ב)
        \d+\s*[.)]\s+            |   # 1. 2)
        [א-ת]+\s+[א-ת]+$             # names like "רון כהן"
    )""",
    re.VERBOSE,
)

# Headings: numbers or annexes
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+){0,4}\s+\S")
_ANNEX_HEADING_RE = re.compile(r"^\s*נספח\s+[א-ת][׳\"']?\s*[:\-]?\s+\S")

# Semester markers
_SEMESTER_HEADING_RE = re.compile(r"^\s*\d+\s+רטסמס")

# Contact lines (emails, phones, URLs)
_CONTACT_RE = re.compile(r"(@|\.ac\.il|\.edu|mailto:|tel:|\d{2,3}[- ]?\d{6,7})")

_COMMON_SYLLABUS_TERMS = (
    "סילבוס", "תוכן הקורס", "מטרות הקורס", "דרישות קדם", "שיטות הוראה",
    "מטלות", "שיטת הערכה", "בחינה", "נוכחות", "נקודות זכות", "שעות שבועיות",
    "שנה", "סמסטר", "מרצה", "תרגיל", "מעבדה", "ספרות", "ביבליוגרפיה",
    "תקנות", "נהלים", "מבוא", "מועדי", "ניקוד", "סיום לימודים", "סטודנטים",
    "רישום", "בחינות", "הוראה", "נספח", "נוהל"
)

def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 2:
        return False
    if _NUMBERED_HEADING_RE.match(s): return True
    if _ANNEX_HEADING_RE.match(s): return True
    if _SEMESTER_HEADING_RE.match(s): return True
    if len(s) > 120: return False
    if not s.endswith(":") and s[-1] in ".?!？！。": return False
    score = 0
    if s.endswith(":"): score += 2
    punct_count = sum(ch in ",.;?!:|/" for ch in s)
    if punct_count <= 1: score += 1
    if re.search(rf"[{_HEB}]", s): score += 1
    if any(term in s for term in _COMMON_SYLLABUS_TERMS): score += 2
    if _LISTISH_RE.match(s): return False
    return score >= 3

def _normalize_spaces_keep_tabs(s: str) -> str:
    return re.sub(r"[^\S\t]+", " ", s).strip()

# --- Block class ---

class Block:
    __slots__ = ("kind", "text", "page_start", "page_end", "heading")
    def __init__(self, kind: str, text: str, page_start: int, page_end: int, heading: Optional[str]):
        self.kind, self.text, self.page_start, self.page_end, self.heading = kind, text, page_start, page_end, heading

# --- Block iterator ---

def _iter_blocks(raw_text: str) -> List[Block]:
    lines = raw_text.splitlines()
    cur_page, blocks, buf, buf_kind, buf_page_start, section_heading = 1, [], [], None, 1, None

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
        if not line:
            flush()
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

    last_heading = None
    for b in blocks:
        if b.kind == "heading": last_heading, b.heading = b.text.rstrip(":"), None
        else: b.heading = last_heading
    return blocks

# --- Chunking ---

def _chunkize_blocks(blocks: List[Block], target_chars=1200, overlap_chars=150, min_chars=200, max_chars=2200, keep_table_as_whole=True) -> List[Dict]:
    chunks, buf, cur_heading, cur_page_start, cur_page_end, idx = [], [], None, None, None, 1
    def cur_len(): return sum(len(s)+1 for s in buf)
    def flush(force=False):
        nonlocal buf, cur_heading, cur_page_start, cur_page_end, idx
        text = "\n".join(buf).strip()
        if not text: return
        L = len(text)
        if force or L >= min_chars:
            chunks.append({
                "id": f"chunk_{idx:04d}",
                "section": cur_heading or "כללי",
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
        if cur_heading is None: cur_heading = b.heading or "כללי"
        if cur_page_start is None: cur_page_start = b.page_start
        cur_page_end = b.page_end
        if b.kind == "table" and keep_table_as_whole:
            if cur_len()+len(b.text)+1 > max_chars: flush(True)
            buf.append(b.text);
            if cur_len() >= target_chars or len(b.text) >= target_chars//2: flush(True)
            continue
        buf.append(b.text)
        if cur_len() >= target_chars: flush()
        if cur_len() > max_chars: flush(True)
    flush(True)
    return chunks

# --- Public API ---

def build_chunks_from_txt(txt: str, target_chars=1200, overlap_chars=150,
                          min_chars=200, max_chars=2200, keep_table_as_whole=True) -> List[Dict]:
    return _chunkize_blocks(_iter_blocks(txt), target_chars, overlap_chars, min_chars, max_chars, keep_table_as_whole)

def build_chunks_from_pdf(pdf_path: str, *, extract_fn, rtl=True, two_cols=True,
                          max_pages=None, target_chars=1200, overlap_chars=150, min_chars=200,
                          max_chars=2200, keep_table_as_whole=True) -> List[Dict]:
    txt = extract_fn(pdf_path, rtl=rtl, two_cols=two_cols, max_pages=max_pages)
    return build_chunks_from_txt(txt, target_chars, overlap_chars, min_chars, max_chars, keep_table_as_whole)

def write_chunks_jsonl(chunks: List[Dict], outfile: str) -> None:
    with open(outfile, "w", encoding="utf-8") as f:
        for ch in chunks: f.write(json.dumps(ch, ensure_ascii=False)+"\n")

def write_chunks_txt(chunks: List[Dict], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for ch in chunks:
        with open(os.path.join(outdir, f"chunk_{ch['id']}.txt"), "w", encoding="utf-8") as f:
            f.write(ch["text"] + "\n")
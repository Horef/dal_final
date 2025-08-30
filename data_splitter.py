# data_splitter.py
# Hebrew-friendly, layout-agnostic syllabus chunker for miniRAG
# --------------------------------------------------------------
# - Groups lines into semantic blocks (headings, paragraphs, lists, tables)
# - Uses page markers like "=== Page N ===" if present
# - Preserves list/table structure when possible (tabs or pipes)
# - Produces overlapping chunks by characters (configurable)
# - No external dependencies (stdlib only)
#
# Typical usage from main.py:
#   from data_splitter import build_chunks_from_txt
#   chunks = build_chunks_from_txt(text, target_chars=1200, overlap_chars=150)
#   # optionally write JSONL:
#   from data_splitter import write_chunks_jsonl
#   write_chunks_jsonl(chunks, "Processed Data/my_syllabus.chunks.jsonl")

from __future__ import annotations
import re
import json
from typing import List, Dict, Optional, Tuple

# --- Heuristics & Regexes ---

_HEB = r"\u0590-\u05FF"  # Hebrew block
# Page marker your extractor already emits
_PAGE_RE = re.compile(r"^===\s*Page\s+(\d+)\s*===\s*$")

# A "table-like" line: has >= 2 tabs OR explicit pipe columns
_TABLEISH_RE = re.compile(r"(?:[^\t]*\t[^\t]*\t[^\t]*)|(?:\S+\s*\|\s*\S+)")

# A "list-ish" line: bullets, dashes, or enumerations (Hebrew or numeric)
_LISTISH_RE = re.compile(
    rf"""^\s*(?:
        [\-\u2022\*\u25CF]               |   # bullets -, •, *, ●
        \(\s*[\divxlIVXL]+\s*\)          |   # (1) (2) (iv) ...
        \(?[א-ת]\)\s*                    |   # (א) (ב) ...
        [א-ת]\s*[.\)]\s+                 |   # א.  ב) ...
        \d+\s*[.)]\s+                        # 1.  2)
    )""",
    re.VERBOSE,
)

# Numbered headings like: "1. תקנות ...", "1.1.2 תוכנית הלימודים ..."
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+){0,3}\s+\S")

# Appendix headings like: "נספח א׳: נוהל הגשת תוכנית לימוד"
_ANNEX_HEADING_RE = re.compile(r"^\s*נספח\s+[א-ת][׳\"']?\s*[:\-]?\s+\S")

# A "heading-ish" line (Hebrew friendly):
#  - short/medium length
#  - not ending with .?! unless it ends with colon:
#  - low punctuation density or ends with ':'
#  - contains common syllabus terms OR looks like a section caption
_COMMON_SYLLABUS_TERMS = (
    "סילבוס", "תוכן הקורס", "מטרות הקורס", "דרישות קדם", "שיטות הוראה",
    "מטלות", "שיטת הערכה", "בחינה", "נוכחות", "נקודות זכות", "שעות שבועיות",
    "שנה", "סמסטר", "מרצה", "תרגיל", "מעבדה", "ספרות", "ביבליוגרפיה",
    # policy/handbook terms commonly seen in תקנות/נהלים:
    "תקנות", "נהלים", "מבוא", "מועדי", "ניקוד", "סיום לימודים", "סטודנטים",
    "רישום", "בחינות", "הוראה", "נספח", "נוהל"
)
def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 2:
        return False

    # hard positives for numbered/appendix headings
    if _NUMBERED_HEADING_RE.match(s):
        return len(s) <= 140  # guard against accidental overlong lines

    if _ANNEX_HEADING_RE.match(s):
        return True

    if len(s) > 120:  # too long to be a title
        return False

    # Do not treat obvious sentences as headings unless they end with ':'
    if not s.endswith(":") and s[-1] in ".?!？！。":
        return False

    # Punctuation density
    punct_count = sum(ch in ",.;?!:|/" for ch in s)
    if s.endswith(":"):
        score = 2
    else:
        score = 0
    if punct_count <= 1:
        score += 1

    if re.search(rf"[{_HEB}]", s):
        score += 1

    if any(term in s for term in _COMMON_SYLLABUS_TERMS):
        score += 2

    # Don’t start with a bullet/numbering (that’s a list)
    if _LISTISH_RE.match(s):
        return False

    return score >= 3


def _normalize_spaces_keep_tabs(s: str) -> str:
    """
    Collapse whitespace except tabs (tabs are crucial to preserve
    table-ish alignment coming from the extractor).
    """
    # Replace any whitespace that is NOT a tab with a single space
    s = re.sub(r"[^\S\t]+", " ", s)
    # Trim
    return s.strip()


# --- Block building ---

class Block:
    __slots__ = ("kind", "text", "page_start", "page_end", "heading")
    def __init__(self, kind: str, text: str, page_start: int, page_end: int, heading: Optional[str]):
        self.kind = kind            # 'heading', 'para', 'list', 'table'
        self.text = text
        self.page_start = page_start
        self.page_end = page_end
        self.heading = heading      # nearest preceding heading (context)

    def __len__(self) -> int:
        return len(self.text)


def _iter_blocks(raw_text: str) -> List[Block]:
    """
    Turn raw syllabus text into a sequence of typed blocks (headings/lists/tables/paras),
    remembering page numbers and nearest section heading.
    """
    lines = raw_text.splitlines()
    cur_page = 1
    blocks: List[Block] = []
    buf: List[str] = []
    buf_kind: Optional[str] = None
    section_heading: Optional[str] = None
    buf_page_start = cur_page

    def flush():
        nonlocal buf, buf_kind, buf_page_start
        if not buf:
            return
        txt = "\n".join(buf).strip("\n")
        if txt.strip():
            blocks.append(Block(buf_kind or "para", txt, buf_page_start, cur_page, section_heading))
        buf = []
        buf_kind = None

    for ln in lines:
        m = _PAGE_RE.match(ln)
        if m:
            # new page marker → flush current block
            flush()
            try:
                cur_page = int(m.group(1))
            except Exception:
                cur_page = cur_page
            continue

        # Normalize (keep tabs!)
        line = _normalize_spaces_keep_tabs(ln)

        if not line:
            # blank line -> paragraph/list/table breaker
            flush()
            continue

        # Classify line
        if _is_heading(line):
            # Heading starts a new block, flush first
            flush()
            blocks.append(Block("heading", line, cur_page, cur_page, heading=None))
            section_heading = line.rstrip(":")
            continue

        is_table = bool(_TABLEISH_RE.search(line))
        is_list  = bool(_LISTISH_RE.match(line))

        # Merge strategy: keep homogeneous sequences together
        kind = "table" if is_table else "list" if is_list else "para"
        if buf_kind != kind:
            # different kind → flush and start new
            flush()
            buf_kind = kind
            buf_page_start = cur_page

        buf.append(line)

    flush()
    # Assign nearest preceding heading to non-heading blocks
    last_heading: Optional[str] = None
    for b in blocks:
        if b.kind == "heading":
            last_heading = b.text.rstrip(":")
            b.heading = None
        else:
            b.heading = last_heading
    return blocks


# --- Chunking ---

def _chunkize_blocks(
    blocks: List[Block],
    target_chars: int = 1200,
    overlap_chars: int = 150,
    min_chars: int = 200,
    max_chars: int = 2200,
    keep_table_as_whole: bool = True,
) -> List[Dict]:
    """
    Turn blocks into overlapping chunks by characters.
    - Flush on heading boundaries
    - Try not to split a single table block if keep_table_as_whole=True
    """
    chunks: List[Dict] = []
    idx = 1

    cur_heading: Optional[str] = None
    cur_page_start: Optional[int] = None
    cur_page_end: Optional[int] = None
    buf: List[str] = []

    def cur_len() -> int:
        return sum(len(s) + 1 for s in buf)

    def flush(force: bool = False):
        nonlocal idx, buf, cur_heading, cur_page_start, cur_page_end
        text = "\n".join(buf).strip()
        if not text:
            return
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
                # keep overlap from END of previous chunk
                keep = text[-overlap_chars:]
                buf = [keep]
                # keep same heading, but page_start becomes unknown/continued
                cur_page_start = cur_page_end
            else:
                buf = []
                cur_heading = None
                cur_page_start = None
                cur_page_end = None

    for b in blocks:
        if b.kind == "heading":
            # New heading → close current chunk (forcing if anything there)
            flush(force=True)
            cur_heading = b.text.rstrip(":")
            cur_page_start = b.page_start
            cur_page_end = b.page_end
            # No text added for heading-only line; we let next content fill
            continue

        # Update heading if missing
        if cur_heading is None:
            cur_heading = b.heading or "כללי"

        # Initialize pages
        if cur_page_start is None:
            cur_page_start = b.page_start
        cur_page_end = b.page_end

        # If table and we should keep it whole, consider flushing before/after
        if b.kind == "table" and keep_table_as_whole:
            # If adding it would exceed max_chars, flush first
            if cur_len() + len(b.text) + 1 > max_chars:
                flush(force=True)
                cur_heading = cur_heading or (b.heading or "כללי")
                cur_page_start = b.page_start
                cur_page_end = b.page_end

            buf.append(b.text)
            # If table is huge by itself, flush immediately
            if cur_len() >= target_chars or len(b.text) >= target_chars // 2:
                flush(force=True)
            continue

        # Normal add
        buf.append(b.text)

        if cur_len() >= target_chars:
            flush(force=False)

        # Do not exceed max hard limit
        if cur_len() > max_chars:
            flush(force=True)

    # Final flush
    flush(force=True)
    return chunks


# --- Public API ---

def build_chunks_from_txt(
    txt: str,
    target_chars: int = 1200,
    overlap_chars: int = 150,
    min_chars: int = 200,
    max_chars: int = 2200,
    keep_table_as_whole: bool = True,
) -> List[Dict]:
    """
    Build overlapping chunks from already-extracted plain text.
    Returns a list of chunk dicts: {id, section, page_start, page_end, char_count, text}
    """
    blocks = _iter_blocks(txt)
    chunks = _chunkize_blocks(
        blocks,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        keep_table_as_whole=keep_table_as_whole,
    )
    return chunks


def build_chunks_from_pdf(
    pdf_path: str,
    *,
    extract_fn,                # pass: extract_text_from_pdf
    rtl: bool = True,
    two_cols: bool = True,
    max_pages: Optional[int] = None,
    target_chars: int = 1200,
    overlap_chars: int = 150,
    min_chars: int = 200,
    max_chars: int = 2200,
    keep_table_as_whole: bool = True,
) -> List[Dict]:
    """
    Convenience: extract text then chunk.
    You must pass your extractor function as extract_fn to avoid import cycles.
    """
    txt = extract_fn(pdf_path, rtl=rtl, two_cols=two_cols, max_pages=max_pages)
    return build_chunks_from_txt(
        txt,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        keep_table_as_whole=keep_table_as_whole,
    )


def write_chunks_jsonl(chunks: List[Dict], outfile: str) -> None:
    with open(outfile, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

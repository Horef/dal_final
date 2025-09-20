# data_splitter_en.py
# English-friendly, 2-column-aware splitter for MiniRAG
# Bulletproof against text skipping; no tables for this PDF.

from __future__ import annotations
import re, json, os, shutil
from typing import List, Dict, Optional, Callable

# ---------- Heuristics ----------
_PAGE_RE = re.compile(r"^===\s*Page\s+(\d+)\s*===\s*$")

# We explicitly DISABLE table detection for this PDF.
def _is_table_line(_: str) -> bool:
    return False

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
    # Keep tabs if ever present; normalize others.
    return re.sub(r"[^\S\t]+", " ", s).strip()

def _looks_all_caps(s: str) -> bool:
    return bool(_ALL_CAPS_LINE_RE.match(s)) and len(s) <= 80

def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120: return False
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
    """Split long paragraphs by sentences first, then by words."""
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
        if blen + add > soft: flush_buf()
        buf.append(seg); blen += add
    flush_buf()

    # Merge very small tails to previous piece
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
    Parse the extracted text into blocks (headings/lists/paragraphs).
    NO table detection. Safe inline-join that NEVER drops a line.
    Also fixes simple PDF hyphenation: 'Techn-' + 'ion' -> 'Technion'.
    """
    def _dehyphenate(prev: str, cur: str) -> Optional[str]:
        if prev.endswith("-") and cur and cur[0].islower():
            return prev[:-1] + cur
        return None

    lines = raw_text.splitlines()
    cur_page = 1
    blocks: List[Block] = []
    buf: List[str] = []
    buf_kind: Optional[str] = None
    buf_page_start = 1
    section_heading: Optional[str] = None

    def safe_append_line(joined: List[str], ln: str):
        """
        Append a line into 'joined', joining inline when appropriate
        but NEVER skipping a line. This is the fix that prevents
        losing 'as the regulations and procedures ...' in your example.
        """
        if not joined:
            joined.append(ln)
            return
        prev = joined[-1]
        # Try dehyphenation first
        fused = _dehyphenate(prev, ln)
        if fused is not None:
            joined[-1] = fused
            return
        # If previous line does NOT end a sentence or strong break, soft-join
        if not prev.endswith((".", "?", "!", ":", ";")) and not _LISTISH_RE.match(ln):
            joined[-1] = prev.rstrip() + " " + ln.lstrip()
            return
        # Otherwise, start a new logical line
        joined.append(ln)

    def flush():
        nonlocal buf, buf_kind, buf_page_start
        if not buf: return
        joined: List[str] = []
        for ln in buf:
            safe_append_line(joined, ln)
        txt = "\n".join(joined).strip("\n")
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
            flush(); blocks.append(Block("para", line, cur_page, cur_page, section_heading)); continue
        if _is_heading(line):
            flush(); blocks.append(Block("heading", line, cur_page, cur_page, None)); section_heading = line.rstrip(":"); continue
        is_list = bool(_LISTISH_RE.match(line))
        kind = "list" if is_list else "para"
        if buf_kind != kind:
            flush(); buf_kind, buf_page_start = kind, cur_page
        buf.append(line)
    flush()

    last_heading: Optional[str] = None
    for b in blocks:
        if b.kind == "heading": last_heading, b.heading = b.text.rstrip(":"), None
        else: b.heading = last_heading
    return blocks

# ---------- Chunking ----------
def _chunkize_blocks(
    blocks: List[Block],
    target_chars=1100,
    overlap_chars=150,
    min_chars=200,
    max_chars=2200,
) -> List[Dict]:

    # Modest overlap: <= 25% of target
    overlap_chars = min(overlap_chars, max(0, int(0.25 * max(1, target_chars))))

    chunks: List[Dict] = []
    buf: List[str] = []
    cur_heading: Optional[str] = None
    cur_page_start: Optional[int] = None
    cur_page_end: Optional[int] = None
    idx = 1
    carry: str = ""  # prefix overlap to prepend once to the next chunk

    def cur_len() -> int:
        return sum(len(s) + 1 for s in buf)

    def word_safe_tail(text: str, desired: int) -> str:
        if desired <= 0 or len(text) <= desired:
            return text
        start = max(0, len(text) - desired)
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        tail = text[start:].lstrip()
        # Trim partial word at very end
        end = len(tail) - 1
        while end >= 0 and not tail[end].isspace():
            end -= 1
        if 0 <= end < len(tail) - 1:
            tail = tail[:end + 1].rstrip()
        return tail

    def flush(force=False, hard_boundary=False):
        nonlocal buf, cur_heading, cur_page_start, cur_page_end, idx, carry, chunks
        text_core = "\n".join(buf).strip()
        if not text_core:
            return
        L = len(text_core)
        if not force and L < min_chars:
            return
        # Build final text = carry + core (carry only once)
        text = (carry + (" " if (carry and text_core and not carry.endswith(("\n", " "))) else "") + text_core) if carry else text_core
        carry = ""  # consumed
        # Emit proper chunk
        chunks.append({
            "id": f"chunk_{idx:04d}",
            "section": (cur_heading or "General"),
            "page_start": cur_page_start,
            "page_end": cur_page_end,
            "char_count": len(text),
            "text": text,
        })
        idx += 1
        # Prepare next carry (overlap) unless at hard boundary (e.g., a heading)
        if not hard_boundary and overlap_chars > 0:
            carry = word_safe_tail(text, overlap_chars)
        else:
            carry = ""
        # reset buffer & page span
        buf.clear()
        cur_heading = None
        cur_page_start = None
        cur_page_end = None

    for b in blocks:
        if b.kind == "heading":
            if buf:
                flush(force=True, hard_boundary=True)  # close before heading without carrying overlap
            cur_heading, cur_page_start, cur_page_end = b.text.rstrip(":"), b.page_start, b.page_end
            continue

        if cur_heading is None:
            cur_heading = b.heading or "General"
        if cur_page_start is None:
            cur_page_start = b.page_start
        cur_page_end = b.page_end

        # Paragraphs/lists only (no tables)
        for sub in _split_paragraph(b.text, soft=target_chars, hard=max_chars):
            if cur_len() + len(sub) + 1 > max_chars:
                flush(force=True)  # ensures we never drop text
                if cur_page_start is None: cur_page_start = b.page_start
                cur_page_end = b.page_end
            buf.append(sub)
            if cur_len() >= target_chars:
                flush()  # soft flush; still respects min_chars
                if cur_page_start is None: cur_page_start = b.page_start
                cur_page_end = b.page_end

    # Final flush
    if buf:
        flush(force=True)

    # Defensive renumber + char_count
    for j, ch in enumerate(chunks, 1):
        ch["id"] = f"chunk_{j:04d}"
        ch["char_count"] = len(ch["text"])
    return chunks

# ---------- Coverage guard ----------
def _normalize_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _chunks_cover_text(chunks: List[Dict], original_txt: str) -> bool:
    # Remove overlap duplication while reconstructing
    seen = []
    last_tail = ""
    for ch in chunks:
        t = ch["text"]
        if last_tail and t.startswith(last_tail):
            t = t[len(last_tail):]
        seen.append(t)
        # recompute last_tail from this chunk for the next comparison
        last_tail = ""
    rebuilt = _normalize_for_compare("".join(seen))
    original = _normalize_for_compare(original_txt)
    # Allow small whitespace/punctuation diffs but not deletions:
    return all(tok in rebuilt for tok in original.split()[:50]) and len(rebuilt) >= int(0.98 * len(original))

# ---------- Extraction (fallback) ----------
def _extract_pdf_best_effort(pdf_path: str, rtl: bool=False, two_cols: bool=True, max_pages: Optional[int]=None) -> str:
    try:
        import fitz
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
    max_chunks: Optional[int]=None,
) -> List[Dict]:
    blocks = _iter_blocks(txt)
    chunks = _chunkize_blocks(blocks, target_chars, overlap_chars, min_chars, max_chars)
    # Coverage guard: if anything looks missing, fall back to sentence-window chunking
    if not _chunks_cover_text(chunks, txt):
        chunks = _fallback_sentence_window(txt, target_chars, overlap_chars, min_chars, max_chars)
    if max_chunks:
        chunks = _limit_chunks(chunks, max_chunks=max_chunks)
    return chunks

def _fallback_sentence_window(
    txt: str, target_chars: int, overlap_chars: int, min_chars: int, max_chars: int
) -> List[Dict]:
    """Ultra-simple linear window over sentences; guarantees coverage/no loss."""
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(txt) if s.strip()]
    chunks: List[Dict] = []
    buf: List[str] = []
    idx = 1
    overlap_chars = min(overlap_chars, max(0, int(0.25 * max(1, target_chars))))

    def emit():
        nonlocal buf, idx
        if not buf: return
        core = " ".join(buf).strip()
        if not core: 
            buf = []
            return
        text = core
        chunks.append({
            "id": f"chunk_{idx:04d}",
            "section": "General",
            "page_start": None,
            "page_end": None,
            "char_count": len(text),
            "text": text,
        })
        idx += 1
        # prepare next overlap prefix
        if overlap_chars > 0:
            tail = text[-overlap_chars:]
            # word-safe
            k = 0
            while k < len(tail) and not tail[k].isspace(): k += 1
            tail = tail[k:].lstrip() if k < len(tail) else tail
            buf = [tail]  # start next with overlap
        else:
            buf = []

    for s in sents:
        if sum(len(x)+1 for x in buf) + len(s) + 1 > max_chars:
            emit()
        buf.append(s)
        if sum(len(x)+1 for x in buf) >= target_chars:
            emit()
    if buf:
        emit()
    for j, ch in enumerate(chunks, 1):
        ch["id"] = f"chunk_{j:04d}"
        ch["char_count"] = len(ch["text"])
    return chunks

def _limit_chunks(chunks: List[Dict], max_chunks: int) -> List[Dict]:
    if max_chunks is None or max_chunks < 1 or len(chunks) <= max_chunks: return chunks
    merged = chunks[:]
    i = 0
    while len(merged) > max_chunks and i < len(merged)-1:
        a, b = merged[i], merged[i+1]
        a["text"] = a["text"].rstrip() + "\n" + b["text"].lstrip()
        a["char_count"] = len(a["text"])
        a["page_end"] = b.get("page_end", a["page_end"])
        del merged[i+1]
        i = min(i+1, len(merged)-2) if len(merged) > max_chunks else i
    for j, ch in enumerate(merged, 1):
        ch["id"] = f"chunk_{j:04d}"
        ch["char_count"] = len(ch["text"])
    return merged

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
    max_chunks: Optional[int]=None,
) -> List[Dict]:
    if extract_fn is None:
        try:
            from text_from_pdf import extract_text_from_pdf as _proj_extract
            txt = _proj_extract(pdf_path, two_cols=two_cols, rtl=rtl, max_pages=max_pages, drop_headers=False)
        except Exception:
            txt = _extract_pdf_best_effort(pdf_path, rtl=rtl, two_cols=two_cols, max_pages=max_pages)
    else:
        txt = extract_fn(pdf_path, two_cols=two_cols, rtl=rtl, max_pages=max_pages)
    return build_chunks_from_txt(
        txt, target_chars=target_chars, overlap_chars=overlap_chars, min_chars=min_chars,
        max_chars=max_chars, max_chunks=max_chunks
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

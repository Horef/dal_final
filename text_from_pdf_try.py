# text_from_pdf_try.py
"""
Robust PDF → text extractor for mixed layouts (Hebrew-friendly).
- Auto-detects 1..4 vertical columns per page (no fixed halfway cut).
- RTL-aware ordering (right-to-left columns & word order).
- Table-ish lines: inserts '\t' when intra-line gaps are unusually large.
"""

import os
import statistics
import fitz
from tqdm import tqdm


def extract_text_from_pdf(pdf_path, two_cols=False, rtl=False, max_pages=None):
    """
    Extract text from PDF, robust to mixed per-page layouts (columns/tables).
    Args:
        pdf_path (str): path to PDF
        two_cols (bool): kept for backward-compat; if True we still auto-detect
                         columns but bias toward 2 if ambiguous.
        rtl (bool): Hebrew/Arabic reading order (True = RTL)
        max_pages (int|None): process up to N pages
    Returns:
        str: extracted text (UTF-8)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    doc = fitz.open(pdf_path)
    out_pages = []

    for pno in tqdm(range(len(doc)), desc="Extracting text from PDF", unit="page"):
        if max_pages is not None and pno >= max_pages:
            break
        page = doc.load_page(pno)
        page_text = _extract_page_text(page, rtl=rtl, prefer_two_cols=two_cols)
        out_pages.append(f"=== Page {pno+1} ===\n{page_text}".rstrip())

    doc.close()
    return "\n\n".join(out_pages) + "\n"


# -------------------------- internals -------------------------- #

def _extract_page_text(page, rtl=False, prefer_two_cols=False):
    # Words: (x0, y0, x1, y1, "text", block_no, line_no, word_no)
    words = page.get_text("words")
    if not words:  # fallbacks
        txt = page.get_text()
        return txt[::-1] if rtl else txt

    W = page.rect.width
    # Build histogram of word mid-x to find vertical gutters dynamically
    mids = [0.5 * (w[0] + w[2]) for w in words]
    cols = _detect_columns(mids, W, prefer_two_cols=prefer_two_cols)

    # Assign words to columns by midpoint
    col_words = [[] for _ in cols]
    for w in words:
        mid = 0.5 * (w[0] + w[2])
        idx = _which_interval(mid, cols)
        if idx is None:
            # out-of-bounds (rare); stuff into nearest column
            idx = _nearest_interval(mid, cols)
        col_words[idx].append(w)

    # Column reading order: RTL -> rightmost first, LTR -> leftmost first
    col_order = range(len(cols)-1, -1, -1) if rtl else range(len(cols))

    # Build text per column, keeping local reading order
    col_texts = []
    for ci in col_order:
        lines = _cluster_lines(col_words[ci])
        line_texts = []
        for line in lines:
            line_texts.append(_join_line_words(line, rtl=rtl))
        # remove extraneous blank runs
        col_texts.append(_squash_blank_lines("\n".join(line_texts)))

    # Join columns with a blank line gap to mark flow
    return "\n\n".join([t for t in col_texts if t.strip()])


def _detect_columns(mids, page_width, prefer_two_cols=False):
    """
    Returns a list of x-intervals [(xL,xR), ...] covering 1..4 columns.
    Heuristic: find "valleys" (low density) in a 1D histogram of word x-midpoints.
    """
    if not mids:
        return [(0.0, page_width)]

    bins = max(40, min(120, int(page_width // 12)))  # scale with page width
    # simple histogram
    min_x, max_x = 0.0, page_width
    bw = (max_x - min_x) / bins
    counts = [0] * bins
    for m in mids:
        j = int((m - min_x) / bw)
        if 0 <= j < bins:
            counts[j] += 1

    # valley threshold and gutter width
    peak = max(counts) if counts else 1
    thresh = max(1, int(peak * 0.15))  # "low density"
    min_gutter_pts = max(8.0, page_width * 0.015)  # ~≥8pt or 1.5% width

    # collect contiguous low regions as gutters
    gutters = []
    in_low = False
    start = 0
    for i, c in enumerate(counts):
        if c <= thresh and not in_low:
            in_low = True
            start = i
        elif c > thresh and in_low:
            in_low = False
            end = i - 1
            # convert to physical width
            gx0 = min_x + start * bw
            gx1 = min_x + (end + 1) * bw
            if (gx1 - gx0) >= min_gutter_pts:
                gutters.append((gx0, gx1))
    if in_low:
        gx0 = min_x + start * bw
        gx1 = min_x + bins * bw
        if (gx1 - gx0) >= min_gutter_pts:
            gutters.append((gx0, gx1))

    # Make columns from gutters; cap at 4 columns to avoid over-splitting
    splits = [0.0]
    for g in gutters:
        splits.append(0.5 * (g[0] + g[1]))  # center of gutter
    splits.append(page_width)
    splits = sorted(set(splits))
    intervals = [(splits[i], splits[i+1]) for i in range(len(splits)-1)]

    # Remove empty/ridiculously narrow intervals
    min_col_pts = max(40.0, page_width * 0.08)  # ≥8% of width
    intervals = [iv for iv in intervals if (iv[1] - iv[0]) >= min_col_pts]
    if not intervals:
        intervals = [(0.0, page_width)]

    # If too many columns, merge closest neighbors until ≤4
    while len(intervals) > 4:
        # merge smallest width column into neighbor
        widths = [iv[1] - iv[0] for iv in intervals]
        k = widths.index(min(widths))
        if k == 0:
            merged = (intervals[0][0], intervals[1][1])
            intervals = [merged] + intervals[2:]
        elif k == len(intervals)-1:
            merged = (intervals[-2][0], intervals[-1][1])
            intervals = intervals[:-2] + [merged]
        else:
            # merge with narrower neighbor
            left_w = intervals[k][1] - intervals[k-1][0]
            right_w = intervals[k+1][1] - intervals[k][0]
            if left_w <= right_w:
                merged = (intervals[k-1][0], intervals[k][1])
                intervals = intervals[:k-1] + [merged] + intervals[k+1:]
            else:
                merged = (intervals[k][0], intervals[k+1][1])
                intervals = intervals[:k] + [merged] + intervals[k+2:]

    # If caller asked for two columns and we ended with 1 or 3, try to bias to 2
    if prefer_two_cols and len(intervals) != 2:
        # if 1: split down the middle; if 3: merge the two closest
        if len(intervals) == 1:
            a, b = intervals[0]
            mid = 0.5 * (a + b)
            intervals = [(a, mid), (mid, b)]
        elif len(intervals) >= 3:
            # merge two closest neighbors
            gaps = [intervals[i+1][0] - intervals[i][1] for i in range(len(intervals)-1)]
            # merge pair with smallest gap (most likely accidental split)
            i = gaps.index(min(gaps))
            merged = (intervals[i][0], intervals[i+1][1])
            intervals = intervals[:i] + [merged] + intervals[i+2:]

    return intervals


def _which_interval(x, intervals):
    for i, (a, b) in enumerate(intervals):
        if a <= x <= b:
            return i
    return None


def _nearest_interval(x, intervals):
    # return index of interval with nearest center to x
    best_i, best_d = 0, float("inf")
    for i, (a, b) in enumerate(intervals):
        c = 0.5 * (a + b)
        d = abs(x - c)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def _cluster_lines(words):
    """
    Group words to lines by y proximity, then return list of line-word lists.
    """
    if not words:
        return []

    # Sort by top y
    words = sorted(words, key=lambda w: (w[1], w[0]))
    heights = [(w[3] - w[1]) for w in words]
    med_h = statistics.median(heights) if heights else 8.0
    y_tol = max(2.0, med_h * 0.6)

    lines = []
    cur = []
    cur_y = None

    for w in words:
        y = w[1]
        if cur_y is None:
            cur = [w]
            cur_y = y
            continue
        if abs(y - cur_y) <= y_tol:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
            cur_y = y
    if cur:
        lines.append(cur)

    return lines


def _join_line_words(line_words, rtl=False):
    """
    Inside one visual line, sort words in reading order and insert tabs when gaps are big.
    """
    if not line_words:
        return ""

    # RTL: sort by right edge descending; LTR: by left edge ascending
    if rtl:
        line_words = sorted(line_words, key=lambda w: (-w[2], w[1]))
    else:
        line_words = sorted(line_words, key=lambda w: (w[0], w[1]))

    # Compute gaps between consecutive words
    gaps = []
    for i in range(len(line_words) - 1):
        a = line_words[i]
        b = line_words[i + 1]
        gap = (a[0] - b[2]) if rtl else (b[0] - a[2])  # positive if space between
        gaps.append(max(0.0, gap))

    # Median gap as baseline; big gaps → tab
    med_gap = statistics.median(gaps) if gaps else 0.0
    tab_threshold = med_gap * 2.25  # tuned to keep table columns readable

    parts = [line_words[0][4]]
    for i in range(len(gaps)):
        sep = "\t" if gaps[i] > tab_threshold and tab_threshold > 0 else " "
        parts.append(sep + line_words[i + 1][4])

    text = "".join(parts)

    # small cleanup: collapse many spaces, keep tabs
    text = " ".join(text.replace("\u00a0", " ").split())
    return text


def _squash_blank_lines(s):
    lines = [ln.rstrip() for ln in s.splitlines()]
    out = []
    blank = False
    for ln in lines:
        if ln.strip():
            out.append(ln)
            blank = False
        else:
            if not blank:
                out.append("")
            blank = True
    return "\n".join(out).strip()

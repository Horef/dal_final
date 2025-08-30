import os, statistics, re
import fitz
from collections import Counter
from tqdm import tqdm

def extract_text_from_pdf(pdf_path, two_cols=False, rtl=False, max_pages=None, drop_headers=True):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    doc = fitz.open(pdf_path)

    # first pass: parse each page into ordered text lines (no headers stripped yet)
    page_lines = []
    for pno in tqdm(range(len(doc)), desc="Extracting text from PDF", unit="page"):
        if max_pages is not None and pno >= max_pages:
            break
        page = doc.load_page(pno)
        lines = _extract_page_lines(page, rtl=rtl, prefer_two_cols=two_cols)
        page_lines.append(lines)

    # optional: detect repeated header/footer lines and drop them
    if drop_headers and page_lines:
        top_cands = []
        bot_cands = []
        for lines in page_lines:
            if lines:
                top_cands.append(_normalize_header(lines[0]))
                if len(lines) > 1:  # guard
                    bot_cands.append(_normalize_header(lines[-1]))
        top_common = _common_repeated(top_cands, len(page_lines))
        bot_common = _common_repeated(bot_cands, len(page_lines))

        for i, lines in enumerate(page_lines):
            if lines:
                if _normalize_header(lines[0]) in top_common:
                    lines = lines[1:]
            if lines:
                if _normalize_header(lines[-1]) in bot_common:
                    lines = lines[:-1]
            page_lines[i] = lines

    # join pages
    out_pages = []
    for i, lines in enumerate(page_lines, 1):
        out_pages.append(f"=== Page {i} ===\n" + "\n".join(lines).rstrip())
    doc.close()
    return "\n\n".join(out_pages) + "\n"

# ---------------- internals ---------------- #

def _extract_page_lines(page, rtl=False, prefer_two_cols=False):
    words = page.get_text("words")
    if not words:
        txt = page.get_text()
        return [txt[::-1] if rtl else txt]

    W = page.rect.width
    mids = [0.5 * (w[0] + w[2]) for w in words]
    cols = _detect_columns(mids, W, prefer_two_cols=prefer_two_cols)

    # assign words to column buckets
    buckets = [[] for _ in cols]
    for w in words:
        mid = 0.5 * (w[0] + w[2])
        j = _which_interval(mid, cols)
        if j is None:
            j = _nearest_interval(mid, cols)
        buckets[j].append(w)

    # reading order of columns
    col_order = range(len(cols)-1, -1, -1) if rtl else range(len(cols))
    all_lines = []
    for ci in col_order:
        lines = _cluster_lines(buckets[ci])
        for line in lines:
            joined = _join_line_words(line, rtl=rtl)
            joined = _dehyphenate(joined)
            if joined.strip():
                all_lines.append(joined)
            else:
                # keep a single blank between blocks
                if len(all_lines) and all_lines[-1] != "":
                    all_lines.append("")
    # squash multiple blanks
    return _squash_blank_lines_list(all_lines)

def _detect_columns(mids, page_width, prefer_two_cols=False):
    if not mids:
        return [(0.0, page_width)]
    bins = max(40, min(120, int(page_width // 12)))
    bw = page_width / bins
    counts = [0]*bins
    for m in mids:
        j = int(min(bins-1, max(0, m / bw)))
        counts[j] += 1
    peak = max(counts) if counts else 1
    thresh = max(1, int(peak*0.15))
    min_gutter_pts = max(8.0, page_width*0.015)

    gutters, in_low, start = [], False, 0
    for i,c in enumerate(counts):
        if c <= thresh and not in_low:
            in_low, start = True, i
        elif c > thresh and in_low:
            in_low = False
            gx0, gx1 = start*bw, i*bw
            if (gx1-gx0) >= min_gutter_pts:
                gutters.append((gx0, gx1))
    if in_low:
        gx0, gx1 = start*bw, bins*bw
        if (gx1-gx0) >= min_gutter_pts:
            gutters.append((gx0, gx1))

    splits = [0.0] + [0.5*(g[0]+g[1]) for g in gutters] + [page_width]
    splits = sorted(set(splits))
    intervals = [(splits[i], splits[i+1]) for i in range(len(splits)-1)]
    min_col_pts = max(40.0, page_width*0.08)
    intervals = [iv for iv in intervals if (iv[1]-iv[0]) >= min_col_pts] or [(0.0, page_width)]
    while len(intervals) > 4:  # merge smallest
        k = min(range(len(intervals)), key=lambda i: intervals[i][1]-intervals[i][0])
        if k == 0:
            intervals = [(intervals[0][0], intervals[1][1])] + intervals[2:]
        elif k == len(intervals)-1:
            intervals = intervals[:-2] + [(intervals[-2][0], intervals[-1][1])]
        else:
            left = (intervals[k-1][0], intervals[k][1])
            right = (intervals[k][0], intervals[k+1][1])
            intervals = intervals[:k-1] + ([left] if (left[1]-left[0]) <= (right[1]-right[0]) else [right]) + intervals[k+2:]
    if prefer_two_cols and len(intervals) in (1,3,4):
        if len(intervals) == 1:
            a,b = intervals[0]; mid = 0.5*(a+b); intervals = [(a,mid),(mid,b)]
        else:
            # merge to 2 by joining closest neighbors
            while len(intervals) > 2:
                gaps = [intervals[i+1][0]-intervals[i][1] for i in range(len(intervals)-1)]
                i = gaps.index(min(gaps))
                intervals = intervals[:i] + [(intervals[i][0], intervals[i+1][1])] + intervals[i+2:]
    return intervals

def _which_interval(x, intervals):
    for i,(a,b) in enumerate(intervals):
        if a <= x <= b:
            return i
    return None

def _nearest_interval(x, intervals):
    return min(range(len(intervals)), key=lambda i: abs(x - 0.5*(intervals[i][0]+intervals[i][1])))

def _cluster_lines(words):
    if not words:
        return []
    words = sorted(words, key=lambda w: (w[1], w[0]))
    heights = [(w[3]-w[1]) for w in words]
    med_h = statistics.median(heights) if heights else 8.0
    y_tol = max(2.0, med_h*0.6)

    lines, cur, cur_y = [], [], None
    for w in words:
        y = w[1]
        if cur_y is None or abs(y-cur_y) <= y_tol:
            cur.append(w); cur_y = y if cur_y is None else cur_y
        else:
            lines.append(cur); cur = [w]; cur_y = y
    if cur: lines.append(cur)
    return lines

def _join_line_words(line_words, rtl=False):
    if not line_words:
        return ""
    # RTL → sort by right edge desc; LTR → left edge asc
    if rtl:
        line_words = sorted(line_words, key=lambda w: (-w[2], w[1]))
    else:
        line_words = sorted(line_words, key=lambda w: (w[0], w[1]))

    # compute character-width baseline
    widths = [(w[2]-w[0]) / max(1, len(w[4])) for w in line_words if w[4].strip()]
    char_w = statistics.median(widths) if widths else 4.0

    gaps = []
    for i in range(len(line_words)-1):
        a, b = line_words[i], line_words[i+1]
        gap = (a[0]-b[2]) if rtl else (b[0]-a[2])
        gaps.append(max(0.0, gap))

    # big gaps → tab; threshold relative to char width
    med_gap = statistics.median(gaps) if gaps else 0.0
    tab_thr = max(2.5*char_w, med_gap*2.0)

    parts = [line_words[0][4].replace("\u00a0", " ")]
    for i, g in enumerate(gaps):
        sep = "\t" if g > tab_thr else " "
        parts.append(sep + line_words[i+1][4].replace("\u00a0", " "))
    text = "".join(parts)
    text = re.sub(r"[^\S\t]+", " ", text).strip()  # collapse spaces but KEEP tabs
    return text

def _dehyphenate(s):
    # join hyphenated line-break leftovers ("-"/"־")
    return re.sub(r"\s*[-־]\s+", "-", s)

def _squash_blank_lines_list(lines):
    out, blank = [], False
    for ln in lines:
        if ln.strip():
            out.append(ln); blank = False
        else:
            if not blank:
                out.append(""); blank = True
    # trim leading/trailing blank
    while out and out[0] == "": out = out[1:]
    while out and out[-1] == "": out = out[:-1]
    return out

def _normalize_header(s):
    # remove digits and excessive spaces for robust matching
    return re.sub(r"\s+", " ", re.sub(r"\d", "", s)).strip()

def _common_repeated(cands, n_pages):
    cnt = Counter([c for c in cands if c])
    # treat as header/footer if it appears on ≥ 50% pages and ≥3 pages
    return {s for s,c in cnt.items() if c >= max(3, int(0.5*n_pages))}

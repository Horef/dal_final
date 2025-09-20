# main.py
import os
import argparse
import shutil

from text_from_pdf import extract_text_from_pdf
from data_splitter_en import (
    build_chunks_from_txt,
    write_chunks_jsonl,
    write_chunks_txt,
)

def clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def find_pdfs(root: str):
    pdfs = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dp, fn))
    return sorted(pdfs)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    ap = argparse.ArgumentParser("MiniRAG English splitter runner")
    ap.add_argument("--raw-data", default="Unprocessed Data",
                    help="Folder with input PDFs")
    ap.add_argument("--processed-data", default="Processed Data",
                    help="Folder to save extracted TXT and JSONL manifest")
    ap.add_argument("--dataset", default="MiniRAG/dataset/Technion/data",
                    help="Folder where per-chunk .txt files will be written (cleaned at start)")
    ap.add_argument("--two-cols", action="store_true", default=True,
                    help="Treat pages as two columns (left→right)")
    ap.add_argument("--rtl", action="store_true", default=False,
                    help="Right-to-left reading order (False for English)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Limit number of pages (debug)")
    # Chunking knobs
    ap.add_argument("--target", type=int, default=500,
                    help="Target chars per chunk (soft)")
    ap.add_argument("--max", dest="max_chars", type=int, default=800,
                    help="Hard cap chars per chunk")
    ap.add_argument("--min", dest="min_chars", type=int, default=150,
                    help="Minimum chars to flush a chunk")
    ap.add_argument("--overlap", type=int, default=120,
                    help="Overlap chars between chunks (clamped to 25% of target)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Upper bound on number of chunks (adjacent merges)")
    args = ap.parse_args()

    raw_data_path = args.raw_data
    processed_data_path = args.processed_data
    dataset_path = args.dataset

    # Ensure parent folders exist
    ensure_dir(processed_data_path)
    ensure_dir(os.path.dirname(dataset_path) or ".")

    # CLEAN dataset output
    print(f"Cleaning dataset output folder: {dataset_path}")
    clean_dir(dataset_path)

    # Process PDFs (recursive)
    pdf_files = find_pdfs(raw_data_path)
    print(f"Found {len(pdf_files)} PDF file(s) under '{raw_data_path}'.")

    for pdf_path in pdf_files:
        base_file = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            # Extract text (two-column, English LTR)
            text = extract_text_from_pdf(
                pdf_path,
                two_cols=args.two_cols,
                rtl=args.rtl,
                max_pages=args.max_pages,
                drop_headers=False,  # SAFER: do not risk deleting real body lines
            )

            # Save extracted text
            txt_out = os.path.join(processed_data_path, f"{base_file}.txt")
            with open(txt_out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[{base_file}] Saved text → {txt_out}")

            # Build chunks (new splitter: safe overlap + no tiny remnant chunks)
            chunks = build_chunks_from_txt(
                text,
                target_chars=args.target,
                overlap_chars=args.overlap,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                keep_table_as_whole=True,
                max_chunks=args.max_chunks,
            )
            print(f"[{base_file}] Built {len(chunks)} chunks")

            # Save JSONL in Processed Data
            jsonl_out = os.path.join(processed_data_path, f"{base_file}.chunks.jsonl")
            write_chunks_jsonl(chunks, jsonl_out)
            print(f"[{base_file}] Saved JSONL → {jsonl_out}")

            # Save per-chunk .txt files in dataset
            doc_outdir = os.path.join(dataset_path, base_file)
            ensure_dir(doc_outdir)
            write_chunks_txt(chunks, doc_outdir)
            print(f"[{base_file}] Wrote {len(chunks)} chunk text files → {doc_outdir}")

        except Exception as e:
            print(f"[{base_file}] ERROR: {e}")

    print("Done.")

if __name__ == "__main__":
    main()

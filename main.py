import os
import argparse
import shutil

from text_from_pdf import extract_text_from_pdf
from data_splitter_en import (
    build_chunks_from_txt,
    write_chunks_jsonl,
    write_chunks_txt,
    write_chunks,           # cleans a folder then writes JSONL + per-chunk .txt
)

def find_pdfs(root):
    return [f for f in os.listdir(root) if f.lower().endswith(".pdf")]

def main():
    ap = argparse.ArgumentParser("MiniRAG English splitter runner")
    ap.add_argument("--raw-data", default="Unprocessed Data",
                    help="Folder with input PDFs")
    ap.add_argument("--processed-data", default="Processed Data",
                    help="Folder to save extracted TXT and JSONL manifests")
    ap.add_argument("--dataset", default="MiniRAG/dataset/Technion/data",
                    help="Folder where per-chunk .txt files will be written (cleaned)")
    ap.add_argument("--clean-dataset", action="store_true", default=True,
                    help="Clean the dataset folder before writing")
    ap.add_argument("--two-cols", action="store_true", default=True,
                    help="Treat pages as two columns (left→right)")
    ap.add_argument("--rtl", action="store_true", default=False,
                    help="Right-to-left reading order (should be False for English)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Limit number of pages (debug)")
    # Chunking knobs
    ap.add_argument("--target", type=int, default=1100,
                    help="Target characters per chunk (soft)")
    ap.add_argument("--max", dest="max_chars", type=int, default=2200,
                    help="Hard cap on characters per chunk")
    ap.add_argument("--min", dest="min_chars", type=int, default=200,
                    help="Minimum characters to flush a chunk")
    ap.add_argument("--overlap", type=int, default=150,
                    help="Overlap characters between adjacent chunks")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Upper bound on number of chunks (merges adjacent)")
    args = ap.parse_args()

    raw_data_path = args.raw_data
    processed_data_path = args.processed_data
    dataset_path = args.dataset

    os.makedirs(processed_data_path, exist_ok=True)

    pdf_files = find_pdfs(raw_data_path)
    print(f"Found {len(pdf_files)} PDF file(s) in '{raw_data_path}'.")

    # Clean the dataset root if requested (will also clean per-doc subfolders later)
    if args.clean_dataset and os.path.exists(dataset_path):
        print(f"Cleaning dataset folder: {dataset_path}")
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)

    print("Extracting text and building chunks...")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(raw_data_path, pdf_file)
        base = os.path.splitext(pdf_file)[0]
        try:
            # 1) Extract text (two-column, English LTR)
            text = extract_text_from_pdf(
                pdf_path,
                two_cols=args.two_cols,
                rtl=args.rtl,
                max_pages=args.max_pages,
                drop_headers=True,  # your extractor will remove repeating headers/footers
            )

            # 2) Save extracted text for inspection
            txt_out = os.path.join(processed_data_path, f"{base}.txt")
            with open(txt_out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[{base}] Saved text → {txt_out}")

            # 3) Build chunks
            chunks = build_chunks_from_txt(
                text,
                target_chars=args.target,
                overlap_chars=args.overlap,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                keep_table_as_whole=True,  # harmless if there are no tables
                max_chunks=args.max_chunks,
            )
            print(f"[{base}] Built {len(chunks)} chunks")

            # 4) Save JSONL manifest (Processed Data)
            jsonl_out = os.path.join(processed_data_path, f"{base}.chunks.jsonl")
            write_chunks_jsonl(chunks, jsonl_out)
            print(f"[{base}] Saved JSONL → {jsonl_out}")

            # 5) Save per-chunk .txt files into dataset path (CLEAN subfolder first)
            doc_outdir = os.path.join(dataset_path, base)
            write_chunks(chunks, doc_outdir, basename=base, clean=True)
            # (write_chunks cleans doc_outdir, writes {base}.jsonl + chunk_XXXX.txt files)
            print(f"[{base}] Wrote {len(chunks)} chunk text files → {doc_outdir}")

        except Exception as e:
            print(f"[{base}] ERROR: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
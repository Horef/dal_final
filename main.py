import os
import argparse
import shutil

from text_from_pdf import extract_text_from_pdf
from data_splitter_en import (
    build_chunks_from_txt,
    write_chunks_jsonl,
    write_chunks_txt,
)

def find_pdfs(root):
    return [f for f in os.listdir(root) if f.lower().endswith(".pdf")]

def main():
    ap = argparse.ArgumentParser("MiniRAG English splitter runner")
    ap.add_argument("--raw-data", default="Unprocessed Data",
                    help="Folder with input PDFs")
    ap.add_argument("--processed-data", default="Processed Data",
                    help="Folder to save extracted TXT and JSONL manifest")
    ap.add_argument("--dataset", default="MiniRAG/dataset/Technion/data",
                    help="Folder where per-chunk .txt files will be written")
    ap.add_argument("--clean-dataset", action="store_true", default=True,
                    help="Clean dataset folder before writing each doc")
    ap.add_argument("--two-cols", action="store_true", default=True,
                    help="Treat pages as two columns (left→right)")
    ap.add_argument("--rtl", action="store_true", default=False,
                    help="Right-to-left reading order (False for English)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Limit number of pages (debug)")
    # Chunking knobs
    ap.add_argument("--target", type=int, default=1100,
                    help="Target chars per chunk (soft)")
    ap.add_argument("--max", dest="max_chars", type=int, default=2200,
                    help="Hard cap chars per chunk")
    ap.add_argument("--min", dest="min_chars", type=int, default=200,
                    help="Minimum chars to flush a chunk")
    ap.add_argument("--overlap", type=int, default=150,
                    help="Overlap chars between chunks")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Upper bound on number of chunks (adjacent merges)")
    args = ap.parse_args()

    raw_data_path = args.raw_data
    processed_data_path = args.processed_data
    dataset_path = args.dataset

    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(dataset_path, exist_ok=True)

    pdf_files = find_pdfs(raw_data_path)
    print(f"Found {len(pdf_files)} PDF file(s) in '{raw_data_path}'.")

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
                drop_headers=True,
            )

            # 2) Save extracted text for inspection (Processed Data)
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
                keep_table_as_whole=True,
                max_chunks=args.max_chunks,
            )
            print(f"[{base}] Built {len(chunks)} chunks")

            # 4) Save JSONL manifest ONLY in Processed Data
            jsonl_out = os.path.join(processed_data_path, f"{base}.chunks.jsonl")
            write_chunks_jsonl(chunks, jsonl_out)
            print(f"[{base}] Saved JSONL → {jsonl_out}")

            # 5) Save per-chunk .txt files ONLY in dataset path (clean subfolder first)
            doc_outdir = os.path.join(dataset_path, base)
            if args.clean_dataset and os.path.isdir(doc_outdir):
                shutil.rmtree(doc_outdir)
            os.makedirs(doc_outdir, exist_ok=True)

            # Only write TXT chunk files (NO JSON here)
            write_chunks_txt(chunks, doc_outdir)
            print(f"[{base}] Wrote {len(chunks)} chunk text files → {doc_outdir}")

        except Exception as e:
            print(f"[{base}] ERROR: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
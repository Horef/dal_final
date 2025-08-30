
import os
from text_from_pdf import extract_text_from_pdf
from data_splitter import build_chunks_from_txt, write_chunks_jsonl

if __name__ == "__main__":
    raw_data_path = 'Unprocessed Data'
    processed_data_path = 'Processed Data'
    os.makedirs(processed_data_path, exist_ok=True)

    pdf_files = [f for f in os.listdir(raw_data_path) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in '{raw_data_path}'.")

    print("Extracting text from PDF files...")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(raw_data_path, pdf_file)
        base = os.path.splitext(pdf_file)[0]
        try:
            text = extract_text_from_pdf(pdf_path, rtl=True, two_cols=True, max_pages=None)

            # save the extracted text
            txt_out = os.path.join(processed_data_path, f"{base}.txt")
            with open(txt_out, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Saved text → {txt_out}")

            # build chunks for miniRAG
            chunks = build_chunks_from_txt(
                text,
                target_chars=1200,
                overlap_chars=150,
                min_chars=200,
                max_chars=2200,
                keep_table_as_whole=True,
            )
            jsonl_out = os.path.join(processed_data_path, f"{base}.chunks.jsonl")
            write_chunks_jsonl(chunks, jsonl_out)
            print(f"Saved {len(chunks)} chunks → {jsonl_out}")

        except Exception as e:
            print(f"An error occurred while processing {pdf_file}: {e}")

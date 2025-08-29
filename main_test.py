import os
#from text_from_pdf import extract_text_from_pdf
from text_from_pdf_try import extract_text_from_pdf

if __name__ == "__main__":
    # finding all files in the Unprocessed Data folder
    raw_data_path = 'Unprocessed Data'
    processed_data_path = 'Processed Data'
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    pdf_files = [f for f in os.listdir(raw_data_path) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in '{raw_data_path}'.")

    print("Extracting text from PDF files...")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(raw_data_path, pdf_file)
        try:
            text = extract_text_from_pdf(pdf_path, rtl=True, two_cols=True, max_pages=None)
            # save the extracted text to a file
            output_file = os.path.join(processed_data_path, f"{os.path.splitext(pdf_file)[0]}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Successfully processed {pdf_file} and saved to {output_file}.")
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred while processing {pdf_file}: {e}")
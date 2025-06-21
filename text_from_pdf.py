"""
This is a helper script to extract text from a PDF file.
"""

import os
import fitz
from tqdm import tqdm

def extract_text_from_pdf(pdf_path, two_cols=False, rtl=False, max_pages=None):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.
        two_cols (bool, optional): If True, assumes the PDF has two columns and extracts text accordingly.
                                    Defaults to False.
        rtl (bool, optional): If True, assumes the text is in right-to-left format.
        max_pages (int, optional): The maximum number of pages to extract text from. 
                                    If None, all pages will be processed.
    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        
    Returns:
        str: The extracted text from the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    if two_cols:
        text_data = extract_two_column_text(pdf_path, max_pages=max_pages)
        # removing whitespace from the text
        text_data = [{"page": entry["page"], "column1": entry["column1"].strip(), "column2": entry["column2"].strip()} for entry in text_data]

        # if RTL, reverse the text, and unify the columns in correct order
        if rtl:
            # Unify the output text to a single string, order to columns w.r.t. rtl
            text_data = "\n".join([f"Page {entry['page']}:\nColumn 1:\n{entry['column2']}\nColumn 2:\n{entry['column1']}" for entry in text_data])
        else:
            # Unify the output text to a single string, order to columns
            text_data = "\n".join([f"Page {entry['page']}:\nColumn 1:\n{entry['column1']}\nColumn 2:\n{entry['column2']}" for entry in text_data])
    else:
        text_data = extract_singe_column_text(pdf_path, max_pages=max_pages)
        if rtl:
            # If the text is in RTL format, you might need to reverse the text
            text_data = text_data[::-1]
        
    return text_data

def extract_singe_column_text(pdf_path, max_pages=None):
    doc = fitz.open(pdf_path)
    text = ''

    # Iterate through pages and extract text
    for page_num in tqdm(range(len(doc)), desc="Extracting text from PDF", unit="page"):
        if max_pages is not None and page_num >= max_pages:
            break
        page = doc.load_page(page_num)
        text += page.get_text() + "\n"

    doc.close()
    return text

def extract_two_column_text(pdf_path, max_pages=None):
    doc = fitz.open(pdf_path)
    text_data = []

    for page_num in tqdm(range(len(doc)), desc="Extracting text from PDF", unit="page"):
        if max_pages is not None and page_num >= max_pages:
            break
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks") # Extract text blocks with coordinates

        # Assuming two columns of roughly equal width, you can sort blocks by x-coordinate
        # and then process them column by column.
        column1_blocks = sorted([b for b in blocks if b[0] < page.rect.width / 2], key=lambda x: x[1]) # Sort by y-coordinate
        column2_blocks = sorted([b for b in blocks if b[0] >= page.rect.width / 2], key=lambda x: x[1])

        column1_text = "\n".join([b[4] for b in column1_blocks])
        column2_text = "\n".join([b[4] for b in column2_blocks])

        text_data.append({"page": page_num + 1, "column1": column1_text, "column2": column2_text})
    
    doc.close()
    return text_data

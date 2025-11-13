# Uni-Assistant: MiniRAG-based Technion Course Assistant

Uni-Assistant is a specialized Retrieval-Augmented Generation (RAG) system for Technion students.  
Its goal is to provide **a lightweight “in your phone” digital assistant** that can answer questions about:

- Courses and prerequisites  
- Study programs and degree requirements  
- Other Technion-related academic information  

The system is built on top of the **MiniRAG** architecture and adds a full evaluation pipeline using **RAGAS**.

---

## Project Overview

At a high level, Uni-Assistant:

1. **Ingests Technion documents** (course syllabi, regulations, etc.) from PDFs.  
2. **Chunks** them into segments tailored for English Technion content.  
3. **Indexes** the chunks using a MiniRAG-style heterogeneous graph and vector store.  
4. **Answers student questions**.  
5. **Evaluates** the quality of the answers using RAGAS metrics.

> **Note:** This project currently focuses on the **English** pipeline.  
> Hebrew-specific modules (e.g., `data_splitter_hebrew.py`) exist in the repo but are not used.

---

## Repository Structure (high-level)

- `Unprocessed Data/` – Raw Technion documents (mostly PDFs).
- `Processed Data/` – Cleaned & chunked documents ready for indexing.
- `MiniRAG/` – Adapted MiniRAG code and scripts for indexing, QA, and evaluation.
- `main.py` – Orchestration script for PDF → text → chunks.
- `data_splitter_en.py` – English-specific splitter.
- `text_from_pdf.py` – PDF to plain-text extraction.
- `data_splitter_hebrew.py` – Hebrew splitter (currently unused).
- `*.py` under `MiniRAG/` – Indexing, QA, evaluation, and web app (see pipeline below).

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Horef/dal_final.git
cd dal_final



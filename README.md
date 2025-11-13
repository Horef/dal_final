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


### 2. Create and activate a virtual environment (recommended)

    python -m venv .venv
    source .venv/bin/activate      # On Windows: .venv\Scripts\activate

### 3. Install dependencies

If you have a requirements file (e.g., from MiniRAG or this project):

    pip install -r MiniRAG/requirements.txt

Additionally, make sure you have:

    pip install ragas datasets evaluate

(Adjust this section according to your final environment / requirements file.)

---

## Data Preparation and Chunking (Step 1)

**Goal:** Convert Technion PDFs into clean, semantically meaningful chunks for RAG.

1. Place all raw PDF files inside `Unprocessed Data/`.
2. Run the main pipeline (which internally uses `text_from_pdf.py` and `data_splitter_en.py`):

       python main.py
       # or, if arguments are required:
       python main.py --input_dir "Unprocessed Data" --output_dir "Processed Data"

This step:

- Extracts text from PDFs.
- Applies **English-friendly chunking** via `data_splitter_en.py`.
- Writes the resulting chunks into `Processed Data/` in the format expected by MiniRAG.

> The Hebrew splitter (`data_splitter_hebrew.py`) is present but not part of the current pipeline.

---

## MiniRAG + RAGAS Pipeline

Once you have chunked data in `Processed Data/`, the core pipeline consists of **five main scripts** followed by the web interface.

### Step 0 – Build the Index (`step_0_index.py`)

**Script:** `MiniRAG/step_0_index.py`

Responsibilities:

- Load processed chunks from `Processed Data/`.
- Build a **MiniRAG heterogeneous graph** combining:
  - Text chunks
  - Named entities / metadata
- Create / update the vector index used for retrieval.
- Persist the index and graph to disk (e.g., under a dedicated folder).

Example run:

    cd MiniRAG
    python step_0_index.py

Check the script’s `--help` for index paths and configuration options.

---

### Step 1 – Question Answering Experiments (`step_1_QA.py`)

**Script:** `MiniRAG/step_1_QA.py`

Responsibilities:

- Load the MiniRAG index/graph from Step 0.
- Run QA over:
  - A predefined question set (e.g., Technion course/prerequisite questions), or
  - User-provided questions.
- Store:
  - Question
  - Retrieved context
  - Model answer
  - (Optionally) reference/ground truth answer

Outputs from this step are later used as input to the evaluation pipeline.

Example:

    python step_1_QA.py
    # Add arguments for question files / index location if needed.

---

### Step 2 – Bridge MiniRAG → RAGAS (`step_2_evaluation.py`)

**Script:** `MiniRAG/step_2_evaluation.py`

This step is the **transition layer** between your adapted MiniRAG outputs and RAGAS.

Responsibilities:

- Take the QA logs from `step_1_QA.py` (questions, answers, contexts, references).
- Convert them into the **RAGAS-compatible dataset format** (e.g., a `datasets.Dataset` or structured JSON/CSV).
- Save the transformed dataset to disk, ready for metric calculation.

Example:

    python step_2_evaluation.py

---

### Step 3 – Compute RAGAS Metrics (`Step_3_calculate_metrics.py`)

**Script:** `MiniRAG/Step_3_calculate_metrics.py` (note the capital `S` in some setups)

Responsibilities:

- Load the dataset produced by `step_2_evaluation.py`.
- Run RAGAS metrics such as (depending on configuration):
  - Answer correctness
  - Faithfulness / factuality
  - Context relevance / recall
  - Answer completeness
- Produce an aggregated report (per question and/or global scores).

Example:

    python Step_3_calculate_metrics.py

The resulting metrics can be used to compare different:

- Indexing strategies  
- Chunking heuristics  
- Models or prompting templates  

---

## Running the Web Interface

**Script:** `MiniRAG/web_app_en.py` (English interface)

This script provides a simple **web front-end** where Technion students can ask questions about courses, prerequisites, and study programs.

From the project root:

    cd MiniRAG
    python web_app_en.py

Then open the URL printed in the terminal (often `http://127.0.0.1:8000` or `http://localhost:5000`, depending on the framework used) in your browser or mobile device.

---

## Typical End-to-End Workflow

1. **Prepare data**  
   Place PDFs in `Unprocessed Data/`.

2. **Chunk documents**  
   Run `main.py` (uses `data_splitter_en.py`) to create processed chunks in `Processed Data/`.

3. **Index with MiniRAG**  
   Run `step_0_index.py` to build the graph + vector index.

4. **Generate QA data**  
   Run `step_1_QA.py` to collect question–answer–context pairs.

5. **Convert to RAGAS format**  
   Run `step_2_evaluation.py`.

6. **Evaluate**  
   Run `Step_3_calculate_metrics.py` to compute RAGAS metrics.

7. **Serve the assistant**  
   Run `web_app_en.py` for the interactive Uni-Assistant web app.

---

## Acknowledgements

This project heavily builds on and adapts:

- **MiniRAG** – for the lightweight graph-based RAG framework.  
- **RAGAS** – for evaluation of RAG answer quality.

Please refer to their respective repositories and documentation for deeper technical details.

---

## Future Work / Ideas

Some potential extensions:

- Adding full **Hebrew** support (chunking + index + evaluation).  
- Supporting additional Technion data sources (FAQ pages, forums, regulations).  
- Deploying the assistant as:
  - A mobile-friendly PWA
  - An on-device application using small language models

---

Feel free to open issues or pull requests in this repo to improve the pipeline, add new evaluation strategies, or refine the Technion-specific heuristics.



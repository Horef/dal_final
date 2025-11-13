# Uni-Assistant: MiniRAG-based Technion Course Assistant

Uni-Assistant is a specialized Retrieval-Augmented Generation (RAG) system for Technion students.  
Its goal is to provide **a lightweight “in your phone” digital assistant** that can answer questions about:

- Regulations and procedures 
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
- `rag_env.yml` – Provided for ease of enviroment reproduction.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Horef/dal_final.git
cd dal_final


### 2. Create and activate a virtual environment (recommended)

    conda env create -f rag_env.yml
    conda activate rag_env




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
- Writes the resulting chunks into `Processed Data/` in in the format expected by MiniRAG.
- A dataset is created in MiniRAG/dataset/Technion/data (change --dataset parameter for a different behaviour).

> The Hebrew splitter (`data_splitter_hebrew.py`) is present but not part of the current pipeline.

---

## MiniRAG + RAGAS Pipeline

Once you have chunked data, the core pipeline consists of **five main scripts** followed by the web interface.

### Step 0 – Build the Index (`Step_0_index.py`)

**Script:** `MiniRAG/reproduce/Step_0_index.py`

Responsibilities:

- Load processed chunks from MiniRAG/dataset/Technion/data.
- Build a **MiniRAG heterogeneous graph** combining:
  - Text chunks
  - Named entities / metadata extracted via a Small Language Model
- Create / update the vector index used for retrieval.
- Persist the index and graph to disk (e.g., under a dedicated folder).

Example run:

    cd MiniRAG
    python reproduce/Step_0_index.py
---

### Step 1 – Question Answering Experiments (`Step_1_QA.py`)

**Script:** `MiniRAG/reproduce/Step_1_QA.py`

Responsibilities:

- Load the MiniRAG index/graph from Step 0.
    - currently, the successfull 40h long Step 0 run result is stored in MiniRAG/Step0_res.
    - In order to use a different dirrectory, change --workingdir parameter.
- Run QA over:
  - A predefined question set (the code expects to find a csv file at MiniRAG/Technion/qa/query_set.csv).
  - Provide an alternative by changing the --querypath parameter - the header of the csv must be 'Question', 'Golden Answer'.
- Store:
  - Question
  - Reference/ground truth answer
  - Retrieved  MiniRAG context
  - MiniRAG answer
  - Naive Context
  - Naive Answer
  

Outputs from this step are later used as input to the evaluation pipeline.
The output uses '@' as a delimiter. 

Example:

    python reproduce/Step_1_QA.py

---

### Step 2 – Bridge MiniRAG → RAGAS (`Step_2_evaluation.py`)

**Script:** `MiniRAG/reproduce/Step_2_evaluation.py`

This step is the **transition layer** between adapted MiniRAG outputs and RAGAS.

Responsibilities:

- Take the QA logs from `Step_1_QA.py` (questions, answers, contexts, references).
- Use an LLM (via OpenAI key) to evaluate faithfulness, answer relevancy, context recall, and context precision of each question-answer.
- Save the scores as csv files for acumilative metrics calculation.

**IMPORTANT:** You MUST set your OpenAI API key as an environment variable

Example:
    
    python reproduce/Step_2_evaluation.py

---

### Step 3 – Compute RAGAS Metrics (`Step_3_calculate_metrics.py`)

**Script:** `MiniRAG/reproduce/Step_3_calculate_metrics.py`

Responsibilities:

- Load the dataset produced by `Step_2_evaluation.py`.
- Calculate RAGAS metrics averages for MiniRAG / Naive RAG
- Produce an aggregated report.

Example:

    python reproduce/Step_3_calculate_metrics.py

 
---

## Running the Web Interface

**Script:** `MiniRAG/chat_interface_en/web_app_en.py` (English interface)

This script provides a simple **web front-end** where Technion students can ask questions about courses, prerequisites, and study programs.

From the project root:

    cd MiniRAG
    streamlit run chat_interface_en/web_app_en.py

---

## Typical End-to-End Workflow

1. **Prepare data**  
   Place PDFs in `Unprocessed Data/`.

2. **Chunk documents**  
   Run `main.py` (uses `data_splitter_en.py`) to create processed chunks in `Processed Data/`.

3. **Index with MiniRAG**  
   Run `Step_0_index.py` to build the graph + vector index.

4. **Generate QA data**  
   Run `Step_1_QA.py` to collect question–answer–context pairs.

5. **Convert to RAGAS format**  
   Run `Step_2_evaluation.py`.

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



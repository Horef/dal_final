#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step_0_index.py  —  MiniRAG indexer (HF-only, no OpenAI)

- Runs from the REPO ROOT:  python reproduce/Step_0_index.py [--args...]
- Uses local Hugging Face LLMs for completion (Phi-3.5-mini-instruct, Qwen2.5-3B, MiniCPM3-4B, GLM Edge 1.5B, Aya-23-8B)
- Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings (384-dim)

Args:
  --model        One of: PHI | aya | GLM | MiniCPM | qwen   (default: PHI)
  --outputpath   CSV to write logs (unused here but kept for compatibility)
  --workingdir   Vector DB and caches directory (default: ./LiHua-World)
  --datapath     Root folder to recursively index .txt files
  --querypath    Path to queries CSV (kept for compatibility)

Requirements (install once):
  pip install "transformers>=4.42" accelerate torch sentencepiece tiktoken
"""

import os
import sys
import argparse
import asyncio
from typing import List

# Ensure repo root is on PYTHONPATH when running from root with `python reproduce/...`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from minirag import MiniRAG
from minirag.llm import hf_embed
from minirag.utils import EmbeddingFunc


# ----------------------------
# Configuration & CLI parsing
# ----------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG (HF-only)")
    parser.add_argument("--model", type=str, default="PHI",
                        help="PHI | aya | GLM | MiniCPM | qwen")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument("--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv")
    return parser.parse_args()


args = get_args()

# Map CLI choice to HF model names (you can swap to any compatible instruct model)
if args.model == "PHI":
    HF_LLM = "microsoft/Phi-2"
elif args.model == "aya":
    HF_LLM = "CohereLabs/aya-23-8B"
elif args.model == "GLM":
    HF_LLM = "THUDM/glm-edge-1.5b-chat"
elif args.model == "MiniCPM":
    HF_LLM = "openbmb/MiniCPM3-4B"
elif args.model == "qwen":
    HF_LLM = "Qwen/Qwen2.5-3B-Instruct"  # strong multilingual (incl. Hebrew)
else:
    print("Invalid model name. Use: PHI | aya | GLM | MiniCPM | qwen")
    sys.exit(1)

WORKING_DIR = args.workingdir
DATA_PATH   = args.datapath
QUERY_PATH  = args.querypath  # kept for parity, not used below
OUTPUT_PATH = args.outputpath

print("USING LLM:", HF_LLM)
print("USING WORKING DIR:", WORKING_DIR)

os.makedirs(WORKING_DIR, exist_ok=True)

# ----------------------------
# Build a single HF text-gen pipeline (loads once)
# ----------------------------

_device_map = "auto"
_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Some small models lack pad/eos; we set sane fallbacks after loading tokenizer
_hf_tokenizer = AutoTokenizer.from_pretrained(HF_LLM, trust_remote_code=True)
_hf_model = AutoModelForCausalLM.from_pretrained(
    HF_LLM,
    torch_dtype=_dtype,
    device_map=_device_map,
    trust_remote_code=True,
)

# Fallback ids
if _hf_tokenizer.pad_token_id is None:
    # Prefer eos as pad if missing
    if _hf_tokenizer.eos_token_id is not None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token
    else:
        _hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        _hf_model.resize_token_embeddings(len(_hf_tokenizer))

_hf_pipe = pipeline(
    "text-generation",
    model=_hf_model,
    tokenizer=_hf_tokenizer,
    # device is handled via device_map inside model
)

async def hf_model_complete(prompt: str, **kwargs) -> str:
    """
    Async wrapper compatible with MiniRAG's awaited LLM interface.
    Generates short, deterministic-ish completions for indexing/entity tasks.
    """
    gen = _hf_pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=_hf_tokenizer.pad_token_id,
        eos_token_id=_hf_tokenizer.eos_token_id,
        truncation=False,  # let MiniRAG control prompt sizing if needed
    )
    text = gen[0]["generated_text"]
    # Strip the prompt prefix if the pipeline returns prompt+completion
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


# ----------------------------
# Build MiniRAG with HF-only stack
# ----------------------------

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,   # ← uses local HF model, no OpenAI
    llm_model_max_token_size=200,
    llm_model_name=HF_LLM,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,              # all-MiniLM-L6-v2 = 384
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)


# ----------------------------
# Utilities
# ----------------------------

def find_txt_files(root_path: str) -> List[str]:
    txt_files = []
    for root, _dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return sorted(txt_files)


# ----------------------------
# Main
# ----------------------------

def main():
    chunks = find_txt_files(DATA_PATH)
    total = len(chunks)
    if total == 0:
        print(f"No .txt files found under: {DATA_PATH}")
        return

    for idx, chunk_path in enumerate(chunks, start=1):
        print(f"{idx}/{total}  {chunk_path}")
        with open(chunk_path, "r", encoding="utf-8", errors="ignore") as f:
            rag.insert(f.read())   # ← fixed: no space, correct method name

    print("INFO: Document processing pipeline completed")
    print("INFO: If entity extraction is enabled in MiniRAG, it will run automatically.")


if __name__ == "__main__":
    main()

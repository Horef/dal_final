#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step_0_index.py  —  MiniRAG indexer (HF-only, no OpenAI)

- Runs from the REPO ROOT:  python reproduce/Step_0_index.py [--args...]
- Uses local Hugging Face LLMs for completion (bloomz-3.5-mini-instruct, Qwen2.5-3B, MiniCPM3-4B, GLM Edge 1.5B, Aya-23-8B)
- Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings (384-dim)

Args:
  --model        One of: bloomz | neo | bloom1 | GLM | MiniCPM | qwen | dictalm  (default: dictalm)
  --outputpath   CSV to write logs (unused here but kept for compatibility)
  --workingdir   Vector DB and caches directory (default: ./Technion)
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
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "8")

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

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG (HF-only)")
    parser.add_argument("--model", type=str, default="dictalm",
                        help="bloomz | neo | dictalm | bloom1 | GLM | MiniCPM | qwen")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./Technion")
    parser.add_argument("--datapath", type=str, default="./dataset/Technion/data/")
    parser.add_argument("--querypath", type=str, default="./dataset/Technion/qa/query_set.csv")
    parser.add_argument("--checkpoints", type=int, default=10, 
                        help="Number of checkpoints to save during indexing (default: 10)")
    parser.add_argument("--save", type=int, default=1, 
                        help="Whether to save the index after processing (1 = yes, 0 = no; default: 1)")
    return parser.parse_args()




args = get_args()

# Map CLI choice to HF model names (you can swap to any compatible instruct model)
if args.model == "bloomz":
    HF_LLM = "bigscience/bloomz-560m"          # instruction-tuned, multilingual NO HEBREW!

elif args.model == "dictalm":
    HF_LLM = "dicta-il/dictalm2.0-instruct-GGUF"

elif args.model == "dictalm_no_gguf":
    HF_LLM = "dicta-il/dictalm2.0-instruct-AWQ"

elif args.model == "neo":
    HF_LLM = "Norod78/hebrew-gpt_neo-small"

# elif args.model == "bloom1":
#     HF_LLM = "bigscience/bloom-1b1"
#
# elif args.model == "GLM":
#     HF_LLM = "THUDM/glm-edge-1.5b-chat"
#
# elif args.model == "MiniCPM":
#     HF_LLM = "openbmb/MiniCPM3-4B"
#
# elif args.model == "qwen":
#     HF_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
else:
    print("Invalid model name. Use: bloomz | neo | bloom1 | GLM | MiniCPM | qwen")
    sys.exit(1)


USE_GGUF = HF_LLM.lower().endswith("-gguf")



WORKING_DIR = args.workingdir
DATA_PATH   = args.datapath
QUERY_PATH  = args.querypath
OUTPUT_PATH = args.outputpath

print("USING LLM:", HF_LLM)
print("USING WORKING DIR:", WORKING_DIR)

os.makedirs(WORKING_DIR, exist_ok=True)

# ----------------------------
# Build a single HF text-gen pipeline (loads once)
# ----------------------------


_hf_tokenizer = AutoTokenizer.from_pretrained(
    HF_LLM,
    use_fast=False,
    trust_remote_code=True,
    padding_side="left",
)

_dtype = torch.float32
_hf_model = AutoModelForCausalLM.from_pretrained(
    HF_LLM,
    torch_dtype=_dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
_hf_model.to("cpu").eval()

if _hf_tokenizer.pad_token_id is None:
    if _hf_tokenizer.eos_token_id is not None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token
    else:
        _hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        _hf_model.resize_token_embeddings(len(_hf_tokenizer))



# Manual device placement (M60: CC 5.2; torch 1.13.1 works)
device_arg = -1

# if torch.cuda.is_available():
#     print("CUDA is available")
#     try:
#         maj, minr = torch.cuda.get_device_capability(0)
#         print(f"Detected GPU: {torch.cuda.get_device_name(0)} (capability {maj}.{minr})")
#
#         if (maj, minr) >= (5, 2):  # Tesla M60 is 5.2
#             _hf_model.to("cuda:0")
#             device_arg = 0
#             print("Model successfully moved to CUDA:0")
#         else:
#             print(f"GPU capability {maj}.{minr} is below required (5.2), using CPU instead")
#     except Exception as e:
#         print(f"Error while checking CUDA device: {e}")
#         device_arg = -1
# else:
#     print("CUDA is NOT available, running on CPU")

print(f"Final device_arg = {device_arg}")

_hf_pipe = pipeline(
    "text-generation",
    model=_hf_model,
    tokenizer=_hf_tokenizer,
    device=-1,
    return_full_text=False,
)

_HF_GEN_SEM = asyncio.Semaphore(1)
# --- async wrapper, now serialized + small hygiene ---
async def hf_model_complete(prompt: str, **kwargs) -> str:
    # serialized calls (even on CPU, avoids thread contention)
    async with _HF_GEN_SEM:
        gen = _hf_pipe(
            prompt,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=_hf_tokenizer.pad_token_id,
            eos_token_id=_hf_tokenizer.eos_token_id,
        )
        text = gen[0]["generated_text"]
        return text.strip()


# ----------------------------
# Build MiniRAG with HF-only stack
# ----------------------------

_EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
_EMB_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL)
_EMB_MODEL.to("cpu").eval()

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_max_token_size=200,
    llm_model_name=HF_LLM,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=_EMB_TOKENIZER,
            embed_model=_EMB_MODEL,
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
    ch_id = 0

    chunks = find_txt_files(DATA_PATH)
    total = len(chunks)
    if total == 0:
        print(f"No .txt files found under: {DATA_PATH}")
        return

    for idx, chunk_path in enumerate(chunks, start=1):
        print(f"{idx}/{total}  {chunk_path}")
        if args.checkpoints > 1 and (idx % (total // args.checkpoints) == 0 or idx == total):
            ch_id += 1
            if args.save:
                print(f"--- Saving index checkpoint #{ch_id} ---")
                # saving the checkpoint using pickle
                with open(os.path.join(WORKING_DIR, f'checkpoints/rag_ch_{ch_id}.pkl'), 'wb') as f:
                    pickle.dump(rag, f)
                with open(os.path.join(WORKING_DIR, 'checkpoints/rag_final.pkl'), 'wb') as f:
                    pickle.dump(rag, f)
                print(f"--- Checkpoint #{ch_id} saved ---")

        with open(chunk_path, "r", encoding="utf-8", errors="ignore") as f:
            rag.insert(f.read())

    if args.save:
        print("--- Saving final index ---")
        with open(os.path.join(WORKING_DIR, 'checkpoints/rag_final.pkl'), 'wb') as f:
            pickle.dump(rag, f)
        print("--- Final index saved ---")
    print("INFO: Document processing pipeline completed")
    print("INFO: If entity extraction is enabled in MiniRAG, it will run automatically.")


if __name__ == "__main__":
    main()

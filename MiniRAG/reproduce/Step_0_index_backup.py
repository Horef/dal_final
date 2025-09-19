#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step_0_index_backup.py  —  MiniRAG indexer (HF-only, no OpenAI)

- Runs from the REPO ROOT:  python reproduce/Step_0_index_backup.py [--args...]
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
#import pickle
import cloudpickle as pickle

os.environ.setdefault("OMP_NUM_THREADS", "12")

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

#EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EMBEDDING_MODEL = "dicta-il/dictabert"


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG (HF-only)")
    parser.add_argument("--model", type=str, default="dictalm",
                        help="Currently only dictalm works!")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./Technion")
    parser.add_argument("--datapath", type=str, default="./dataset/Technion/data/")
    parser.add_argument("--querypath", type=str, default="./dataset/Technion/qa/query_set_old.csv")
    parser.add_argument("--checkpoints", type=int, default=10, 
                        help="Number of checkpoints to save during indexing (default: 10)")
    parser.add_argument("--save", type=int, default=1, 
                        help="Whether to save the index after processing (1 = yes, 0 = no; default: 1)")
    return parser.parse_args()




args = get_args()

# Map CLI choice to HF model names (you can swap to any compatible instruct model)


if args.model == "dictalm":
    HF_LLM = "dicta-il/dictalm2.0-instruct-GGUF"

# elif args.model == "bloomz":
#     HF_LLM = "bigscience/bloomz-560m"          # instruction-tuned, multilingual NO HEBREW!
#
# elif args.model == "dictalm_no_gguf":
#     HF_LLM = "dicta-il/dictalm2.0-instruct"
#
# elif args.model == "neo":
#     HF_LLM = "Norod78/hebrew-gpt_neo-small"

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
    print("Currently only dictalm works!")
    sys.exit(1)


USE_GGUF = HF_LLM.lower().endswith("-gguf")

WORKING_DIR = args.workingdir
DATA_PATH   = args.datapath
QUERY_PATH  = args.querypath
OUTPUT_PATH = args.outputpath

print("USING LLM:", HF_LLM)
print("USING WORKING DIR:", WORKING_DIR)

os.makedirs(WORKING_DIR, exist_ok=True)



if USE_GGUF:
    # llama.cpp backend (fast CPU)
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    GGUF_REPO = "dicta-il/dictalm2.0-instruct-GGUF"
    GGUF_FILE = "dictalm2.0-instruct.Q4_K_M.gguf"

    gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)

    llm = Llama(
        model_path=gguf_path,
        n_ctx=8192,
        n_threads=12,
        n_batch=1024,
        logits_all=False,
        verbose=False,
    )

    _HF_GEN_SEM = asyncio.Semaphore(1)

    async def hf_model_complete(prompt: str, **kwargs) -> str:
        async with _HF_GEN_SEM:
            out = llm(
                prompt,
                max_tokens=32,
                temperature=0.0,
                top_p=1.0,
                stop=["</s>", "### הוראה:", "### תגובה:"],
            )
            return out["choices"][0]["text"].strip()

else:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

    _hf_tokenizer = AutoTokenizer.from_pretrained(
        HF_LLM, use_fast=True, trust_remote_code=True, padding_side="left"
    )
    _hf_model = AutoModelForCausalLM.from_pretrained(
        HF_LLM,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cpu").eval()

    if _hf_tokenizer.pad_token_id is None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token or "[PAD]"
        _hf_model.resize_token_embeddings(len(_hf_tokenizer))

    _gen_cfg = GenerationConfig(
        max_new_tokens=32, do_sample=False, temperature=0.0, top_p=1.0,
        pad_token_id=_hf_tokenizer.pad_token_id, eos_token_id=_hf_tokenizer.eos_token_id,
    )
    _hf_pipe = pipeline(
        "text-generation",
        model=_hf_model,
        tokenizer=_hf_tokenizer,
        device=-1,
        generation_config=_gen_cfg,
        return_full_text=False,
    )

    _HF_GEN_SEM = asyncio.Semaphore(1)

    async def hf_model_complete(prompt: str, **kwargs) -> str:
        async with _HF_GEN_SEM:
            gen = _hf_pipe(prompt)
            return gen[0]["generated_text"].strip()




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

    # chunk_token_size=4000,
    # chunk_overlap_token_size=10,


    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=512,
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

    if args.checkpoints <= 0 or total <= 1:
        step = None
    else:
        # spread ~args.checkpoints checkpoints across 'total' items
        step = max(1, total // args.checkpoints)

    if args.save:
        os.makedirs(os.path.join(WORKING_DIR, "checkpoints"), exist_ok=True)

    for idx, chunk_path in enumerate(chunks, start=1):
        print(f"{idx}/{total}  {chunk_path}")

        # Insert content
        with open(chunk_path, "r", encoding="utf-8", errors="ignore") as f:
            rag.insert(f.read())

        # Periodic checkpointing
        if args.save:
            if step is None:
                do_checkpoint = (idx == total)
            else:
                do_checkpoint = (idx % step == 0) or (idx == total)

            if do_checkpoint:
                ch_id += 1
                print(f"--- Saving index checkpoint #{ch_id} ---")
                with open(os.path.join(WORKING_DIR, f"checkpoints/rag_ch_{ch_id}.pkl"), "wb") as f:
                    pickle.dump(rag, f)
                with open(os.path.join(WORKING_DIR, "checkpoints/rag_final.pkl"), "wb") as f:
                    pickle.dump(rag, f)
                print(f"--- Checkpoint #{ch_id} saved ---")

    print("INFO: Document processing pipeline completed")
    print("INFO: If entity extraction is enabled in MiniRAG, it will run automatically.")


if __name__ == "__main__":
    main()

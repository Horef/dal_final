

import os
import sys
import argparse
import asyncio
from typing import List
import pickle

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
    parser.add_argument("--model", type=str, default="bloomz",
                        help="Only bloomz works at this stage")
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

if args.model == "bloomz":
    HF_LLM = "bigscience/bloomz-560m"          # instruction-tuned, multilingual (but no Hebrew)

# elif args.model == "bloom1":
#     HF_LLM = "bigscience/bloom-1b1"
# elif args.model == "GLM":
#     HF_LLM = "THUDM/glm-edge-1.5b-chat"
# elif args.model == "MiniCPM":
#     HF_LLM = "openbmb/MiniCPM3-4B"
# elif args.model == "qwen":
#     HF_LLM = "Qwen/Qwen2.5-3B-Instruct"
else:
    print("Only bloomz works at this stage")
    sys.exit(1)

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


_hf_tokenizer = AutoTokenizer.from_pretrained(HF_LLM, trust_remote_code=False)

_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
_hf_model = AutoModelForCausalLM.from_pretrained(
    HF_LLM,
    torch_dtype=_dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)

_hf_model.eval()
# # Fallback ids
if _hf_tokenizer.pad_token_id is None:
    if _hf_tokenizer.eos_token_id is not None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token
    else:
        _hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        _hf_model.resize_token_embeddings(len(_hf_tokenizer))


# Manual device placement (M60: CC 5.2; torch 1.13.1 works)
device_arg = -1

if torch.cuda.is_available():
    print("CUDA is available")
    try:
        maj, minr = torch.cuda.get_device_capability(0)
        print(f"Detected GPU: {torch.cuda.get_device_name(0)} (capability {maj}.{minr})")

        if (maj, minr) >= (5, 2):  # Tesla M60 is 5.2
            _hf_model.to("cuda:0")
            device_arg = 0
            print("Model successfully moved to CUDA:0")
        else:
            print(f"GPU capability {maj}.{minr} is below required (5.2), using CPU instead")
    except Exception as e:
        print(f"Error while checking CUDA device: {e}")
        device_arg = -1
else:
    print("CUDA is NOT available, running on CPU")

print(f"Final device_arg = {device_arg}")

_hf_pipe = pipeline(
    "text-generation",
    model=_hf_model,
    tokenizer=_hf_tokenizer,
    device=device_arg,
)


async def hf_model_complete(prompt: str, **kwargs) -> str:
    """
    Async wrapper compatible with MiniRAG's awaited LLM interface.
    Generates short, deterministic-ish completions for indexing/entity tasks.
    """
    gen = _hf_pipe(
        prompt,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=_hf_tokenizer.pad_token_id,
        eos_token_id=_hf_tokenizer.eos_token_id,
    )
    text = gen[0]["generated_text"]
    # Strip the prompt prefix if the pipeline returns prompt+completion
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


# ----------------------------
# Build MiniRAG with HF-only stack
# ----------------------------

_EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
_EMB_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL).eval()

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

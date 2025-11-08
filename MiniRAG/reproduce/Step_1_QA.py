# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import torch
import sys
import os
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import hf_embed
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="bloomz")
    parser.add_argument("--outputpath", type=str, default="./logs/qa_output.csv")
    parser.add_argument("--workingdir", type=str, default="./Step0_res")
    parser.add_argument("--datapath", type=str, default="./dataset/Technion/data/")
    parser.add_argument(
        "--querypath", type=str, default="./dataset/Technion/qa/query_set.csv"
    )
    args = parser.parse_args()
    return args


args = get_args()


if args.model == "bloomz":
    LLM_MODEL = "bigscience/bloomz-560m"
# elif args.model == "neo":
#     HF_LLM = "Norod78/hebrew-gpt_neo-small"
# elif args.model == "bloom1":
#     LLM_MODEL = "bigscience/bloom-1b1"
# elif args.model == "GLM":
#     LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
# elif args.model == "MiniCPM":
#     LLM_MODEL = "openbmb/MiniCPM3-4B"
# elif args.model == "qwen":
#     LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
_hf_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=False)

use_cuda = torch.cuda.is_available()
_dtype = torch.float16 if use_cuda else torch.float32

if use_cuda:
    torch.cuda.empty_cache()

_hf_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=_dtype,
    low_cpu_mem_usage=True,
    device_map={"": 0} if use_cuda else None,
    trust_remote_code=False,
)

_hf_model.eval()


if _hf_tokenizer.pad_token_id is None:
    if _hf_tokenizer.eos_token_id is not None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token
    else:
        _hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        _hf_model.resize_token_embeddings(len(_hf_tokenizer))

if use_cuda:
    torch.cuda.empty_cache()



if torch.cuda.is_available():
    device_str = "cuda:0"
    device_arg = 0
    try:
        maj, minr = torch.cuda.get_device_capability(0)
        print(f"Detected GPU: {torch.cuda.get_device_name(0)} (capability {maj}.{minr})")
    except Exception as e:
        print(f"GPU info probe failed: {e}")
    _hf_model.to(device_str)
    print("Model successfully moved to CUDA:0")
else:
    device_str = "cpu"
    device_arg = -1
    print("CUDA is NOT available, running on CPU")

print(f"Final device_arg = {device_arg}")


_hf_pipe = pipeline(
    "text-generation",
    model=_hf_model,
    tokenizer=_hf_tokenizer,
    return_full_text=False,
)


async def hf_model_complete(prompt: str, **kwargs) -> str:
    """Async wrapper compatible with MiniRAG's awaited LLM interface."""
    gen = _hf_pipe(
        prompt,
        max_new_tokens=5000,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=_hf_tokenizer.pad_token_id,
        eos_token_id=_hf_tokenizer.eos_token_id,
    )
    return gen[0]["generated_text"].strip()


# ----------------------------
# Build MiniRAG with HF-only stack
# ----------------------------

_EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
_EMB_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL).eval()

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
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
# Now QA
QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row["Question"])
        GA_LIST.append(row["Gold Answer"])


def run_experiment(output_path):
    # cleaning the output file
    if os.path.exists(output_path):
        os.remove(output_path)

    headers = ["Question", "Gold Answer", "minirag_context", "minirag", "naive_context", "naive"]

    q_already = []
    if os.path.exists(output_path):
        with open(output_path, mode="r", encoding="utf-8") as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.append(row["Question"])

    row_count = len(q_already)
    print("row_count", row_count)

    with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file, delimiter="@")
        if row_count == 0:
            writer.writerow(headers)

        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):  #
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
                print(f'minirag_answer: "{minirag_answer}"')
            except Exception as e:
                print("Error in minirag_answer", e)
                minirag_answer = "Error"
            try:
                minirag_context = (
                    rag.query(QUESTION, param=QueryParam(mode='mini', only_need_context=True))
                    .replace("\n", "")
                    .replace("\r", "")
                )
                print(f'minirag_context: "{minirag_context}"')
            except Exception as e:
                print("Error in minirag_context", e)
                minirag_context = "Error"

            try:
                naive_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="naive"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
                print(f'naive_answer: "{naive_answer}"')
            except Exception as e:
                print("Error in naive_answer", e)
                naive_answer = "Error"
            try:
                naive_context = (
                    rag.query(QUESTION, param=QueryParam(mode='naive', only_need_context=True))
                    .replace("\n", "")
                    .replace("\r", "")
                )
                print(f'naive_context: "{naive_context}"')
            except Exception as e:
                print("Error in naive_context", e)
                naive_context = "Error"

            writer.writerow([QUESTION, Gold_Answer, minirag_context, minirag_answer, naive_context, naive_answer])

    print(f"Experiment data has been recorded in the file: {output_path}")


# if __name__ == "__main__":
run_experiment(OUTPUT_PATH)
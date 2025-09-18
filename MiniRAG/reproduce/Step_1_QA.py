# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "12")

import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    hf_model_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "dicta-il/dictabert"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="dictalm")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./Technion")
    parser.add_argument("--datapath", type=str, default="./dataset/Technion/data/")
    parser.add_argument(
        "--querypath", type=str, default="./dataset/Technion/qa/query_set.csv"
    )
    args = parser.parse_args()
    return args


args = get_args()


if args.model == "bloomz":
    LLM_MODEL = "bigscience/bloomz-560m"
elif args.model == "dictalm":
    LLM_MODEL = "dicta-il/dictalm2.0"
elif args.model == "dictalm_no_gguf":
    LLM_MODEL = "dicta-il/dictalm2.0-instruct"
elif args.model == "neo":
    LLM_MODEL = "Norod78/hebrew-gpt_neo-small"
elif args.model == "bloom1":
    LLM_MODEL = "bigscience/bloom-1b1"
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == "MiniCPM":
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == "qwen":
    LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
else:
    print("Invalid model name")
    exit(1)


USE_GGUF = LLM_MODEL.lower().endswith("-gguf")


WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath

print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


os.makedirs(WORKING_DIR, exist_ok=True)

# ----------------------------
# Build completion function
# ----------------------------
if USE_GGUF:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    GGUF_REPO = "dicta-il/dictalm2.0-instruct-GGUF"
    GGUF_FILE = "dictalm2.0-instruct.Q4_K_M.gguf"
    gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)

    llm_cpp = Llama(
        model_path=gguf_path,
        n_ctx=8192,
        n_threads=int(os.environ.get("OMP_NUM_THREADS", "12")),
        n_batch=1024,
        logits_all=False,
        verbose=False,
    )

    _GEN_SEM = asyncio.Semaphore(1)

    async def llm_complete(prompt: str, **kwargs) -> str:
        async with _GEN_SEM:
            out = llm_cpp(
                prompt,
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                stop=["</s>", "### הוראה:", "### תגובה:"],
            )
            return out["choices"][0]["text"].strip()
else:
    llm_complete = hf_model_complete


# ----------------------------
# Preload DictaBERT once (correct dims)
# ----------------------------
_EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
_EMB_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL)
_EMB_MODEL.to("cpu").eval()

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=768,  # DictaBERT is 768-dim
        max_token_size=512,
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
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)

        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                minirag_context = (
                    rag.query(QUESTION, param=QueryParam(mode='mini', only_need_context=True))
                      .replace("\n", "").replace("\r", "")
                )
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                      .replace("\n", "").replace("\r", "")
                )
            except Exception as e:
                print("Error in minirag_answer", e)
                minirag_context = ""
                minirag_answer = "Error"

            try:
                naive_context = (
                    rag.query(QUESTION, param=QueryParam(mode='naive', only_need_context=True))
                      .replace("\n", "").replace("\r", "")
                )
                naive_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="naive"))
                      .replace("\n", "").replace("\r", "")
                )
            except Exception as e:
                print("Error in naive_answer", e)
                naive_context = ""
                naive_answer = "Error"

            writer.writerow([QUESTION, Gold_Answer, minirag_context, minirag_answer, naive_context, naive_answer])

    print(f"Experiment data has been recorded in the file: {output_path}")


# if __name__ == "__main__":
run_experiment(OUTPUT_PATH)
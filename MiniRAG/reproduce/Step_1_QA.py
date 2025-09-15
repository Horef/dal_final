# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    hf_model_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="bloomz")
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
elif args.model == "aleph_gpt":
    HF_LLM = "onlplab/aleph-gpt-base"
elif args.model == "neo":
    HF_LLM = "Norod78/hebrew-gpt_neo-small"
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

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    # llm_model_func=gpt_4o_mini_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
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

        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):  #
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                minirag_context = (
                    rag.query(QUESTION, param=QueryParam(mode='mini', only_need_context=True))
                    .replace("\n", "")
                    .replace("\r", "")
                )
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
            except Exception as e:
                print("Error in minirag_answer", e)
                minirag_answer = "Error"

            try:
                naive_context = (
                    rag.query(QUESTION, param=QueryParam(mode='naive', only_need_context=True))
                    .replace("\n", "")
                    .replace("\r", "")
                )
                naive_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="naive"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
            except Exception as e:
                print("Error in naive_answer", e)
                naive_answer = "Error"

            writer.writerow([QUESTION, Gold_Answer, minirag_context, minirag_answer, naive_context, naive_answer])

    print(f"Experiment data has been recorded in the file: {output_path}")


# if __name__ == "__main__":
run_experiment(OUTPUT_PATH)
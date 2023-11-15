"""
Finetune an LLM using the generated datasets
"""
from typing import Tuple, List, Dict
from math import isnan
import pandas as pd
import numpy as np
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import datasets
from tqdm import tqdm


def load_llama() -> Tuple[AutoModel, AutoTokenizer]:
    # model_name = "codellama/CodeLlama-7b-Python-hf"
    model_name = "codellama/CodeLLama-7b-instruct-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    return model, tokenizer


def massage_solved_dataset(infile: str, outfile: str) -> pd.DataFrame:
    def soln_cols(n: int) -> List[str]:
        return [f"solution {i}" for i in range(n)]

    assert infile.endswith(".jsonl") and outfile.endswith(".jsonl"), \
        f"Expected jsonl files, but got infile={infile} and outfile={outfile}"
    df = pd.read_json(infile, lines=True)
    df = df[["id", "key", "value"]]
    df = df[df["key"].isin(["id", "original problem", "restyled problem"] + soln_cols(3))]
    df = df.drop_duplicates(subset=["id", "key"], keep="first")
    df = df.pivot(index="id", columns="key", values="value")
    df = df.where(pd.notnull(df), None)

    # split solns into separate rows: each id should be `file:problem-id:soln-id`
    # fixme: for now, we make the simplifying assumption that all solutions
    #   are good, so use any problem/solution pair to fine-tune
    rows = []
    for id, row in tqdm(df.iterrows(), total=len(df)):
        for i, soln in enumerate(soln_cols(3)):
            if row[soln]:
                rows.append({
                    "id": f"{id}:{i}",
                    "original problem": row["original problem"],
                    "restyled problem": row["restyled problem"],
                    "solution": row[soln]
                })
    df = pd.DataFrame.from_records(rows)
    df.to_json(outfile, orient="records", lines=True)
    return df
    

def demo_llama():
    model, tokenizer = load_llama()
    input_text = [
        "#PROBLEM\nTry solving this programming problem:  Write a function in Python to generate the first n Fibonacci numbers.\n",
        "#SOLUTION\n",
    ]
    input_tokens = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_tokens, max_length=1000, do_sample=True, top_p=0.95, top_k=60)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 20)

    # demo_llama()
    root = "/home/djl328/prob-repl/datasets/wiz"
    massage_solved_dataset(
        infile=f"{root}/all-solved-20k:30k.jsonl",
        outfile=f"{root}/paired-20k:30k.jsonl"
    )
    # data = datasets.load_dataset("json", data_files=f"{root}/all-solved-20:30k.jsonl")
    # print(data)

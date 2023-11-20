"""
Finetune an LLM using the generated datasets
"""
from typing import Tuple, List, Dict
from math import isnan
import pandas as pd
import numpy as np
import torch as T
from tqdm import tqdm
import wandb

# hugging face
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import datasets


def load_llama() -> Tuple[AutoModel, AutoTokenizer]:
    # model_name = "codellama/CodeLlama-7b-Python-hf"
    model_name = "codellama/CodeLLama-7b-instruct-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    return model, tokenizer


def load_dummy_model() -> Tuple[AutoModel, AutoTokenizer]:
    model_name = "Salesforce/codegen-350M-mono"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
                    "original problem": "# Question:\n" + row["original problem"],
                    "restyled problem": "# Question:\n" + row["restyled problem"],
                    "solution": "# Answer:\n" + row[soln]
                })
    df = pd.DataFrame.from_records(rows)
    df.to_json(outfile, orient="records", lines=True)
    return df


def load_dataset(filename: str) -> datasets.Dataset:
    assert filename.endswith(".jsonl"), \
        f"Expected jsonl file, but got filename={filename}"
    
    df = pd.read_json(filename, lines=True)
    data = datasets.Dataset.from_pandas(df)
    return data


def demo_llama():
    model, tokenizer = load_llama()
    input_text = [
        "#PROBLEM\nTry solving this programming problem:  Write a function in Python to generate the first n Fibonacci numbers.\n",
        "#SOLUTION\n",
    ]
    input_tokens = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_tokens, max_length=1000, do_sample=True, top_p=0.95, top_k=60)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def tuning():
    model, tokenizer = load_llama()
    

def format_prompts(x) -> List[str]:
    outputs = []
    problems, solutions = x["restyled problem"], x["solution"]
    assert len(problems) == len(solutions), \
        (f"Expected to have the same number of problems and solutions, but got "
         f"|problems|={len(problems)}, |solutions|={len(solutions)}")
    for problem, solution in zip(problems, solutions):
        outputs.append(f"# Question: {problem}\n "
                       f"# Answer: {solution}")
    return outputs


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 20)

    root = "/home/djl328/prob-repl/datasets/wiz"
    # root = "../../datasets/wiz"

    # # massage dataset
    # massage_solved_dataset(f"{root}/all-solved-20k:30k.jsonl",
    #                        f"{root}/paired-plus-20k:30k.jsonl")

    # demo_llama()
    dataset = load_dataset(f"{root}/paired-plus-20k:30k.jsonl")
    print(dataset)

    # model, tokenizer = load_llama()
    model, tokenizer = load_dummy_model()
    print(
        f"eos token id: {tokenizer.eos_token_id}\n"
        f"eos token: {tokenizer.eos_token}\n"
        f"pad token id: {tokenizer.pad_token_id}\n"
        f"pad token: {tokenizer.pad_token}"
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = "<|pad|>"

    response_template = "# Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=2,
    )

    wandb.init(project="sft")
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        data_collator=collator,
        dataset_text_field="solution",
        args=args,
        max_seq_length=2048,
    )
    trainer.train()
    

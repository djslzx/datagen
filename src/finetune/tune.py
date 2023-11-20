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
import sys

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


def massage_solved_dataset(infile: str, outfile_prefix: str):
    def soln_cols(n: int) -> List[str]:
        return [f"solution {i}" for i in range(n)]

    assert infile.endswith(".jsonl"), f"Expected jsonl file, but got infile={infile}"

    # remap source names
    def rename(s_id: str) -> str:
        name_map = {
            "CA-20k": "CA",
            "NS-euler": "NSE",
            "NS": "NSCA",
            "Wiz-deep": "WD",
            "Wiz-wide": "WW",
        }
        s_src, s_num = s_id.split(":")
        s_src = name_map[s_src] if s_src in name_map else s_src
        return f"{s_src}:{s_num}"

    df = pd.read_json(infile, lines=True)
    df = df[["id", "key", "value"]]
    df["id"] = df["id"].apply(rename)
    df = df[df["key"].isin(["id", "original problem", "restyled problem"] + soln_cols(3))]
    df = df.drop_duplicates(subset=["id", "key"], keep="first")
    df = df.pivot(index="id", columns="key", values="value")
    df = df.where(pd.notnull(df), None)
    df["source"] = df.index.map(lambda x: x.split(":")[0])

    # split solns into separate rows: each id should be `file:problem-id:soln-id`
    # fixme: for now, we make the simplifying assumption that all solutions
    #   are good, so use any problem/solution pair to fine-tune

    # split each source file into its own dataset
    for source in df["source"].unique():
        data = df[df["source"] == source]
        rows = []
        for id, row in tqdm(data.iterrows(), total=len(data), desc=f"Massaging dataset={source}"):
            for i, soln in enumerate(soln_cols(3)):
                if row[soln]:
                    rows.append({
                        "id": f"{id}:{i}",
                        "source": source,
                        "original problem": row["original problem"],
                        "restyled problem": row["restyled problem"],
                        "solution": row[soln],
                    })
        out = pd.DataFrame.from_records(rows)
        out.to_json(f"{outfile_prefix}-{source}.jsonl", orient="records", lines=True)


def load_dataset(filename: str) -> datasets.Dataset:
    assert filename.endswith(".jsonl"), \
        f"Expected jsonl file, but got filename={filename}"
    
    df = pd.read_json(filename, lines=True)
    data = datasets.Dataset.from_pandas(df)
    return data


def demo_llama():
    model, tokenizer = load_llama()
    input_text = [
        "#PROBLEM\Write a function in Python to generate the first n Fibonacci numbers.\n",
        "#SOLUTION\n",
    ]
    input_tokens = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_tokens, max_length=1000, do_sample=True, top_p=0.95, top_k=60)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def tune_all(model: AutoModel, tokenizer: AutoTokenizer, train_datasets: List[str]):
    """
    Finetune multiple versions of the model, one for each training dataset.
    """
    pass


def tune_once(model: AutoModel, tokenizer: AutoTokenizer, dataset: datasets.Dataset, dataset_name: str,
              max_seq_length=2048):

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

    if not tokenizer.pad_token:
        print("WARNING: no pad token defined by tokenizer, setting to default value", file=sys.stderr)
        # fixme: not sure if this is the right pad token to use, as the original tokenizer has no padding token
        tokenizer.pad_token = "<|pad|>"

    # filter out elements in the dataset that are longer than the cutoff
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x["restyled problem"] + x["solution"])) < max_seq_length)

    # Compensate for lack of context in response template 
    response_template_w_context = "\n# Answer:"
    response_template_ids = tokenizer.encode(response_template_w_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=f"{dataset_name}-output",
        per_device_train_batch_size=2,
    )

    wandb.init(project="sft")
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        data_collator=collator,
        formatting_func=format_prompts,
        args=args,
        max_seq_length=max_seq_length,
    )
    trainer.train()
    trainer.save_model(f"output/model-ft-{dataset_name}")
    

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 20)

    args = sys.argv[1:]
    assert len(args) == 2, f"Expected name, filename but got {args}"
    name, filename = args

    # # massage dataset
    # root = "/home/djl328/prob-repl/datasets/wiz"
    # root = "../../datasets/wiz"
    # massage_solved_dataset(f"{root}/all-solved-20k:30k.jsonl",
    #                        f"{root}/paired-20k:30k")

    dataset = load_dataset(filename)
    # model, tokenizer = load_llama()
    model, tokenizer = load_dummy_model()
    tune_once(model, tokenizer, dataset, dataset_name=name)

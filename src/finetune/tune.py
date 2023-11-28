"""
Finetune an LLM using the generated datasets
"""
from typing import Tuple, List, Dict, Callable
from math import isnan
import pandas as pd
import numpy as np
import torch as T
from tqdm import tqdm
import wandb
import sys
from datetime import datetime
import argparse

# hugging face
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetInfo, DatasetDict, load_dataset


def timestamp():
    return datetime.now().isoformat()


def split_by_percentages(xs: List, ps: Dict[str, float]) -> Dict[str, List]:
    """
    Given a list and dictionary of names/weights,
    split the list by weights and return the named splits.
    """
    assert abs(1 - sum(ps.values())) <= 1e-6, "Split percentages must add up to 1"
    outs = {}
    start = 0
    n = len(xs)
    for key, p in ps.items():
        step = int(p * n)
        outs[key] = xs[start:start + step]
        start += step
    return outs


def test_split_by_percentages():
    cases = [
        [1,2,3], {"a": 1/3, "b": 1/3, "c": 1/3},
        {"a": [1], "b": [2], "c": [3]},
        [1] * 80 + [2] * 10 + [3] * 10, {"train": 0.8, "validate": 0.1, "test": 0.1},
        {"train": [1] * 80,
         "validate": [2] * 10,
         "test": [3] * 10},
    ]
    for xs, ps, y in zip(cases[0::3], cases[1::3], cases[2::3]):
        out = split_by_percentages(xs, ps)
        assert out == y, f"Expected {y} but got {out}"
    

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


def massage_solved_dataset(
        in_file: str, 
        out_dir: str,
        name_map: Dict[str, str] = None,
):
    """
    Clean up datasets consisting of problems, solutions, and checkers.
    - pivot from kv to columnar form
    - rename datasets, e.g. CA-20k to CA
    - extract source from id
    - split into separate output files by source
    - shuffle dataset
    """
    assert in_file.endswith(".jsonl"), f"Expected jsonl file, but got in_file={in_file}"

    if not name_map:
        name_map = {
            "CA-20k": "CA",
            "NS-euler": "NSE",
            "NS": "NSCA",
            "Wiz-deep": "WD",
            "Wiz-wide": "WW",
        }

    def soln_keys(n: int) -> List[str]:
        return [f"solution {i}" for i in range(n)]

    def rename(s_id: str) -> str:
        s_src, s_num = s_id.split(":")
        s_src = name_map[s_src] if s_src in name_map else s_src
        return f"{s_src}:{s_num}"

    n_solns = 3
    df = pd.read_json(in_file, lines=True)
    df = df[["id", "key", "value"]]
    df["id"] = df["id"].apply(rename)
    df = df[df["key"].isin(["id", "original problem", "restyled problem"] + soln_keys(n_solns))]
    df = df.drop_duplicates(subset=["id", "key"], keep="first")

    df = df.pivot(index="id", columns="key", values="value")
    df = df.where(pd.notnull(df), None)
    df["source"] = df.index.map(lambda x: x.split(":")[0])

    # fixme: for now, we make the simplifying assumption that all solutions
    #   are good, so use any problem/solution pair to fine-tune

    # shuffle data
    df = df.sample(frac=1)

    # split each source file into its own dataset
    for source in sorted(df["source"].unique()):
        data = df[df["source"] == source]
        rows = []
        print(f"Found {len(data)} lines in {source}, processing...", file=sys.stderr)
        for id, row in tqdm(data.iterrows(), total=len(data), desc=f"Massaging {source}"):
            for i, soln in enumerate(soln_keys(n_solns)):
                if row[soln]:
                    rows.append({
                        "id": f"{id}:{i}",
                        "source": source,
                        "original problem": row["original problem"],
                        "restyled problem": row["restyled problem"],
                        "solution": row[soln],
                    })
        ds = Dataset.from_pandas(
            pd.DataFrame.from_records(rows),
            info=DatasetInfo(
                dataset_name=source,
                description=f"{source} dataset",
            ),
        )
        tt = ds.train_test_split(test_size=0.2)
        vt = tt["test"].train_test_split(test_size=0.5)
        dd = DatasetDict({
            "train": tt["train"],
            "validation": vt["train"],
            "test": vt["test"],
        })
        dd.save_to_disk(f"{out_dir}/{source}")


def tune_all(model: AutoModel, tokenizer: AutoTokenizer, train_datasets: List[str]):
    """
    Finetune multiple versions of the model, one for each training dataset.
    """
    pass


def tune_once(model: AutoModel, 
              tokenizer: AutoTokenizer, 
              dataset: Dict, 
              max_seq_length=2048,
              problem_key="restyled problem",
):
    def format_prompt(q: str, a: str) -> str:
        return f"# Question: {q}\n# Answer: {a}\n#DONE#"

    def format_prompts(x) -> List[str]:
        outputs = []
        problems, solutions = x[problem_key], x["solution"]
        assert len(problems) == len(solutions), \
            (f"Expected to have the same number of problems and solutions, but got "
             f"|problems|={len(problems)}, |solutions|={len(solutions)}")
        for problem, solution in zip(problems, solutions):
            outputs.append(format_prompt(problem, solution))
        return outputs

    def encode_len(x: Dict):
        return len(tokenizer.encode(format_prompt(x[problem_key], x['solution'])))

    assert problem_key in {"original problem", "restyled problem"}, \
        (f"Invalid problem key {problem_key}: must be one of "
         "{'original problem', 'restyled problem'}")

    ts = timestamp()

    if not tokenizer.pad_token:
        print("WARNING: no pad token defined by tokenizer, setting to default", file=sys.stderr)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.truncation = True
    tokenizer.padding = "max_length"

    # Compensate for lack of context in response template 
    response_template_w_context = "\n# Answer:"
    response_template_ids = tokenizer.encode(response_template_w_context,
                                             add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, 
        tokenizer=tokenizer
    )
    args = TrainingArguments(
        output_dir=f"../models/ft/{dataset['train'].info.dataset_name}/{ts}",
        per_device_train_batch_size=2,
        bf16=True,
        evaluation_strategy="steps",
    )

    # filter out data that is too long
    train = dataset['train'].filter(lambda x: encode_len(x) < max_seq_length)
    validation = dataset['validation'].filter(lambda x: encode_len(x) < max_seq_length)

    wandb.init(project="sft")
    trainer = SFTTrainer(
        model,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=collator,
        formatting_func=format_prompts,
        args=args,
        max_seq_length=max_seq_length,
    )
    trainer.train()
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", type=str, choices=["data", "tune"])
    p.add_argument("--data", type=str)
    p.add_argument("--out-dir", type=str)
    p.add_argument("--model", type=str)

    args = p.parse_args()
    if args.mode == "data":
        massage_solved_dataset(in_file=args.data, out_dir=args.out_dir)
    elif args.mode == "tune":
        dataset = DatasetDict.load_from_disk(args.data)
        model, tokenizer = load_dummy_model()
        tune_once(model, tokenizer, dataset=dataset, max_seq_length=1024)

    # todos:
    # - add validation loss to metrics during ft
    # - sweep using wandb
    # - set up memorization test (100 examples)
    # - while sweep runs:
    #   - set up test/eval framework
    #     - use standard datasets (HumanEval, DS1000, APPS, MBPP)
    #     - report %checkers passed for each of n solutions sampled per problem
    #   - set up problem distribution visualization


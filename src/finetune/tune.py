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
from datetime import datetime

# hugging face
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import datasets


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
        split: Dict[str, float] = None,
):
    """
    Clean up datasets consisting of problems, solutions, and checkers.
    - pivot from kv to columnar form
    - rename datasets, e.g. CA-20k to CA
    - extract source from id
    - split into separate output files by source
    - train/validation/test split
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

    if not split:
        split = {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
        }

    def soln_cols(n: int) -> List[str]:
        return [f"solution {i}" for i in range(n)]

    def rename(s_id: str) -> str:
        s_src, s_num = s_id.split(":")
        s_src = name_map[s_src] if s_src in name_map else s_src
        return f"{s_src}:{s_num}"

    df = pd.read_json(in_file, lines=True)
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

    # shuffle data
    df = df.sample(frac=1)

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
        # train/validate/test split
        for name, subset in split_by_percentages(rows, split).items():
            out = pd.DataFrame.from_records(subset)
            out.to_json(f"{out_dir}/{source}/{name}.jsonl", orient="records", lines=True)


def load_json_dataset(filename: str) -> datasets.Dataset:
    assert filename.endswith(".jsonl"), \
        f"Expected jsonl file, but got filename={filename}"
    
    df = pd.read_json(filename, lines=True)
    data = datasets.Dataset.from_pandas(df)
    return data


def tune_all(model: AutoModel, tokenizer: AutoTokenizer, train_datasets: List[str]):
    """
    Finetune multiple versions of the model, one for each training dataset.
    """
    pass


def tune_once(model: AutoModel, 
              tokenizer: AutoTokenizer, 
              dataset_name: str,
              dataset: datasets.Dataset, 
              max_seq_length=2048,
              problem_key = "restyled problem",
):
    def format_prompt(q: str, a: str) -> str:
        return f"# Question: {q}\n# Answer: {a}\n#DONE#"

    def format_prompts(x) -> List[str]:
        """
        Format each question-answer pair as a single string with headers.
        These will be split later using the response template
        """
        outputs = []
        problems, solutions = x[problem_key], x["solution"]
        assert len(problems) == len(solutions), \
            (f"Expected to have the same number of problems and solutions, but got "
             f"|problems|={len(problems)}, |solutions|={len(solutions)}")
        for problem, solution in zip(problems, solutions):
            outputs.append(format_prompt(problem, solution))
        return outputs

    # todo
    def compute_metrics(eval_predictions):
        pass

    assert problem_key in {"original problem", "restyled problem"}, \
        (f"Invalid problem key {problem_key}: must be one of "
         "{'original problem', 'restyled problem'}")

    ts = timestamp()

    if not tokenizer.pad_token:
        print("WARNING: no pad token defined by tokenizer, setting to [PAD]", file=sys.stderr)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    print(tokenizer.eos_token, tokenizer.eos_token_id,
          tokenizer.pad_token, tokenizer.pad_token_id)

    tokenizer.truncation = True
    tokenizer.padding = "max_length"

    # # filter out elements in the dataset that are longer than the cutoff
    # dataset = dataset.filter(lambda x:
    #                          len(tokenizer.encode(format_prompt(x[problem_key], 
    #                                                             x["solution"]))) < max_seq_length)

    # Compensate for lack of context in response template 
    response_template_w_context = "\n# Answer:"
    response_template_ids = tokenizer.encode(response_template_w_context, 
                                             add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=f"../models/ft/{dataset_name}/{ts}",
        per_device_train_batch_size=2,
        bf16=True,
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
    

if __name__ == "__main__":
    # test_split_by_percentages()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 20)

    args = sys.argv[1:]
    assert len(args) == 2, f"Expected name, filename but got {args}"
    name, filename = args

    # massage dataset
    root = "/home/djl328/prob-repl/datasets/wiz"
    # root = "../../datasets/wiz"
    massage_solved_dataset(in_file=f"{root}/all-solved-20k:30k.jsonl",
                           out_dir=f"{root}/paired-20k:30k")
    exit(0)

    dataset = load_json_dataset(filename)
    model, tokenizer = load_dummy_model()
    tune_once(model, tokenizer, dataset_name=name, dataset=dataset, max_seq_length=1024)

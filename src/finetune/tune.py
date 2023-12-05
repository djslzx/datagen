"""
Finetune an LLM using the generated datasets
"""
from typing import Tuple, List, Dict, Callable, Set, Optional
from math import isnan
import pandas as pd
import numpy as np
import torch as T
from tqdm import tqdm
import wandb
import sys
from datetime import datetime
import argparse
import yaml
import pdb

# hugging face
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetInfo, DatasetDict, load_dataset
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training


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


def check_in(x, name: str, options: Set) -> bool:
    assert x in options, f"Invalid {name} {x}: must be in {options}"
    

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
    df = df[df["key"].isin(["id", "restyled problem"] + soln_keys(n_solns))]
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
                        "problem": row["restyled problem"],
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


def format_prompt(q: str, a: str) -> str:
    return f"# Question: {q}\n# Answer: {a}\n#DONE#"

def format_question(q: str) -> str:
    return f"# Question: {q}\n# Answer: "

def format_prompts(x) -> List[str]:
    outputs = []
    problems, solutions = x["problem"], x["solution"]
    assert len(problems) == len(solutions), \
        (f"Expected to have the same number of problems and solutions, but got "
         f"|problems|={len(problems)}, |solutions|={len(solutions)}")
    for problem, solution in zip(problems, solutions):
        outputs.append(format_prompt(problem, solution))
    return outputs


def tune_once(model: AutoModel, 
              tokenizer: AutoTokenizer, 
              dataset: Dict, 
              max_seq_length: int,
              batch_size: int,
              epochs: int,
              lr_init: float,
              lr_scheduler_type: str,
              output_dir: str,
              logging_steps: int,
):

    if not tokenizer.pad_token:
        print("WARNING: no pad token defined by tokenizer, setting to default", 
              file=sys.stderr)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.truncation = True
    tokenizer.padding = "max_length"
    tokenizer.padding_side = "right"

    def encode_len(x: Dict):
        return len(tokenizer.encode(format_prompt(x['problem'], x['solution'])))

    # Compensate for lack of context in response template 
    response_template_w_context = "\n# Answer:"
    response_template_ids = tokenizer.encode(response_template_w_context,
                                             add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, 
        tokenizer=tokenizer
    )
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=5000,
        num_train_epochs=epochs,
        learning_rate=lr_init,
        lr_scheduler_type=lr_scheduler_type,
        logging_Steps=logging_steps,
    )

    # filter out data that is too long
    orig_train_len = len(dataset['train'])
    train = dataset['train'].filter(lambda x: encode_len(x) < max_seq_length)
    orig_validation_len = len(dataset['validation'])
    validation = dataset['validation'].filter(lambda x: encode_len(x) < max_seq_length)
    print(f"Train after filtering to max_seq_len={max_seq_length}: "
          f"{orig_train_len} => {len(train)}")
    print(f"Validation after filtering to max_seq_len={max_seq_length}: "
          f"{orig_validation_len} => {len(validation)}")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model,
        train_dataset=train,
        eval_dataset=validation,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=format_prompts,
        args=args,
        max_seq_length=max_seq_length,
    )
    trainer.train()
    

def check_memorized(model: AutoModel, tokenizer: AutoTokenizer, dataset: Dataset):
    problems = []
    for x in dataset['train']:
        problems.append(format_question(x["problem"]))

    tokenizer.truncation = True
    tokenizer.padding = "max_length"
    tokenizer.padding_side = "left"

    inputs = tokenizer(problems, padding='max_length', max_length=512, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for x in outputs:
        print(x)

    # rollout most probable sequence solution to each problem
    pass

    # check that sequence matches the canonical solution
    pass


def load_kbit_model(model_name: str, k: Optional[int]) -> AutoModel:
    if not k:
        return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    elif k == 4:
        return AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
    elif k == 8:
        return AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    else:
        raise ValueError(f"Invalid k for kbit model: k={k}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["data", "finetune", "memorize-train", "memorize-test"])
    p.add_argument("--out-dir")
    p.add_argument("--dataset")
    p.add_argument("--model-name", default="codellama/CodeLLama-7b-instruct-hf")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr-init", type=float, default=5e-5)
    p.add_argument("--lr-scheduler-type", choices=["linear", "cosine", "constant"], default="linear")
    p.add_argument("--kbit", type=int, choices=[4, 8])
    p.add_argument("--logging-steps", type=int, default=500)

    args = p.parse_args()
    if args.mode == "data":
        massage_solved_dataset(in_file=args.dataset, out_dir=args.out_dir)
    elif args.mode == "finetune":
        dataset = DatasetDict.load_from_disk(args.dataset)
        dataset_name = dataset['train'].info.dataset_name
        model = load_kbit_model(args.model_name, k=args.kbit)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
        ts = timestamp()
        tune_once(
            model=model, 
            tokenizer=tokenizer, 
            dataset=dataset, 
            max_seq_length=args.max_seq_length, 
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_init=args.lr_init,
            lr_scheduler_type=args.lr_scheduler_type,
            output_dir=f"/home/djl328/prob-repl/models/sft/{dataset_name}/{ts}"
            logging_steps=args.logging_steps,
        )
    elif args.mode.startswith("memorize-"):
        dataset = DatasetDict.load_from_disk(args.dataset)
        dataset['train'] = dataset['train'].select(range(10))
        dataset['validation'] = dataset['validation'].select([])
        dataset = DatasetDict(dataset)
        model = load_kbit_model(args.model_name, k=args.kbit)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.padding_side = "right"
        ts = timestamp()

        if args.mode == "memorize-train":
            wandb.init(project="sft-memorize")
            tune_once(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr_init=args.lr_init,
                lr_scheduler_type=args.lr_scheduler_type,
                output_dir=f"/home/djl328/prob-repl/models/test/{ts}",
                logging_steps=args.logging_steps,
            )
        else:
            check_memorized(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
            )


if __name__ == "__main__":
    main()
    # todos:
    # - set up memorization test (100 examples)
    # - while sweep runs:
    #   - set up test/eval framework
    #     - use standard datasets (HumanEval, DS1000, APPS, MBPP)
    #     - report %checkers passed for each of n solutions sampled per problem
    #   - set up problem distribution visualization



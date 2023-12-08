"""
Finetune an LLM using the generated datasets
"""
from typing import Tuple, List, Dict, Callable, Set, Optional
from tqdm import tqdm
import wandb
import sys
import argparse
import pdb
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset, DatasetInfo, DatasetDict

import finetune.models as models
import finetune.data as data
import finetune.util as util


def check_memorized(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: DatasetDict):
    prompts = []
    references = []
    for x in dataset['train']:
        prompts.append(models.format_question(x["problem"]))
        references.append(models.format_prompt(x["problem"], x["solution"]))

    outputs = models.sample_model(model, tokenizer, prompts,
                                  max_length=512, max_new_tokens=200,
                                  do_sample=True, num_beams=1)
    for output, reference in zip(outputs, references):
        # count the number of tokens that match
        output_tokens = output.split()
        reference_tokens = reference.split()
        n_matches = 0
        n_tokens = len(reference_tokens)
        for i, (ot, rt) in enumerate(zip(output_tokens, reference_tokens)):
            if ot == rt:
                n_matches += 1
            else:
                break
        print(f"Matched {n_matches} tokens out of {n_tokens} ({n_matches / n_tokens:.2f})")
    print()


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
    p.add_argument("--kbit", type=int, choices=[4, 8, 16])
    p.add_argument("--logging-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=5000)
    p.add_argument("--n-solns", type=int, default=3)
    p.add_argument("--n-tests", type=int, default=3)
    p.add_argument("--id", default="")

    args = p.parse_args()
    if args.mode == "data":
        data.prepare_dataset(
            in_file=args.dataset,
            out_dir=args.out_dir,
            n_solns=args.n_solns,
            n_tests=args.n_tests
        )
    elif args.mode == "finetune":
        dataset = DatasetDict.load_from_disk(args.dataset)
        dataset_name = dataset['train'].info.dataset_name
        model, tokenizer = models.load_model(args.model_name, k=args.kbit)
        ts = util.timestamp()
        suffix = f"{args.id}-{ts}" if args.id else f"{ts}"
        models.finetune_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_init=args.lr_init,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            output_dir=f"/home/djl328/prob-repl/models/sft/{dataset_name}/{suffix}",
        )
    elif args.mode.startswith("memorize-"):
        dataset = DatasetDict.load_from_disk(args.dataset)
        dataset['train'] = dataset['train'].select(range(10))
        dataset['validation'] = dataset['train'].select(range(10))
        dataset = DatasetDict(dataset)
        model, tokenizer = models.load_model(args.model_name, k=args.kbit)
        ts = util.timestamp()

        if args.mode == "memorize-train":
            wandb.init(project="sft-memorize")
            models.finetune_model(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr_init=args.lr_init,
                lr_scheduler_type=args.lr_scheduler_type,
                logging_steps=args.logging_steps,
                eval_steps=args.eval_steps,
                output_dir=f"/home/djl328/prob-repl/models/test/{ts}",
            )
        else:
            check_memorized(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
            )


if __name__ == "__main__":
    main()

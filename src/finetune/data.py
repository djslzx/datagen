"""
Dataset handling:
- given file/id, fetch problems, solutions, and tests
"""
import pdb
from typing import Dict, List, Optional, Tuple, Iterator
import sys
import pandas as pd
import numpy as np
from math import ceil
from datasets import Dataset, DatasetDict, DatasetInfo
from tqdm import tqdm
import argparse

from finetune.root import evaluate, util
from finetune import vet


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["id", "key", "value"]]
    df = df.drop_duplicates(subset=["id", "key"], keep="first")
    df = df.pivot(index="id", columns="key", values="value")
    df = df.where(pd.notnull(df), None)
    df["source"] = df.index.map(lambda x: x.split(":")[0])
    df["id"] = df.index  # ensure we have an id column
    return df


def read_raw_dataset_to_solns_and_tests(
        df: pd.DataFrame,
        name_map: Dict[str, str] = None,
        rename: bool = True,
        n_solns: int = 3,
        n_tests: int = 3,
) -> pd.DataFrame:
    """
    Clean up datasets consisting of problems, solutions, and checkers and return a dataframe keyed
    by id, with columns "problem", "solutions", and "tests".

    Transformations:
    - pivot from kv to columnar form
    - rename datasets, e.g. CA-20k to CA
    - extract source from id
    - group solns, tests into lists
    """
    assert type(df) == pd.DataFrame, f"Expected DataFrame but got {type(df)}"

    if rename and name_map is None:
        name_map = {
            "CA-20k": "CA",
            "NS-euler": "NSE",
            "NS": "NSCA",
            "Wiz-deep": "WD",
            "Wiz-wide": "WW",
        }

    def rename_id(s_id: str) -> str:
        s_src, s_num = s_id.split(":")
        s_src = name_map[s_src] if s_src in name_map else s_src
        return f"{s_src}:{s_num}"

    df["id"] = df["id"].apply(rename_id)
    df = long_to_wide(df)
    df = df.rename(columns={"restyled problem": "problem"})

    # keep only n_tests tests and n_solns solns
    soln_keys = [f"solution {i}" for i in range(n_solns)]
    test_keys = [f"test {i}" for i in range(n_tests)]
    df["tests"] = df.apply(lambda row: [row[k] for k in test_keys if row[k]], axis=1)
    df["solutions"] = df.apply(lambda row: [row[k] for k in soln_keys if row[k]], axis=1)
    df = df[["source", "problem", "solutions", "tests"]]

    # remove problems with empty solutions or empty tests
    df = df[df["solutions"].apply(lambda x: len(x) > 0) &
            df["tests"].apply(lambda x: len(x) > 0)]

    # ensure we have an id column
    df["id"] = df.index

    return df


def to_hf_dataset(df: pd.DataFrame, out_dir: str):
    # shuffle data
    df = df.sample(frac=1)

    # split each source file into its own dataset
    for source in sorted(df["source"].unique()):
        data = df[df["source"] == source]
        rows = []
        print(f"Found {len(data)} lines in {source}, processing...", file=sys.stderr)
        for id, row in tqdm(data.iterrows(), total=len(data), desc=f"Massaging {source}"):
            problem = row["problem"]
            for i, soln in enumerate(row["solutions"]):
                rows.append({
                    "id": f"{id}:{i}",
                    "source": source,
                    "problem": problem,
                    "solution": soln,
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


def run_solns_and_tests(df: pd.DataFrame, timeout: float) -> Iterator[dict]:
    # Run solutions in isolation
    for ident, row in tqdm(df.iterrows(), total=len(df), desc="Running solutions in isolation"):
        for i, soln in enumerate(row["solutions"]):
            result = run_soln(f"{ident}:{i}", row["problem"], soln, timeout)
            for item in util.KVItem.from_dict(result):
                yield item.to_dict()

    # Run solutions with tests
    for ident, row in tqdm(df.iterrows(), total=len(df), desc="Running solutions with tests"):
        for i, soln in enumerate(row["solutions"]):
            for j, test in enumerate(row["tests"]):
                result = run_soln_and_test(f"{ident}:{i}:{j}", row["problem"], soln, test, timeout)
                for item in util.KVItem.from_dict(result):
                    yield item.to_dict()


def run_soln(ident: str, problem: str, soln: str, timeout: float) -> dict:
    result = evaluate.run_soln(soln, timeout)
    return {
        "id": ident,
        "problem": problem,
        "solution": soln,
        "test": None,
        **result.to_dict(prefix="result."),
    }


def run_soln_and_test(ident: str, problem: str, soln: str, test: str, timeout: float) -> dict:
    result = evaluate.run_soln_w_test(soln, test, timeout)
    return {
        "id": ident,
        "problem": problem,
        "solution": soln,
        "test": test,
        **result.to_dict(prefix="result."),
    }


def debug_segfaults(df: pd.DataFrame, timeout: float, out: str):
    # try running some of the problematic solutions/tests
    problems = [
        ("NSCA", 8981),
        ("WW", 703),
        ("WW", 1518),
        ("CA", 864),
        ("CA", 247),
    ]
    radius = 0
    ids = []
    for source, n in problems:
        ids.extend([f"{source}:{m}" for m in range(n - radius, n + radius + 1)])
    df = df[df["id"].isin(ids)]
    df.set_index("id", inplace=True)
    util.incrementally_save_jsonl(
        quiet=True,
        filename=out,
        it=run_solns_and_tests(df, timeout=timeout),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--mode", choices=["eval", "process", "split", "debug", "analyze"])
    p.add_argument("-d", "--dataset")
    p.add_argument("-o", "--out")
    p.add_argument("--filter", help="filter dataset ")
    p.add_argument("-t", "--timeout", type=int, default=30)
    p.add_argument("-n", "--n-chunks", type=int)
    p.add_argument("-b", "--chunk-size", type=int)
    args = p.parse_args()

    if args.mode == "eval":
        assert args.dataset and args.out and args.timeout
        df = pd.read_json(args.dataset, lines=True)
        util.incrementally_save_jsonl(
            quiet=True,
            filename=args.out,
            it=run_solns_and_tests(df, timeout=args.timeout),
        )
    elif args.mode == "process":
        assert args.dataset and args.out
        df = pd.read_json(args.dataset, lines=True)
        df = read_raw_dataset_to_solns_and_tests(df)
        df.to_json(args.out, orient="records", lines=True)
    elif args.mode == "split":
        assert args.dataset and args.out and ((args.n_chunks is not None) ^ (args.chunk_size is not None))
        assert args.chunk_size is None or args.chunk_size > 0
        assert args.n_chunks is None or args.n_chunks > 0
        df = pd.read_json(args.dataset, lines=True)
        for source, split in df.groupby("source"):
            n_rows = len(split)
            chunk_size = args.chunk_size if args.chunk_size else ceil(n_rows / args.n_chunks)
            for i in range(args.n_chunks):
                split[i * chunk_size:(i + 1) * chunk_size].to_json(
                    f"{args.out}/{source}/chunk-{i:04d}.jsonl",
                    orient="records",
                    lines=True
                )
    elif args.mode == "debug":
        assert args.dataset and args.out and args.timeout
        df = pd.read_json(args.dataset, lines=True)
        debug_segfaults(df, timeout=args.timeout, out=args.out)
    elif args.mode == "filter-hf":
        assert args.dataset and args.out
        df = pd.read_json(args.dataset, lines=True)
        df = long_to_wide(df)
        df = vet.filter_solutions(df)
        to_hf_dataset(df, args.out)
    else:
        raise ValueError(f"Missing mode: {args.mode}")


if __name__ == "__main__":
    # pandas print all columns
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    main()

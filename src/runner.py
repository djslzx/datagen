"""
Evaluate LLM-generated code and tests.
"""

import re
import json
from pprint import pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Dict, Generator

import execution as ex
import util


def make_programs(solns: List[str], tests: List[str]) -> Generator[Dict, None, None]:
    for soln in solns:
        for test in tests:
            test_names = find_tests(test)
            if not test_names:
                tester = "assert False, 'no tests found'"
            else:
                test_name = test_names[0]
                tester = f"assert {test_name}(), '{test_name} did not pass'"
            program = "\n\n".join([soln, test, tester])
            yield {
                "soln": soln,
                "test": test,
                "program": program,
            }


def eval_dataset(filename: str, n_samples: int, timeout: float) -> Generator[Dict, None, None]:
    df = pd.read_csv(filename, usecols=["id", "key", "value"])
    n_orig = len(df.groupby("id"))

    solns = (
        df[df["key"].str.startswith("solution ")]
        .groupby("id").agg({"value": list})
        .rename(columns={"value": "solutions"})
    )
    tests = (
        df[df["key"].str.startswith("test ")]
        .groupby("id").agg({"value": list})
        .rename(columns={"value": "tests"})
    )

    df = pd.concat([solns, tests], axis=1)
    df = df[(~df["tests"].isna()) & (~df["solutions"].isna())]

    print(f"Found {len(df)} rows from {n_orig} ({len(df)/n_orig * 100}% retained) with at least 1 soln/test")

    # split source file and local id from global id
    df["id"] = df.index
    df["source file"] = df["id"].apply(lambda s: s.split(":")[0])
    df["local id"] = df["id"].apply(lambda s: int(s.split(":")[1]))

    # take subset of rows of size n_samples
    if n_samples is None:
        view = df
        print(f"Keeping full dataset of size {len(df)}...")
    else:
        view = df.groupby("source file").sample(n=n_samples, replace=False)
        print(f"Sampled {n_samples} rows from each source file, yielding dataset of size {len(view)}...")

    print(view)

    # run soln/test pairs
    print("Running soln/test pairs...")
    for i, row in view.iterrows():
        solns = row["solutions"]
        tests = row["tests"]

        for d in make_programs(solns, tests):
            program = d["program"]
            yield {
                "id": row["id"],
                "source file": row["source file"],
                **ex.unsafe_check(program=program, timeout=timeout),
                "functions": find_fns(program),
                **d,
            }


def find_fns(text: str) -> List[str]:
    return re.findall(r"def (.+)\(.*\)", text)


def find_tests(text: str) -> List[str]:
    return re.findall(r"def (test_.+)\(.*\)", text)


if __name__ == "__main__":
    ts = util.timestamp()
    util.incrementally_save_jsonl(
        it=eval_dataset(
            filename="../datasets/wiz/solved-1k-2023-10-25T15:16:13.113059.csv",
            n_samples=None,
            timeout=10,
        ),
        filename=f"../datasets/wiz/evaluated-{ts}",
    )

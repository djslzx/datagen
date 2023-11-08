"""
Evaluate LLM-generated code and tests.
"""

import re
import pandas as pd
from typing import List, Optional, Dict, Generator

import execution as ex
import util
from dc import SolnTestPair, KVItem


def make_programs(solns: List[str], tests: List[str]) -> Generator[SolnTestPair, None, None]:
    """
    Given a list of solutions and a list of tests, produces programs that pair each
    solution with each test.  Yields a dictionary with keys "soln", "test", and
    "program".
    """
    for soln in solns:
        for test in tests:
            test_names = find_tests(test)
            if not test_names:
                tester = "assert False, 'no tests found'"
            else:
                test_name = test_names[0]
                tester = f"assert {test_name}(), '{test_name} did not pass'"
            program = "\n\n".join([soln, test, tester])
            yield SolnTestPair(soln=soln, test=test, program=program)


def eval_dataset(filename: str, n_samples_per_file: Optional[int], timeout: float) -> Generator[Dict, None, None]:
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

    print(f"Found {len(df)} rows from {n_orig} ({len(df) / n_orig * 100}% retained) with at least 1 soln/test")

    # split source file and local id from global id
    df["id"] = df.index

    # take subset of rows of size n_samples
    if n_samples_per_file is None:
        view = df
        print(f"Keeping full dataset of size {len(df)}...")
    else:
        df["source file"] = df["id"].apply(lambda s: s.split(":")[0])
        view = df.groupby("source file").sample(n=n_samples_per_file, replace=False)
        print(f"Sampled {n_samples_per_file} rows from each source file, yielding dataset of size {len(view)}...")

    # run soln/test pairs
    print("Running soln/test pairs...")
    for i, row in view.iterrows():
        solns = row["solutions"]
        tests = row["tests"]

        for d in make_programs(solns, tests):
            eval_out = ex.unsafe_check(program=d.program, timeout=timeout)
            out = {
                "passed": eval_out.passed,
                "result": eval_out.result,
                "test": d.test,
                "solution": d.soln,
                "functions": find_fns(d.program),
            }
            for item in KVItem.from_dict(out):
                yield item.to_dict()


def find_fns(text: str) -> List[str]:
    return re.findall(r"def (.+)\(.*\)", text)


def find_tests(text: str) -> List[str]:
    return re.findall(r"def (test_.+)\(.*\)", text)


if __name__ == "__main__":
    ts = util.timestamp()
    util.incrementally_save_jsonl(
        it=eval_dataset(
            filename="../datasets/wiz/solns-and-tests/all-solns-and-tests.jsonl",
            n_samples_per_file=1,
            timeout=10,
        ),
        filename=f"../datasets/wiz/evaluated-{ts}",
    )

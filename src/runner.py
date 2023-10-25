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
    def isnan(x):
        try:
            return np.isnan(x)
        except TypeError:
            return False

    df = pd.read_csv(filename)
    if n_samples is None:
        view = df
    else:
        indices = np.random.choice(len(df), size=n_samples, replace=False)
        view = df.iloc[indices]
    for i, row in view.iterrows():
        solns = [row[f"soln-{i}"] for i in range(3)]
        solns = [x for x in solns if x and not isnan(x)]

        tests = [util.strip_markdown(row[f"tests(text, soln_{i})"]) for i in range(3)]
        tests = [test for x in tests
                 if x and not isnan(x)
                 for test in util.split_tests(x)]

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
            filename="../datasets/sample-tests-2023-10-04T13:11:58.134555.csv",
            # "../datasets/sample-tests-2023-10-04T14:24:11.544966.csv"
            n_samples=None,
            timeout=10,
        ),
        filename=f"../datasets/evaluated-{ts}",
    )

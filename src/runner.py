"""

Evaluate LLM-generated code and tests.
"""

import re
from pprint import pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Dict, Generator
import multiprocessing as mp

import execution as ex
import util


def make_programs(solns: List[str], tests: List[str]) -> Generator[str, None, None]:
    n_tests = len(tests)
    main = "\n".join([
        "for test in [" + ", ".join([f"test_{i}_{j}"
                                     for i in range(1, n_tests + 1)
                                     for j in range(1, 6)]) +
        "]:",
        "    assert test(), f'{test.__name__} did not pass'", 
    ])
    for soln in solns:
        body = [t.replace("test_", f"test_{i}_")
                for i, t in enumerate(tests, 1)]
        program = "\n\n".join([soln, *body, main])
        yield program


def eval_dataset(filename: str, n_samples: int) -> Generator[Dict, None, None]:
    def isnan(x):
        try:
            return np.isnan(x)
        except TypeError:
            return False

    df = pd.read_csv(filename)
    indices = np.random.randint(low=0, high=len(df), size=n_samples)
    for i, row in df.iloc[indices].iterrows():
        solns = [row[f"soln-{i}"] for i in range(3)]
        tests = [util.strip_markdown(row[f"tests(text, soln_{i})"]) for i in range(3)]
        solns = [x for x in solns if x and not isnan(x)]
        tests = [x for x in tests if x and not isnan(x)]
        programs = make_programs(solns, tests)
        for program in programs:
            out = ex.unsafe_check(program=program, timeout=10)
            out["program"] = program
            out["functions"] = find_fns(program)
            yield out


def find_fns(text: str) -> List[str]:
    return re.findall(r"def (.+)\(.*\)", text)



if __name__ == "__main__":
    for d in eval_dataset(
        filename="../datasets/sample-tests-2023-10-04T13:11:58.134555.csv",  # "../datasets/sample-tests-2023-10-04T14:24:11.544966.csv"
        n_samples=10,
    ):
        del d['program']
        pp(d)

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

import execution as ex
import util


def make_programs(solns: List[str], tests: List[str]) -> Generator[str, None, None]:
    for soln in solns:
        for test in tests:
            test_names = find_tests(test)
            if not test_names:
                tester = "assert False, 'no tests found'"
            else:
                test_name = test_names[0]
                tester = f"assert {test_name}, '{test_name} did not pass'"
            program = "\n\n".join([soln, test, tester])
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
        solns = [x for x in solns if x and not isnan(x)]

        tests = [util.strip_markdown(row[f"tests(text, soln_{i})"]) for i in range(3)]
        tests = [test for x in tests
                 if x and not isnan(x)
                 for test in split_tests(x)]
        print(tests)

        for program in make_programs(solns, tests):
            out = ex.unsafe_check(program=program, timeout=10)
            out["row id"] = i
            out["program"] = program
            out["functions"] = find_fns(program)
            yield out


def find_fns(text: str) -> List[str]:
    return re.findall(r"def (.+)\(.*\)", text)


def find_tests(text: str) -> List[str]:
    return re.findall(r"def (test_.+)\(.*\)", text)


def split_tests(source_code):
    # split by decls
    blocks = ["def test_" + x
              for x in source_code.split("def test_")[1:]
              if x.strip()]
    out = []
    # only keep indented lines; stop at first non-indented line
    for block in blocks:
        lines = block.split("\n")
        block_text = lines[0] + "\n"
        for line in lines[1:]:
            if line.startswith("    "):
                block_text += line + "\n"
            else:
                break
        out.append(block_text)
    return out


if __name__ == "__main__":
    for d in eval_dataset(
            filename="../datasets/sample-tests-2023-10-04T13:11:58.134555.csv",
            # "../datasets/sample-tests-2023-10-04T14:24:11.544966.csv"
            n_samples=10,
    ):
        del d['program']
        pp(d)

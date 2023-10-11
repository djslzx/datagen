"""
Evaluate LLM-generated code and tests.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Dict
import multiprocessing as mp

import util


def run():
    pass


def make_programs(solns: List[str], tests: List[str]) -> List[str]:
    n_tests = len(tests)
    main = "\n".join([
        f"with open('/dev/shm/', 'w') as f:",
        f"  for test in [" + ", ".join([f"test_{i}_{j}"
                                        for i in range(n_tests)
                                        for j in range(5)]) + "]:",
        f"    try:",
        f"      out = test()",
        f"    except BaseException:",
        f"      f.write('0')",
        f"      continue",
        f"    f.write('1')",
    ])
    return [
        "\n\n".join([soln, *[t.replace("test_", f"test_{i}_")
                             for i, t in enumerate(tests)], main])
        for soln in solns
    ]


if __name__ == "__main__":
    # read file containing problems, solutions, and tests
    df = pd.read_csv("../datasets/sample-tests-2023-10-04T14:24:11.544966.csv")
    row = df.loc[1]
    solns = [row[f"soln-{i}"] for i in range(3)]
    tests = [util.strip_markdown(row[f"tests(text, soln_{i})"]) for i in range(3)]
    make_programs(solns, tests)

    # test_cols = [
    #     "tests(text, soln_0)", "tests(text, soln_1)", "tests(text, soln_2)"
    # ]

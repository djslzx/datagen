from __future__ import annotations
from typing import Iterator
import pandas as pd
from tqdm import tqdm

from finetune.root import util, evaluate


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

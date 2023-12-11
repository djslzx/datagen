"""
Evaluate LLM-generated code and tests.
"""

import re
import pandas as pd
from typing import List, Optional, Dict, Iterable, TypedDict, Tuple, Any
from dataclasses import dataclass

import execution
import util
from dc import KVItem


@dataclass
class SolnTestPair:
    soln: str
    test: str
    program: str
    id: Optional[str] = None


def make_programs(solns: List[str], tests: List[str]) -> Iterable[SolnTestPair]:
    """
    Given a list of solutions and a list of tests, produces programs that pair each
    solution with each test.  Yields a dictionary with keys "soln", "test", and
    "program".
    """
    preamble = "\n".join([
        "class TestFailed(Exception):",
        "    pass",
    ])
    for i, soln in enumerate(solns):
        for j, test in enumerate(tests):
            test_names = find_tests(test)
            if not test_names:
                tester = "raise TestFailed('No tests found')"
            else:
                test_name = test_names[0]
                tester = "\n".join([
                    f"if not {test_name}():",
                    f"    raise TestFailed('failed {test_name}')",
                ])
            program = "\n\n".join([preamble, soln, test, tester])
            yield SolnTestPair(id=f"{i}:{j}", soln=soln, test=test, program=program)


def run_tests(program: str, tests: List[str], timeout: float) -> Dict[str, Any]:
    n_passed = 0
    n_tries = 0
    for prog in make_programs([program], tests):
        out = execution.unsafe_check(program=prog.program, timeout=timeout)
        n_passed += int(out.passed)
        n_tries += 1
    return {
        "pass rate": n_passed / n_tries,
    }


def eval_dataset(filename: str, n_samples_per_file: Optional[int], timeout: float) -> Iterable[Dict]:
    df = pd.read_json(filename, lines=True)
    df = df[["id", "key", "value"]]
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
    for _, row in view.iterrows():
        id = row["id"]
        solns = row["solutions"]
        tests = row["tests"]

        for prog in make_programs(solns, tests):
            eval_out = execution.unsafe_check(program=prog.program, timeout=timeout)
            out = {
                "id": f"{id}:{prog.id}",
                "passed": eval_out.passed,
                "result": eval_out.result,
                "test": prog.test,
                "solution": prog.soln,
                "functions": find_fns(prog.program),
            }
            for item in KVItem.from_dict(out):
                yield item.to_dict()


def find_fns(text: str) -> List[str]:
    return re.findall(r"def (.+)\(.*\)", text)


def find_tests(text: str) -> List[str]:
    return re.findall(r"def (test.*)\(\)(?: -> .*)?:", text)


def test_find_tests():
    cases = [
        "def test(): return True", ["test"],
        "def test_1(): return True", ["test_1"],
        "def test_this(): return True", ["test_this"],
        "def test() -> bool: return True", ["test"],
        "def test_1() -> bool: return True", ["test_1"],
        "def test_that() -> bool: return True", ["test_that"],
    ]
    for x, y in zip(cases[::2], cases[1::2]):
        assert find_tests(x) == y


if __name__ == "__main__":
    ts = util.timestamp()
    util.incrementally_save_jsonl(
        it=eval_dataset(
            filename="../datasets/wiz/solved/all-solved-1k.jsonl",
            n_samples_per_file=None,
            timeout=30,
        ),
        filename=f"../datasets/wiz/evaluated-1k-{ts}",
    )

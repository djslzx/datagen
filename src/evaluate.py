"""
Evaluate LLM-generated code and tests.
"""

import re
import pandas as pd
from typing import List, Optional, Dict, Iterable, TypedDict, Tuple, Any, Iterator
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


def make_programs(solns: List[str], tests: List[str]) -> Iterator[SolnTestPair]:
    """
    Given a list of solutions and a list of tests, produces programs that pair each
    solution with each test.  Yields a dictionary with keys "soln", "test", and
    "program".
    """
    preamble = "\n".join([
        "from typing import List, Dict, Tuple, Set, Optional"
        "",
        "class TestFailed(Exception):",
        "    pass",
        "",
        "class MissingTests(Exception):",
        "    pass",
        "",
    ])
    for i, soln in enumerate(solns):
        for j, test in enumerate(tests):
            test_names = find_tests(test)
            if not test_names:
                tester = "raise MissingTests('No tests found')"
            else:
                test_name = test_names[0]
                tester = "\n".join([
                    f"if not {test_name}():",
                    f"    raise TestFailed('failed {test_name}')",
                ])
            program = "\n\n".join([preamble, soln, test, tester])
            yield SolnTestPair(id=f"{i}:{j}", soln=soln, test=test, program=program)


def make_program(soln: str, test: str) -> SolnTestPair:
    return next(make_programs([soln], [test]))


def run_solns_w_tests(solns: List[str], tests: List[str], timeout: float) -> List[execution.Result]:
    return [
        execution.unsafe_check(program=prog.program, timeout=timeout)
        for prog in make_programs(solns, tests)
    ]


def run_soln_w_test(soln: str, test: str, timeout: float) -> execution.Result:
    assert soln
    assert test
    prog = make_program(soln, test)
    return execution.unsafe_check(program=prog.program, timeout=timeout)


def eval_dataset(filename: str, timeout: float) -> Iterable[Dict]:
    df = pd.read_json(filename, lines=True)
    df = df[["id", "key", "value"]]
    n_orig = len(df.groupby("id"))

    problems = (
        df[df["key"] == "restyled problem"]
        .groupby("id").agg({"value": "first"})
        .rename(columns={"value": "problem"})
    )
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

    df = pd.concat([problems, solns, tests], axis=1)
    df = df[(~df["tests"].isna()) & (~df["solutions"].isna())]

    print(df)
    print(f"Found {len(df)} rows from {n_orig} ({len(df) / n_orig * 100}% retained) with at least 1 soln/test")

    # run soln/test pairs
    print("Running soln/test pairs...")
    for id, row in df.iterrows():
        problem = row["problem"],
        solns = row["solutions"]
        tests = row["tests"]

        for prog in make_programs(solns, tests):
            result = execution.unsafe_check(program=prog.program, timeout=timeout)
            out = {
                "id": f"{id}:{prog.id}",
                "problem": problem,
                "solution": prog.soln,
                "test": prog.test,
                "passed": result.passed,
                **result.to_dict(prefix="result."),
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
        filename=f"../datasets/wiz/all-evaluated-20k:30k-{ts}",
        it=eval_dataset(
            filename="../datasets/wiz/all-solved/all-solved-20k:30k.jsonl",
            timeout=30,
        ),
    )

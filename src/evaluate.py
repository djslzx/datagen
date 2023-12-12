"""
Evaluate LLM-generated code and tests.
"""

import re
import pandas as pd
from typing import List, Optional, Dict, Iterable, TypedDict, Tuple, Any, Iterator
from dataclasses import dataclass
import sys
import datasets

import execution
import util


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
        "from typing import List, Dict, Tuple, Set, Optional, Any"
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


def make_soln_program(soln: str) -> str:
    preamble = "\n".join([
        "from typing import List, Dict, Tuple, Set, Optional"
        "",
    ])
    return "\n\n".join([preamble, soln])


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


def run_soln(soln: str, timeout: float) -> execution.Result:
    assert soln
    p = make_soln_program(soln)
    return execution.unsafe_check(program=p, timeout=timeout)


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
        filename=f"../datasets/wiz/all-eval-1k-{ts}",
        # filename=f"../datasets/wiz/all-evaluated-20k:30k-{ts}",
        it=eval_dataset(
            filename="../datasets/wiz/solved/all-solved-1k.jsonl",
            # filename="../datasets/wiz/all-solved/all-solved-20k:30k.jsonl",
            timeout=10,
        ),
    )

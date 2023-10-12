import pytest
from execution import *


def test_should_pass():
    programs = [
        "print('hello world')",
        "x = 1; y = 2",
        "def f(x): return x + 1",  # functions
        "import itertools",  # innocuous std library import
        "if __name__ == '__main__': pass",
    ]
    for p in programs:
        out = unsafe_check(p, timeout=5)
        assert out["passed"], f"Expected program {p} to pass, but got {out}"


def test_should_fail():
    programs = [
        "import os",  # dangerous import
        "x + 1",  # undefined variable
        "assert False",  # obvious exception
    ]
    for p in programs:
        out = unsafe_check(p, timeout=5)
        assert not out["passed"], f"Expected program {p} to fail, but got {out}"

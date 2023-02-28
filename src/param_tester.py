from __future__ import annotations
from typing import *
import itertools as it


class ParamTester:
    """
    Streamline iteration through sets of parameter values.

    Given a set of parameters, each with a list of possible values,
    iterate through each of the possible combinations of parameters
    using the order of parameters given (so we can decide which ones
    we want to test first) and the order of values given for each
    parameter.
    """
    def __init__(self, params: Dict[str, List[Any]]):
        self.params = params

    def __iter__(self):
        # reverse so we get the right prioritization
        for combo in it.product(*reversed(self.params.values())):
            yield {
                k: v
                for k, v in zip(reversed(self.params.keys()), combo)
            }


def test_param_tester():
    p = ParamTester({"a": [1, 2],
                     "b": [0, 1, 2],
                     "c": [0]})
    configs = [
        {"a": 1, "b": 0, "c": 0},
        {"a": 2, "b": 0, "c": 0},
        {"a": 1, "b": 1, "c": 0},
        {"a": 2, "b": 1, "c": 0},
        {"a": 1, "b": 2, "c": 0},
        {"a": 2, "b": 2, "c": 0},
    ]
    for i, config in enumerate(p):
        assert configs[i] == config, f"Expected {configs[i]} on iter {i} but got {config}"


if __name__ == "__main__":
    test_param_tester()

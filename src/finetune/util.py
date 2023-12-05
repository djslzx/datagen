"""
Utils for model finetuning
"""
from datetime import datetime
from typing import List, Dict, Set, Tuple


def timestamp():
    return datetime.now().isoformat()


def split_by_percentages(xs: List, ps: Dict[str, float]) -> Dict[str, List]:
    """
    Given a list and dictionary of names/weights,
    split the list by weights and return the named splits.
    """
    assert abs(1 - sum(ps.values())) <= 1e-6, "Split percentages must add up to 1"
    outs = {}
    start = 0
    n = len(xs)
    for key, p in ps.items():
        step = int(p * n)
        outs[key] = xs[start:start + step]
        start += step
    return outs


def test_split_by_percentages():
    cases = [
        [1, 2, 3], {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
        {"a": [1], "b": [2], "c": [3]},
        [1] * 80 + [2] * 10 + [3] * 10, {"train": 0.8, "validate": 0.1, "test": 0.1},
        {"train": [1] * 80,
         "validate": [2] * 10,
         "test": [3] * 10},
    ]
    for xs, ps, y in zip(cases[0::3], cases[1::3], cases[2::3]):
        out = split_by_percentages(xs, ps)
        assert out == y, f"Expected {y} but got {out}"



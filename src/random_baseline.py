"""
Randomly sample programs from a uniform PCFG over L-systems
"""

from __future__ import annotations
from typing import *
from grammar import Grammar
import parse


def random_sample(n_samples: int) -> Generator[str]:
    g = Grammar.from_components(components=parse.rule_types, gram=2)
    g.normalize_()
    for _ in range(n_samples):
        ttree = g.sample("LSystem")
        s = parse.eval_ttree_as_str(ttree)
        yield s


if __name__ == "__main__":
    for x in random_sample(100):
        print(x)

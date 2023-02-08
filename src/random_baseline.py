"""
Randomly sample programs from a uniform PCFG over L-systems
"""

from __future__ import annotations
from typing import *
import multiprocessing as mp
from grammar import Grammar
import parse

# init metagrammar
MG = Grammar.from_components(components=parse.rule_types, gram=2)
MG.normalize_()


def sample_mg(i: int):
    ttree = MG.sample("LSystem")
    s = parse.eval_ttree_as_str(ttree)
    print(f"{i}: {s}")
    return s


if __name__ == '__main__':
    N_SAMPLES = 1000
    FILE = "randoms.txt"
    with mp.Pool(16) as pool, open(FILE, 'w') as f:
        for x in pool.imap(sample_mg, range(N_SAMPLES)):
            f.write(x + "\n")

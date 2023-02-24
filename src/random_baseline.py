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


def sample_mg() -> str:
    ttree = MG.sample("LSystem")
    s = parse.eval_ttree_as_str(ttree)
    return s


def take_samples_to_file(n_samples: int, out_file: str):
    with mp.Pool(16) as pool, open(out_file, 'w') as f:
        for x in pool.imap(lambda _: sample_mg(), range(n_samples)):
            f.write(x + "\n")


if __name__ == '__main__':
    N_SAMPLES = 100_000
    FILE = "../datasets/random_100k.txt"
    take_samples_to_file(N_SAMPLES, FILE)

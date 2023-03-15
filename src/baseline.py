"""
Randomly sample programs from a uniform PCFG over L-systems
"""

from __future__ import annotations
from typing import *
import multiprocessing as mp

from lang import Language
from lindenmayer import LSys, NilError
from regexpr import Regex
from zoo import zoo_strs

LEN_CAP = 100

# init lsystem metagrammar
lsys = LSys(45, 3, 3, 128, 128)
lsys_fitted = LSys(45, 3, 3, 128, 128)
lsys_fitted.fit([lsys_fitted.parse(s) for s in zoo_strs], alpha=0.01)

# init regex metagrammar
rgx = Regex()


def sample_lsys(i) -> str:
    while True:
        t = lsys_fitted.sample()
        if len(t) <= LEN_CAP:
            break
    s = lsys_fitted.to_str(t)
    print(s)
    return s


def sample_simplified_lsys(i) -> Optional[str]:
    s = lsys.sample()
    try:
        s = lsys.simplify(s)
        t = lsys.to_str(s)
        return t
    except NilError:
        return None


def sample_regex(i) -> str:
    while True:
        t = rgx.sample()
        if len(t) <= LEN_CAP:
            break
    s = rgx.to_str(t)
    print(s)
    return s


def sample_to_file(sampler: Callable, n_samples: int, out_file: str):
    with mp.Pool(16) as pool, open(out_file, 'w') as f:
        for x in pool.imap(sampler, range(n_samples)):
            f.write(x + "\n")


if __name__ == '__main__':
    N_SAMPLES = 100_000
    FILE = "../datasets/regex/random/random_100k.txt"
    sample_to_file(sample_regex, N_SAMPLES, FILE)

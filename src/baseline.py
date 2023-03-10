"""
Randomly sample programs from a uniform PCFG over L-systems
"""

from __future__ import annotations
from typing import *
import multiprocessing as mp

from lindenmayer import LSys, NilError
from zoo import zoo_strs

# init metagrammar
lsys = LSys(45, 3, 3, 128, 128)
lsys_fitted = LSys(45, 3, 3, 128, 128)
lsys_fitted.fit([lsys_fitted.parse(s) for s in zoo_strs], alpha=0.01)
LEN_CAP = 100


def sample(i) -> str:
    t = lsys.sample()
    s = lsys.to_str(t)
    print(s)
    return s


def sample_fitted_capped(i) -> str:
    while True:
        t = lsys_fitted.sample()
        if len(t) <= LEN_CAP:
            break
    s = lsys_fitted.to_str(t)
    print(s)
    return s


def sample_simplify(i) -> Optional[str]:
    s = lsys.sample()
    try:
        s = lsys.simplify(s)
        t = lsys.to_str(s)
        return t
    except NilError:
        return None


def take_samples_to_file(n_samples: int, out_file: str):
    with mp.Pool(16) as pool, open(out_file, 'w') as f:
        for x in pool.imap(sample_fitted_capped, range(n_samples)):
            f.write(x + "\n")


if __name__ == '__main__':
    N_SAMPLES = 100_000
    FILE = "../datasets/random/fitted_100k.txt"
    take_samples_to_file(N_SAMPLES, FILE)

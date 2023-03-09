"""
Randomly sample programs from a uniform PCFG over L-systems
"""

from __future__ import annotations
from typing import *
import multiprocessing as mp

from lindenmayer import LSys, NilError

# init metagrammar
lsys = LSys(90, 3, 3, 128, 128)


def sample(i) -> str:
    t = lsys.sample()
    s = lsys.to_str(t)
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
        for x in pool.imap(sample, range(n_samples)):
            f.write(x + "\n")


if __name__ == '__main__':
    N_SAMPLES = 100_000
    FILE = "../datasets/random/random_100k_test.txt"
    take_samples_to_file(N_SAMPLES, FILE)

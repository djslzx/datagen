"""
Randomly sample programs from a uniform PCFG over L-systems
"""

from __future__ import annotations
from typing import *
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

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


def take_samples(n_samples: int, out_file: str):
    with mp.Pool(16) as pool, open(out_file, 'w') as f:
        for x in pool.imap(sample_mg, range(n_samples)):
            f.write(x + "\n")


def simplify_file(in_file: str, out_file: str):
    with open(in_file, 'r') as f_in, \
         open(out_file, 'w') as f_out:
        n_in = 0
        n_out = 0
        for i, line in enumerate(f_in.readlines()):
            n_in += 1
            try:
                s = parse.simplify(line.strip())
                print(f"{i}: {s}")
                f_out.write(s + "\n")
                n_out += 1
            except parse.ParseError:
                print(f"Skipping line {i}")
                f_out.write("\n")
    print(f"Wrote {n_out} of {n_in} lines.")


def check_compression(in_file: str, out_file: str):
    mat = np.empty((N_SAMPLES, 2), dtype=int)  # in-file nlines, out-file nlines
    with open(in_file, 'r') as f_in, open(out_file, 'r') as f_out:
        for i, line in enumerate(f_in.readlines()):
            mat[i, 0] = len(line)
        for i, line in enumerate(f_out.readlines()):
            mat[i, 1] = len(line)

    # print n_lines stats
    print(f"in_file mean: {np.mean(mat[:, 0])}, "
          f"std dev: {np.std(mat[:, 0])}, "
          f"out_file mean: {np.mean(mat[:, 1])}, "
          f"std dev: {np.std(mat[:, 1])}, ")

    # print compression ratio
    compression = mat[:, 1] / mat[:, 0]
    print(f"compression mean: {np.mean(compression, 0)}, "
          f"std dev: {np.std(compression, 0)}")

    # plt.plot(mat, label=("in", "out"))
    print(mat)
    print(compression)
    plt.scatter(np.arange(0, N_SAMPLES), compression)
    # plt.plot(compression)
    plt.show()


if __name__ == '__main__':
    N_SAMPLES = 100_000
    IN_FILE = "../datasets/random_100k.txt"
    OUT_FILE = "../datasets/random_100k_simpl.txt"

    # simplify_file(IN_FILE, "../datasets/random_100k_simplified.txt")
    check_compression(IN_FILE, OUT_FILE)

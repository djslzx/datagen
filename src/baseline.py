"""
Randomly sample programs from a uniform PCFG over L-systems
"""

from __future__ import annotations

import random
from typing import List, Callable, Optional, Any, Dict
import multiprocessing as mp
import lark.exceptions
import numpy as np
from scipy.spatial import distance
from sklearn import manifold
import itertools as it
import pandas as pd
from tqdm import tqdm
import pickle
import sys

from featurizers import ResnetFeaturizer
from lang.tree import Language, Tree
from lang.lindenmayer import LSys, NilError
from lang.regexpr import Regex
import examples
import util

LEN_CAP = 100

# init lsystem metagrammar
lsys = LSys(kind="deterministic", featurizer=ResnetFeaturizer(), step_length=3, render_depth=3)
lsys_fitted = LSys(kind="deterministic", featurizer=ResnetFeaturizer(), step_length=3, render_depth=3)
lsys_fitted.fit([lsys_fitted.parse(s) for s in examples.lsystem_book_det_examples], alpha=0.01)


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


rgx = Regex()
def sample_regex_uniform(i) -> str:
    while True:
        t = rgx.sample()
        if len(t) <= LEN_CAP:
            break
    s = rgx.to_str(t)
    print(s)
    return s


rgx_fit = Regex()
corpus = [x for key in ["text enums", "text", "text and nums"]
            for x in examples.regex_split[key]]
programs = [rgx_fit.parse(x) for x in corpus]
rgx_fit.fit(programs, alpha=1e-4)
def sample_regex_fitted(i) -> str:
    while True:
        t = rgx_fit.sample()
        if len(t) <= LEN_CAP:
            break
    s = rgx_fit.to_str(t)
    print(s)
    return s


def sample_to_file(sampler: Callable[[int], Any], n_samples: int, out_file: str):
    with mp.Pool(16) as pool, open(out_file, 'w') as f:
        for x in pool.imap(sampler, range(n_samples)):
            f.write(x + "\n")


def generate_regexs():
    N_SAMPLES = 100_000
    # FILE = "../datasets/lsystems/random/100cap_fitted_100k.txt"
    sample_to_file(sample_regex_uniform, N_SAMPLES, "../datasets/regex/uniform_100k.txt")
    sample_to_file(sample_regex_fitted, N_SAMPLES, "../datasets/regex/fitted_100k.txt")
    with open("../datasets/regex/train.txt", "w") as f:
        for x in corpus:
            f.write(x + "\n")


def write_histograms(trees: Dict[str, List], filename: str, n_renders: int):
    # fixme: this is broken, but kept for documentation purposes
    """
    For each pair of training datasets, plot pairwise distances in feature space between instances
    in the datasets.
    """
    datasets = {
        "train": "../datasets/regex/train.txt",
        "ns": "../datasets/regex/ns.txt",
        "pcfg": "../datasets/regex/fitted_100k.txt",
        "uniform": "../datasets/regex/uniform_100k.txt",
        "test": "../datasets/regex/num.txt",
    }
    lang = Regex()
    features = {}
    for name, ts in trees.items():
        print(f"Computing features for {name}", file=sys.stderr)
        features[name] = [
            lang.featurizer.apply([lang.eval(t, env={}) for _ in range(n_renders)])
            for t in tqdm(ts)
        ]
    with open(filename, "wb") as f:
        for name in datasets.keys() - {"test"}:
            # pairwise Hausdorff distances between feature sets
            print(f"Computing distances between feature sets for {name}", file=sys.stderr)
            for (t1, f1), (t2, f2) in tqdm(it.product(zip(trees[name], features[name]),
                                                      zip(trees["test"], features["test"]))):
                d, _, _ = distance.directed_hausdorff(f1, f2)
                row = (d, lang.to_str(t1), lang.to_str(t2), name)
                pickle.dump(row, f)
    # cols: [dist, src, dst, key]


def prune_table(filename_in: str, filename_out: str, N: int):
    with open(filename_in, "rb") as f_in, \
         open(filename_out, "wb") as f_out:
        counts = {}
        while True:
            try:
                row = pickle.load(f_in)
                name = row[-1]
                counts[name] = counts.get(name, 0) + 1
                if counts[name] <= N:
                    pickle.dump(row, f_out)
            except EOFError:
                break


def dissimilarity_matrix(df: pd.DataFrame, key: str) -> np.ndarray:
    # dist, src, dst, key
    df = df.loc[df["key"] == key]
    points = np.concatenate((df["src"].unique(), df["dst"].unique()))
    n = len(points)

    # row: dist, src, dst, key => mat: [src + dst, src + dst]
    mat = np.zeros((n, n))
    index_map = {point: i for i, point in enumerate(points)}

    def f(row: pd.Series):
        i, j = index_map[row['src']], index_map[row['dst']]
        mat[i, j] = row['dist']
        mat[j, i] = row['dist']

    df.apply(f, axis=1)
    return mat


def mds(mat: np.ndarray) -> np.ndarray:
    m = manifold.MDS(dissimilarity="precomputed")
    return m.fit_transform(mat)


def random_lsystem(L: LSys, length: int) -> Tree:
    tokens = ["F", "+", "-"]
    weights = [0.6, 0.2, 0.2]
    s = "90;F;F~" + "".join(random.choices(tokens, weights=weights, k=length))
    return L.parse(s)


if __name__ == '__main__':
    L = LSys(kind="deterministic", featurizer=ResnetFeaturizer(), step_length=3, render_depth=3)
    for _ in range(3):
        imgs = [L.eval(random_lsystem(L, length=100)) for _ in range(100)]
        util.plot(imgs, shape=(10, 10))

    # write_histograms("../out/plots/hists/distances_100k_n=100.dat", n_renders=100)
    # prune_table("../out/plots/hists/distances_100k.dat",
    #             "../out/plots/hists/distances_37k.dat",
    #             N=37_000)
    # cols = ["dist", "src", "dst", "key"]
    # data = []
    # with open("../out/plots/hists/distances_100k.dat", "rb") as f:
    #     while True:
    #         try:
    #             row = pickle.load(f)
    #             data.append(row)
    #         except EOFError:
    #             break
    #
    # df = pd.DataFrame(data, columns=cols)
    # print(df["key"].value_counts())
    # print(df)
    #
    # mat = dissimilarity_matrix(df, key="train")
    # print(mat)
    # print(mat.shape)
    #
    # m = mds(mat)
    # print(m)
    # print(m.shape)
    # mdf = pd.DataFrame(m, columns=["x", "y"])
    # print(mdf)
    #
    # sns.relplot(data=mdf, x="x", y="y", hue="key")
    # plt.show()

    # sns.displot(df, x="dist", col="key", stat="density", common_norm=False)
    # plt.show()

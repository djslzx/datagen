"""
ns without the evo
"""

from typing import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from einops import rearrange, reduce
from pprint import pp

from lang import Language, Tree
import regexpr
import util

def features(L: Language, s: Collection[Tree], n_samples: int) -> np.ndarray:
    # output shape: (|s|, n)
    xs = []
    for x in s:
        if x is not None:
            outputs = [L.eval(x, env={}) for _ in range(n_samples)]
            xs.append(L.featurizer.apply(outputs))
    out = np.array(xs)
    assert out.shape[:2] == (len(s), n_samples), f"Expected shape {(len(s), n_samples)}, but got {out.shape}"
    return out

def best(L: Language, xs: Collection[Tree], fs: np.ndarray, knn: NearestNeighbors) -> Tuple[Tree, float]:
    best_score = 0
    best_tree = None
    for x, f in zip(xs, fs):
        dists, _ = knn.kneighbors(rearrange(f, "n -> 1 n"))
        score = reduce(dists, "1 n_neighbors -> 1", reduction='sum')
        if score > best_score or best_tree is None:
            best_score = score
            best_tree = x
    return best_tree, best_score

def chamfer_dist(k: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns chamfer distance on two 1D vectors, where the vectors are interpreted as the concatenation
    of k n-dimensional vectors.
    """
    def d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        X = rearrange(v1, "(k n) -> k n", k=k)
        Y = rearrange(v2, "(k n) -> k n", k=k)
        return (sum(min(np.dot(x - y, x - y) for y in Y) for x in X) +
                sum(min(np.dot(x - y, x - y) for x in X) for y in Y))
    return d

def search(L: Language, init_popn: List[Tree],
           samples_per_program: int,
           samples_per_iter: int,
           keep_per_iter: int,
           alpha: float,
           iters: int):
    archive = init_popn
    knn = NearestNeighbors(metric=chamfer_dist(samples_per_program))
    for i in range(iters):
        with util.Timing(f"iter {i}", suppress_start=True):
            # sample from fitted grammar
            L.fit(archive, alpha=alpha)
            samples = np.array([L.sample() for _ in range(samples_per_iter)], dtype=object)

            # extract features
            e_archive = features(L, archive, samples_per_program)
            e_archive = rearrange(e_archive, "n samples features -> n (samples features)")
            knn.fit(e_archive)
            e_samples = features(L, samples, samples_per_program)
            e_samples = rearrange(e_samples, "n samples features -> n (samples features)")

            # choose which samples to add to archive: score all samples, then choose best few
            dists, _ = knn.kneighbors(e_samples)
            dists = np.sum(dists, axis=1)
            idx = np.argsort(-dists)  # sort descending
            keep = samples[idx][:keep_per_iter]
            archive.extend(keep)

            # diagnostics
            print(f"Adding {keep} w/ scores {dists[:keep_per_iter]}")

    return archive


if __name__ == "__main__":
    lang = regexpr.Regex()
    s0 = [lang.parse(".")] * 10
    # s0 = [lang.sample() for _ in range(10)]
    p = search(
        lang,
        init_popn=s0,
        samples_per_program=100,
        samples_per_iter=10,
        keep_per_iter=1,
        iters=100,
        alpha=1e-4,
    )
    pp(p)

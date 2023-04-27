"""
ns without the evo
"""

from typing import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski, directed_hausdorff
from scipy.special import softmax
from einops import rearrange, reduce
from pprint import pp
from time import time

from lang import Language, Tree
import lindenmayer
import regexpr
import examples
import util

Distance = Callable[[np.ndarray, np.ndarray], float]

def features(L: Language, s: Collection[Tree], n_samples: int) -> np.ndarray:
    # output shape: (|s|, n)
    xs = []
    for x in s:
        if x is not None:
            outputs = [L.eval(x, env={}) for _ in range(n_samples)]
            xs.append(L.featurizer.apply(outputs))
    out = np.array(xs)
    assert out.shape[:2] == (len(s), n_samples), \
        f"Expected shape {(len(s), n_samples)}, but got {out.shape}"
    return out

def make_dist(d: Distance, k: int) -> Distance:
    """
    Takes a distance that operates on sets and makes it operate on 1D vectors,
    where the vectors are interpreted as the concatenation of a set of k n-dimensional vectors.
    """
    def dist(v1: np.ndarray, v2: np.ndarray) -> float:
        X = rearrange(v1, "(k n) -> k n", k=k)
        Y = rearrange(v2, "(k n) -> k n", k=k)
        return d(X, Y)
    return dist

def chamfer(X: np.ndarray, Y: np.ndarray) -> float:
    return (sum(min(np.dot(x - y, x - y) for y in Y) for x in X) +
            sum(min(np.dot(x - y, x - y) for x in X) for y in Y))

def hausdorff(X: np.ndarray, Y: np.ndarray) -> float:
    return directed_hausdorff(X, Y)[0]

def search(L: Language,
           init_popn: List[Tree],
           d: Distance,
           select: str,
           samples_per_program: int,
           samples_per_iter: int,
           keep_per_iter: int,
           alpha: float,
           iters: int,
           save_to: str,
           debug: bool):
    assert select in {"absolute", "weighted"}
    with open(save_to, "w") as f:
        f.write(f"Params: {locals()}\n")

    archive = init_popn
    knn = NearestNeighbors(metric=make_dist(d=d, k=samples_per_program))
    for i in range(iters):
        with util.Timing(f"iter {i}", suppress_start=True):
            # sample from fitted grammar
            L.fit(archive, alpha=alpha)
            samples = np.array([L.sample() for _ in range(samples_per_iter)], dtype=object)

            # extract features
            e_archive = features(L, archive, samples_per_program)
            e_archive = rearrange(e_archive, "n samples features -> n (samples features)")
            e_samples = features(L, samples, samples_per_program)
            e_samples = rearrange(e_samples, "n samples features -> n (samples features)")

            # choose which samples to add to archive: score all samples, then choose best few
            knn.fit(e_archive)
            dists, _ = knn.kneighbors(e_samples)
            dists = np.sum(dists, axis=1)
            if select == "absolute":
                idx = np.argsort(-dists)[:keep_per_iter]  # sort descending
            elif select == "weighted":
                idx = np.random.choice(samples_per_iter, keep_per_iter,
                                       replace=False, p=softmax(dists))
                # random.multinomial(n=keep_per_iter, pvals=softmax(dists))
            keep = samples[idx]
            archive.extend(keep)

            # diagnostics
            if debug:
                print(f"Generation {i}:")
                for j, x in enumerate(keep):
                    print(f"  {L.to_str(x)}: {dists[idx][j]}")

            # save
            with open(save_to, "w") as f:
                for x in keep:
                    f.write(f"{L.to_str(x)}\n")

    return archive


if __name__ == "__main__":
    TRAIN = ["text enums", "text", "text and nums"]
    id = int(time())

    lang = regexpr.Regex()
    train_data = []
    for key in TRAIN:
        for s in examples.regex_split[key]:
            train_data += [lang.parse(s)]

    # lang = lindenmayer.LSys(theta=45, step_length=3, render_depth=3, n_rows=128, n_cols=128)
    pt = util.ParamTester({
        "L": lang,
        "init_popn": [train_data],
        "d": [chamfer, hausdorff],
        "select": ["absolute", "weighted"],
        "samples_per_program": 1,
        "samples_per_iter": 2,
        "keep_per_iter": 1,
        "iters": 10,
        "alpha": 1e-2,
        "debug": True,
    })
    for i, params in enumerate(pt):
        dist = params["d"].__name__
        select = params["select"]
        save_to = f"../out/simple_ns/{id}-{dist}-{select}.out"
        params.update({
            "save_to": save_to
        })
        print(f"Searching with id={id}, dist={dist}, select={select}")
        search(**params)

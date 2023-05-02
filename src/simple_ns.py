"""
ns without the evo
"""

from typing import List, Tuple, Callable, Collection
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski, directed_hausdorff
from scipy.special import softmax
from einops import rearrange, reduce
from pprint import pp
from time import time
import wandb

from lang import Language, Tree
import lindenmayer
import regexpr
import examples
import util

Distance = Callable[[np.ndarray, np.ndarray], float]

def features(L: Language, s: Collection[Tree], n_samples: int) -> np.ndarray:
    return batched_features(L, s, batch_size=64, n_samples=n_samples)
    # # output shape: (|s|, n)
    # xs = []
    # for x in s:
    #     if x is not None:
    #         outputs = [L.eval(x, env={}) for _ in range(n_samples)]
    #         xs.append(L.featurizer.apply(outputs))
    # out = np.array(xs)
    # assert out.shape[:2] == (len(s), n_samples), \
    #     f"Expected shape {(len(s), n_samples)}, but got {out.shape}"
    # return rearrange(out, "n samples features -> n (samples features)")

def batched_features(L: Language, S: Collection[Tree],
                     batch_size: int, n_samples: int) -> np.ndarray:
    # take samples from programs in S, then batch them and feed them through
    # the feature extractor for L
    def samples():
        for x in S:
            for _ in range(n_samples):
                yield L.eval(x, env={})

    ys = []
    for batch in util.batched(samples(), batch_size=batch_size):
        y = L.featurizer.apply(batch)
        ys.extend(y)
    # output shape: (|S|, n_samples, features)
    out = np.array(ys)
    assert out.shape[0] == (len(S) * n_samples), \
        f"Expected to get {len(S)} * {n_samples} = {len(S) * n_samples} feature vectors, but got out:{out.shape}"
    out = rearrange(out, "(s samples) features -> s (samples features)", s=len(S), samples=n_samples)
    return out

def take_samples(L: Language, n_samples: int, len_cap: int) -> np.ndarray:
    out = []
    while len(out) < n_samples:
        x = L.sample()
        if len(x) <= len_cap:
            out.append(x)
        # log failures?
    return np.array(out, dtype=object)

def select_indices(kind: str, dists: np.ndarray, n: int):
    if kind == "strict":
        return np.argsort(-dists)[:n]  # sort descending
    elif kind == "weighted":
        return np.random.choice(len(dists), n, replace=False, p=softmax(dists))

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
    # todo: make this much faster
    #  min_y(d(x, y)) can be performed using knn data structure
    return (sum(min(np.dot(x - y, x - y) for y in Y) for x in X) +
            sum(min(np.dot(x - y, x - y) for x in X) for y in Y))

def hausdorff(X: np.ndarray, Y: np.ndarray) -> float:
    return directed_hausdorff(X, Y)[0]

def simple_search(L: Language,
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
    assert select in {"strict", "weighted"}
    with open(save_to, "w") as f:
        f.write(f"# Params: {locals()}\n")

    archive = init_popn
    e_archive = features(L, archive, samples_per_program)
    metric = make_dist(d=d, k=samples_per_program) if samples_per_program > 1 else "minkowski"
    knn = NearestNeighbors(metric=metric)
    for t in range(iters):
        with util.Timing(f"iter {t}", suppress_start=True):
            # sample from fitted grammar
            with util.Timing("fit and sample"):
                L.fit(archive, alpha=alpha)
                samples = take_samples(L, samples_per_iter, len_cap=1000)

            # extract features
            with util.Timing("features"):
                e_samples = features(L, samples, samples_per_program)

            # choose which samples to add to archive: score all samples, then choose best few
            with util.Timing("score"):
                knn.fit(e_archive)
                dists, _ = knn.kneighbors(e_samples)
                dists = np.sum(dists, axis=1)
                idx = select_indices(kind=select, dists=dists, n=keep_per_iter)
                keep = samples[idx]
                archive.extend(keep)
                e_archive = np.concatenate((e_archive, e_samples[idx]), axis=0)

            # diagnostics
            if debug:
                print(f"Generation {t}:")
                for j, x in enumerate(keep):
                    print(f"  {L.to_str(x)}: {dists[idx][j]}")

            # save
            with open(save_to, "a") as f:
                for x in keep:
                    f.write(f"{L.to_str(x)}\n")

    return archive


def evo_search(L: Language,
               init_popn: List[Tree],
               d: Distance,
               select: str,
               max_popn_size: int,
               samples_per_program: int,
               samples_per_iter: int,
               keep_per_iter: int,
               alpha: float,
               iters: int,
               save_to: str,
               debug: bool) -> Tuple[List[Tree], List[Tree]]:
    assert samples_per_iter >= 2 * max_popn_size, \
        "Number of samples taken should be significantly larger than number of samples kept"
    assert len(init_popn) >= 5, \
        f"Initial population ({len(init_popn)}) must be geq number of nearest neighbors (5)"

    def embed(S): return features(L, S, samples_per_program)

    full_archive = []
    archive = []
    popn = init_popn
    e_archive = []
    e_popn = embed(popn)
    metric = make_dist(d=d, k=samples_per_program) if samples_per_program > 1 else "minkowski"
    knn = NearestNeighbors(metric=metric)
    for t in range(iters):
        with util.Timing(f"Iteration {t}"):
            # fit and sample
            L.fit(popn, alpha=alpha)
            samples = take_samples(L, samples_per_iter, len_cap=1000)
            with util.Timing("embedding samples"):
                e_samples = embed(samples)

            # score samples wrt archive + popn
            knn.fit(np.concatenate((e_archive, e_popn), axis=0) if archive else e_popn)
            dists, _ = knn.kneighbors(e_samples)
            dists = np.sum(dists, axis=1)

            # select samples to carry over to next generation
            i_popn = select_indices(select, dists, max_popn_size)
            popn = samples[i_popn]
            e_popn = e_samples[i_popn]
            full_archive.extend(popn)

            # archive random subset
            i_archive = np.random.choice(samples_per_iter, size=keep_per_iter, replace=False)
            archive.extend(samples[i_archive])
            e_archive.extend(e_samples[i_archive])

        # diagnostics
        # log top k images
        log = {
            f"top-{k}": wandb.Image(
                rearrange(lang.eval(x), "color row col -> row col color"),
                caption=lang.to_str(x),
            )
            for k, x in enumerate(popn[:5])
        }
        log.update({
            "scores": wandb.Histogram(dists[i_popn])
        })
        wandb.log(log)

        if debug:
            print(f"Generation {t}:")
            for j, x in enumerate(popn):
                print(f"  {L.to_str(x)}: {dists[i_popn][j]}")
            #
            # # plot top k
            # for batch in util.batched(popn, batch_size=36):
            #     bmps = [lang.eval(x) for x in batch]
            #     util.plot(bmps)

        # save
        with open(save_to, "a") as f:
            for x in popn:
                f.write(f"{L.to_str(x)}\n")

    return archive, full_archive


if __name__ == "__main__":
    TRAIN = ["text enums", "text", "text and nums"]
    id = int(time())
    # lang = regexpr.Regex()
    # train_data = []
    # for key in TRAIN:
    #     for s in examples.regex_split[key]:
    #         train_data += [lang.parse(s)]
    lang = lindenmayer.DeterministicLSystem(
        theta=30,
        step_length=3,
        render_depth=5,
        n_rows=128,
        n_cols=128,
        quantize=False,
    )
    train_data = [
        lang.parse("F;F~F"),
        lang.parse("F;F~FF"),
        lang.parse("F[+F][-F]FF;F~FF"),
        lang.parse("F+F-F;F~F+FF"),
        lang.parse("F;F~F[+F][-F]F"),
    ]

    # lang = lindenmayer.LSys(theta=45, step_length=3, render_depth=3, n_rows=128, n_cols=128)
    pt = util.ParamTester({
        "L": lang,
        "init_popn": [train_data],
        "d": [hausdorff],
        "select": ["strict"],
        "samples_per_program": 1,
        "samples_per_iter": 20,
        "max_popn_size": 10,
        "keep_per_iter": 2,
        "iters": 1,
        "alpha": 1,
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
        wandb.init(
            project="novelty",
            config=params,
        )
        A, FA = evo_search(**params)
        # bmps = []
        # for t in FA:
        #     bmp = lang.eval(t).astype(float)
        #     bmps.append(bmp)
        # util.plot(bmps)

"""
ns without the evo
"""
from math import ceil
from sys import stderr
from typing import List, Tuple, Callable, Collection
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import directed_hausdorff
from scipy.special import softmax
from scipy.ndimage import gaussian_filter
from einops import rearrange, reduce
from time import time
import wandb
from tqdm import tqdm

from lang import Language, Tree
import point
import lindenmayer
import regexpr
import examples
import util

Distance = Callable[[np.ndarray, np.ndarray], float]

def features(L: Language, s: Collection[Tree], n_samples: int, batch_size=4, gaussian_blur=False) -> np.ndarray:
    return batched_features(L, s, batch_size=batch_size, n_samples=n_samples, gaussian_blur=gaussian_blur)
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
                     batch_size: int, n_samples: int,
                     gaussian_blur=False) -> np.ndarray:
    # take samples from programs in S, then batch them and feed them through
    # the feature extractor for L
    def samples():
        for x in S:
            for _ in range(n_samples):
                bmp = L.eval(x, env={})
                if gaussian_blur:
                    bmp = gaussian_filter(bmp, sigma=3)
                yield bmp
    ys = []
    n_batches = ceil(len(S) * n_samples / batch_size)
    for batch in tqdm(util.batched(samples(), batch_size=batch_size), total=n_batches):
        print(f"[/{n_batches}]\r", file=stderr)
        y = L.featurizer.apply(batch)
        if batch_size > 1 and len(batch) > 1:
            ys.extend(y)
        else:
            ys.append(y)
    out = np.array(ys)
    # output shape: (|S|, n_samples, features)
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
                samples = take_samples(L, samples_per_iter, len_cap=100)

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
               debug=False,
               archive_early=False,
               gaussian_blur=False) -> Tuple[List[Tree], List[Tree]]:
    assert samples_per_iter >= 2 * max_popn_size, \
        "Number of samples taken should be significantly larger than number of samples kept"
    assert len(init_popn) >= 5, \
        f"Initial population ({len(init_popn)}) must be geq number of nearest neighbors (5)"

    def embed(S): return features(L, S, n_samples=samples_per_program, batch_size=8, gaussian_blur=gaussian_blur)
    def update_archive(A, E_A, S, E_S):
        I_A = np.random.choice(samples_per_iter, size=keep_per_iter, replace=False)
        A.extend(S[I_A])
        E_A.extend(E_S[I_A])

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
            samples = take_samples(L, samples_per_iter, len_cap=100)
            with util.Timing("embedding samples"):
                e_samples = embed(samples)

            if archive_early: update_archive(archive, e_archive, samples, e_samples)

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
            if not archive_early: update_archive(archive, e_archive, samples, e_samples)

        # diagnostics
        log = {"scores": wandb.Histogram(dists[i_popn])}
        # log best and worst images
        if isinstance(L, lindenmayer.LSys):
            def summarize(indices):
                img = rearrange([L.eval(x) for x in samples[indices]], "b color row col -> row (b col) color")
                caption = "Left to right: " + ", ".join(f"{L.to_str(x)} ({score:.4e})"
                                                        for x, score in zip(samples[indices], dists[indices]))
                return wandb.Image(img, caption=caption)

            i_best = np.argsort(-dists)[:5]
            i_worst = np.argsort(dists)[:5]
            log.update({
                "best": summarize(i_best),
                "worst": summarize(i_worst),
            })
        wandb.log(log)

        if debug:
            print(f"Generation {t}:")
            for j, x in enumerate(popn):
                print(f"  {L.to_str(x)}: {dists[i_popn][j]}")

        # save
        with open(save_to, "a") as f:
            f.write(f"# Generation {t}:\n")
            for x in popn:
                f.write(f"{L.to_str(x)}\n")

    return archive, full_archive

def DEFAULT_CONFIG():
    return {
        "d": hausdorff,
        "select": "strict",
        "samples_per_program": 1,
        "samples_per_iter": 1000,
        "max_popn_size": 100,
        "keep_per_iter": 10,
        "iters": 20,
        "alpha": 1,
        "debug": True,
        "gaussian_blur": False,
    }

def run_on_real_points():
    lang = point.RealPoint()
    train_data = [
        lang.parse("(0, 0)"),
        lang.parse("(1, 0)"),
        lang.parse("(0, 1)"),
        lang.parse("(-1, 0)"),
        lang.parse("(0, -1)"),
    ]
    config = DEFAULT_CONFIG()
    config.update({
        "L": lang,
        "init_popn": train_data,
        "save_to": f"../out/simple_ns/{id}-r2-strict.out",
    })
    wandb.init(project="novelty", config=config)
    evo_search(**config)

def run_on_nat_points(id: str):
    lang = point.NatPoint()
    train_data = [
        lang.parse("(one, one)"),
        lang.parse("(inc one, one)"),
        lang.parse("(one, inc one)"),
        lang.parse("(inc one, inc one)"),
        lang.parse("(inc inc one, one)"),
    ]
    config = DEFAULT_CONFIG()
    config.update({
        "L": lang,
        "init_popn": train_data,
        "save_to": f"../out/simple_ns/{id}-z2-strict.out",
    })
    wandb.init(project="novelty", config=config)
    evo_search(**config)

def run_on_lsystems(id):
    lang = lindenmayer.LSys(
        kind="deterministic",
        theta=30,
        step_length=4,
        render_depth=3,
        n_rows=128,
        n_cols=128,
        quantize=False,
        disable_last_layer=False,
        softmax_outputs=True,
    )
    train_data = [
        lang.parse("F;F~F"),
        lang.parse("F;F~FF"),
        lang.parse("F[+F][-F]FF;F~FF"),
        lang.parse("F+F-F;F~F+FF"),
        lang.parse("F;F~F[+F][-F]F"),
    ]
    config = DEFAULT_CONFIG()
    config.update({
        "L": lang,
        "init_popn": [train_data],
        "archive_early": True,
        "gaussian_blur": True,
    })
    pt = util.ParamTester(config)
    for i, params in enumerate(pt):
        dist = params["d"].__name__
        select = params["select"]
        save_to = f"../out/simple_ns/{id}-{dist}-{select}.out"
        params.update({
            "save_to": save_to,
        })
        print(f"Searching with id={id}, dist={dist}, select={select}")
        wandb.init(project="novelty", config=params)
        evo_search(**params)

if __name__ == "__main__":
    id = str(int(time()))
    # run_on_real_points(id)
    # run_on_nat_points(id)
    run_on_lsystems(id)

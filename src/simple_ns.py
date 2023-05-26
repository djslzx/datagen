"""
ns without the evo
"""
import copy
from math import ceil
from pprint import pp
from typing import List, Union, Callable, Collection, Dict, Any
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import directed_hausdorff
from scipy.special import softmax
from scipy.ndimage import gaussian_filter
from einops import rearrange, reduce
import wandb
import yaml
from tqdm import tqdm

from lang import Language, Tree, ParseError
import point
import lindenmayer
import regexpr
import examples
import util

Distance = Callable[[np.ndarray, np.ndarray], float]

def features(L: Language, S: Collection[Tree], n_samples: int, batch_size=4, gaussian_blur_sigma=0) -> np.ndarray:
    # take samples from programs in S, then batch them and feed them through
    # the feature extractor for L
    def samples():
        for x in S:
            for _ in range(n_samples):
                bmp = L.eval(x, env={})
                if gaussian_blur_sigma > 0:
                    bmp = gaussian_filter(bmp, sigma=gaussian_blur_sigma)
                yield bmp
    ys = []
    n_batches = ceil(len(S) * n_samples / batch_size)
    for batch in tqdm(util.batched(samples(), batch_size=batch_size), total=n_batches):
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

def take_samples(L: Language, n_samples: int, length_cap: int, simplify=False) -> np.ndarray:
    out = []
    while len(out) < n_samples:
        x = L.sample()
        if simplify:
            try: x = L.simplify(x)
            except ParseError: continue  # retry
        if len(x) <= length_cap:
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

def log_mds(L: Language, t: int, samples_per_program: int,
            A: List, e_A: List[np.ndarray],
            A_: List, e_A_: List[np.ndarray],
            S: np.ndarray, e_S: np.ndarray,
            P: np.ndarray, e_P: np.ndarray,
            P_: np.ndarray, e_P_: np.ndarray) -> wandb.Table:
    """Log the positions of the archive, samples, and population as a table for viz"""
    assert samples_per_program == 1, f"MDS not implemented for samples_per_program > 1, got {samples_per_program}"
    mds = MDS(n_components=2, metric=True, random_state=0)  # use fixed random state for reproducibility
    # each embedding matrix has shape [k_i, embedding_size], so concat along axis 0
    if not A:
        embeddings = np.concatenate((e_A_, e_S, e_P, e_P_), axis=0)
    else:
        embeddings = np.concatenate((e_A, e_A_, e_S, e_P, e_P_), axis=0)
    mds_embeddings = mds.fit_transform(embeddings)

    # split mds_embeddings into pieces matching original inputs
    table = wandb.Table(columns=["t", "x", "y", "kind", "program"])
    kinds = {"A": A, "A'": A_, "S": S, "P": P, "P'": P_}
    endpoints = util.split_endpoints([len(v) for v in kinds.values()])
    for (kind, xs), (start, end) in zip(kinds.items(), endpoints):
        for i, pt in enumerate(mds_embeddings[start:end]):
            table.add_data(t, *pt, kind, L.to_str(xs[i]))
    return table

def log_best_and_worst(k: int, L: Language, samples: np.ndarray, scores: np.ndarray) -> Dict:
    def summarize(indices):
        img = rearrange([L.eval(x) for x in samples[indices]], "b color row col -> row (b col) color")
        caption = "Left to right: " + ", ".join(f"{L.to_str(x)} ({score:.4e})"
                                                for x, score in zip(samples[indices], scores[indices]))
        return wandb.Image(img, caption=caption)

    i_best = np.argsort(-scores)[:k]
    i_worst = np.argsort(scores)[:k]
    return {"best": summarize(i_best),
            "worst": summarize(i_worst)}


def evo_search(L: Language,
               init_popn: List[Tree],
               holdout: List[Tree],
               d: Distance,
               select: str,
               max_popn_size: int,
               samples_per_program: int,
               samples_ratio: int,
               keep_per_iter: int,
               alpha: float,
               iters: int,
               save_to: str,
               archive_early=False,
               gaussian_blur_sigma=0,
               length_cap=1000,
               length_penalty_type="additive",
               length_penalty_additive_coeff=0.1,
               length_penalty_inverse_coeff=10,
               ablate_mutator=False,
               simplify=False,
               **kvs):
    assert samples_ratio >= 2, \
        "Number of samples taken should be significantly larger than number of samples kept"
    assert len(init_popn) >= 5, \
        f"Initial population ({len(init_popn)}) must be geq number of nearest neighbors (5)"
    assert length_penalty_type in {"additive", "inverse"}

    def embed(S):
        return features(L, S, n_samples=samples_per_program, batch_size=8, gaussian_blur_sigma=gaussian_blur_sigma)

    def update_archive(A, E_A, S, E_S):
        # just take the first `keep_per_iter` instead of random sampling?
        # samples are generated randomly and are effectively unordered, so it should be fine?
        I_A = np.random.choice(samples_per_iter, size=keep_per_iter, replace=False)
        A.extend(S[I_A])
        E_A.extend(E_S[I_A])

    def log_data(t: int, dA, S, I, dists, lengths, scores):
        """
        Save one generation of evolved programs:
        - t: time step
        - dA: new programs in archive
        - S: new program samples
        - I: indices of programs selected for next generation from S
        - dists: dists_i = novelty distance for S_i
        - lengths: lengths_i = length(S_i)
        """
        assert len(S) == len(dists) == len(lengths)
        with open(f"{save_to}.csv", "a") as f:
            writer = csv.writer(f)
            for x in dA:
                writer.writerow((t, L.to_str(x), "A", None, len(x), None, None))
            for i, (x, dist, length, score) in enumerate(zip(S, dists, lengths, scores)):
                writer.writerow((t, L.to_str(x), "S", dist, length, score, i in I))

    def concat(a: Union[np.ndarray, List], b: np.ndarray) -> np.ndarray:
        if not a: return b
        elif not b: return a
        else: return np.concatenate((a, b), axis=0)

    ## prep data files
    # write metadata
    with open(f"{save_to}.metadata", "w") as f:
        pp(wandb.run.config.as_dict(), stream=f)

    # write column headers
    with open(f"{save_to}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "program", "kind", "dist", "length", "score", "chosen"])

    samples_per_iter = samples_ratio * max_popn_size
    archive = []
    popn = init_popn
    e_archive = []
    e_popn = embed(popn)
    e_hold_out = embed(holdout)
    metric = make_dist(d=d, k=samples_per_program) if samples_per_program > 1 else "minkowski"
    holdout_knn = NearestNeighbors(metric=metric)
    holdout_knn.fit(e_hold_out)
    knn = NearestNeighbors(metric=metric)
    for t in range(iters):
        with util.Timing(f"Iteration {t}") as timer:
            if not ablate_mutator:
                L.fit(popn, alpha=alpha)
            samples = take_samples(L, samples_per_iter, length_cap=length_cap, simplify=simplify)  # todo: weight by recency/novelty
            e_samples = embed(samples)
            if archive_early: update_archive(archive, e_archive, samples, e_samples)

            # score samples wrt archive + popn
            knn.fit(concat(e_archive, e_popn))
            dists, _ = knn.kneighbors(e_samples)
            dists = np.sum(dists, axis=1)
            len_samples = np.array([len(x) for x in samples])
            if length_penalty_type == "additive":
                scores = dists - length_penalty_additive_coeff * len_samples  # add penalty term for length
            else:
                scores = dists / (len_samples + length_penalty_inverse_coeff)

            # select samples to carry over to next generation
            i_popn = select_indices(select, scores, max_popn_size)
            if not archive_early: update_archive(archive, e_archive, samples, e_samples)
            log_data(t=t, dA=archive[-keep_per_iter:], S=samples, I=i_popn, dists=dists, lengths=len_samples, scores=scores)
            popn = samples[i_popn]
            e_popn = e_samples[i_popn]

        # distance from hold_out set
        sample_holdout_dists, _ = holdout_knn.kneighbors(e_samples)
        sample_holdout_dists = np.sum(sample_holdout_dists, axis=1)
        select_holdout_dists = sample_holdout_dists[i_popn]

        log = {"scores": wandb.Histogram(scores[i_popn]),
               "sample lengths": wandb.Histogram(len_samples),
               "chosen lengths": wandb.Histogram(len_samples[i_popn]),
               "sample holdout dists": wandb.Histogram(sample_holdout_dists),
               "selected holdout dists": wandb.Histogram(select_holdout_dists),
               "dists": wandb.Histogram(dists),
               "avg score": np.mean(scores),
               "avg length": np.mean(len_samples),
               "avg dist": np.mean(dists),
               "avg sample holdout dist": np.mean(sample_holdout_dists),
               "avg selected holdout dist": np.mean(select_holdout_dists),
               "runtime": timer.time()}

        if t > 0 and isinstance(L, lindenmayer.LSys):
            log.update(log_best_and_worst(5, L, samples, scores))

            # exit early if 'best' samples are actually terrible
            MIN_LENGTH = 10  # length of F;F~F
            i_best = np.argsort(-scores)[:5]
            if np.all(len_samples[i_best] == MIN_LENGTH):
                break

        wandb.log(log)


def run_on_real_points() -> str:
    lang = point.RealPoint()
    train_data = [
        lang.parse("(0, 0)"),
        lang.parse("(1, 0)"),
        lang.parse("(0, 1)"),
        lang.parse("(-1, 0)"),
        lang.parse("(0, -1)"),
    ]
    config = {
        "L": lang,
        "init_popn": train_data,
        "d": hausdorff,
        "select": "strict",
        "samples_per_program": 1,
        "samples_ratio": 10,
        "max_popn_size": 10,
        "keep_per_iter": 1,
        "iters": 10,
        "alpha": 1,
        "gaussian_blur_sigma": 0,
    }
    wandb.init(project="novelty", config=config)
    evo_search(**config, save_to=f"../out/simple_ns/{wandb.run.id}")
    return wandb.run.id

def run_on_nat_points(id: str):
    lang = point.NatPoint()
    train_data = [
        lang.parse("(one, one)"),
        lang.parse("(inc one, one)"),
        lang.parse("(one, inc one)"),
        lang.parse("(inc one, inc one)"),
        lang.parse("(inc inc one, one)"),
    ]
    config = {
        "L": lang,
        "init_popn": train_data,
        "d": hausdorff,
        "select": "strict",
        "samples_per_program": 1,
        "samples_per_iter": 100,
        "max_popn_size": 10,
        "keep_per_iter": 1,
        "iters": 10,
        "alpha": 1,
        "gaussian_blur_sigma": 0,
    }
    wandb.init(project="novelty", config=config)
    evo_search(**config, save_to=f"../out/simple_ns/{id}-z2-strict.out",)

def lsystem_sweep():
    run_on_lsystems('configs/simple-config.yaml')

def run_on_lsystems(filename: str):
    with open(filename) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.init(project="novelty", config=config)
    config = wandb.config
    lang = lindenmayer.LSys(
        **config.render,
        kind="deterministic",
        disable_last_layer=config.disable_last_layer,
        softmax_outputs=config.softmax_outputs,
    )
    train_data = [lang.parse(x) for x in config.train_data]
    holdout_data = [lang.parse(x) for x in config.holdout_data]
    evo_search(**config.search,
               L=lang,
               init_popn=train_data,
               holdout=holdout_data,
               d=hausdorff,
               save_to=f"../out/ns/{wandb.run.id}")


def viz_real_points_results(path: str):
    data = pd.read_csv(path)
    lang = point.RealPoint()
    print(data)
    outputs = np.array([lang.eval(lang.parse(x)) for x in data["program"]])
    data["x"] = outputs[:, 0]
    data["y"] = outputs[:, 1]
    sns.scatterplot(data, x="x", y="y", hue="step", markers="kind")
    plt.show()


if __name__ == '__main__':
    # run_id = run_on_real_points()
    # viz_real_points_results(f"../out/simple_ns/{run_id}.csv")
    # viz_real_points_results(f"../out/simple_ns/7hea21on.csv")
    # run_on_nat_points()
    # run_on_lsystems(filename="configs/static-config.yaml")
    pass

lsystem_sweep()
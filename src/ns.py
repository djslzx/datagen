"""
ns without the evo
"""
import json
from math import ceil
from pprint import pp
from typing import List, Callable, Collection, Dict, Iterator, TextIO, Tuple, Union
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import directed_hausdorff
from scipy.special import softmax
from einops import rearrange
import wandb
import yaml
from tqdm import tqdm

from featurizers import ResnetFeaturizer
from lang.tree import Language, Tree, ParseError
from lang import lindenmayer, point, arc
import util

Distance = Callable[[np.ndarray, np.ndarray], float]


def extract_features(L: Language, S: Collection[Tree], n_samples=1, batch_size=4, load_bar=False) -> np.ndarray:
    # take samples from programs in S, then batch them and feed them through
    # the feature extractor for L
    def samples():
        for x in S:
            for _ in range(n_samples):
                print(x)
                yield L.eval(x, env={'z': list(range(100))})

    ys = []
    n_batches = ceil(len(S) * n_samples / batch_size)
    batches = util.batched(samples(), batch_size=batch_size)
    if load_bar: batches = tqdm(batches, total=n_batches)
    for batch in batches:
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
        try:
            x = L.sample()
        except RecursionError:
            continue  # retry
        if simplify:
            try:
                x = L.simplify(x)
            except ParseError:
                continue  # retry
        if len(x) <= length_cap:
            out.append(x)
        # log failures?
    return np.array(out, dtype=object)


def select_indices(kind: str, dists: np.ndarray, n: int):
    if kind == "strict":
        return np.argsort(-dists)[:n]  # sort descending
    elif kind == "weighted":
        return np.random.choice(len(dists), n, replace=False, p=softmax(dists))
    else:
        raise ValueError(f"Unknown selection kind: {kind}")


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


def log_best_and_worst(k: int, L: Language, samples: np.ndarray, scores: np.ndarray) -> Dict:
    def summarize(indices):
        img = rearrange([L.eval(x) for x in samples[indices]],
                        "b h w c -> h (b w) c")
        caption = "Left to right: " + ", ".join(f"{L.to_str(x)} ({score:.4e})"
                                                for x, score in zip(samples[indices], scores[indices]))
        return wandb.Image(img, caption=caption)

    i_best = np.argsort(-scores)[:k]
    i_worst = np.argsort(scores)[:k]
    return {"best": summarize(i_best),
            "worst": summarize(i_worst)}


def evo_search(L: Language,
               init_popn: List[Tree],
               d: Union[Distance, str],
               select: str,
               max_popn_size: int,
               samples_per_program: int,
               samples_ratio: int,
               keep_per_iter: int,
               alpha: float,
               iters: int,
               archive_early=False,
               length_cap=1000,
               length_penalty_type="additive",
               length_penalty_additive_coeff=0.1,
               length_penalty_inverse_coeff=10,
               ablate_mutator=False,
               simplify=False) -> Iterator[dict]:
    assert samples_ratio >= 2, \
        "Number of samples taken should be significantly larger than number of samples kept"
    assert len(init_popn) >= 5, \
        f"Initial population ({len(init_popn)}) must be geq number of nearest neighbors (5)"
    assert length_penalty_type in {"additive", "inverse", None}

    def embed(S):
        return extract_features(L, S, n_samples=samples_per_program, batch_size=8)

    def update_archive(A, E_A, S, E_S):
        # just take the first `keep_per_iter` instead of random sampling?
        # samples are generated randomly and are effectively unordered, so it should be fine?
        I = np.random.choice(samples_per_iter, size=keep_per_iter, replace=False)
        A.extend(S[I])
        E_A.extend(E_S[I])

    samples_per_iter = samples_ratio * max_popn_size
    archive = []
    popn = init_popn
    e_archive = []
    e_popn = embed(popn)

    # choose metric
    if isinstance(d, str):
        metric = d
    elif samples_per_program > 1:
        metric = make_dist(d=d, k=samples_per_program)
    else:
        metric = "minkowski"

    knn = NearestNeighbors(metric=metric)
    for t in range(iters):
        if not ablate_mutator:
            L.fit(popn, alpha=alpha)
        # todo: weight by recency/novelty
        samples = take_samples(L, samples_per_iter, length_cap=length_cap, simplify=simplify)
        e_samples = embed(samples)
        if archive_early:
            update_archive(archive, e_archive, samples, e_samples)

        # score samples wrt archive + popn
        knn.fit(np.concatenate((e_archive, e_popn), axis=0) if archive else e_popn)
        dists, _ = knn.kneighbors(e_samples)
        dists = np.sum(dists, axis=1)
        len_samples = np.array([len(x) for x in samples])
        if length_penalty_type == "additive":
            scores = dists - length_penalty_additive_coeff * len_samples
        elif length_penalty_type == "inverse":
            scores = dists / (len_samples + length_penalty_inverse_coeff)
        else:
            scores = dists

        # select samples to carry over to next generation
        i_popn = select_indices(select, scores, max_popn_size)
        if not archive_early:
            update_archive(archive, e_archive, samples, e_samples)

        # log data
        for x in archive[-keep_per_iter:]:
            yield {
                "kind": "data",
                "payload": {"t": t, "program": L.to_str(x), "kind": "archive", "length": len(x)},
            }
        for i, (x, dist, length, score) in enumerate(zip(samples, dists, len_samples, scores)):
            yield {
                "kind": "data",
                "payload": {"t": t, "program": L.to_str(x), "kind": "samples", "length": int(length),
                            "dist": float(dist), "score": float(score), "chosen?": i in i_popn},
            }

        yield {
            "kind": "log",
            "payload": {
                "t": t,
                "samples": samples,
                "scores": scores,
                "chosen scores": scores[i_popn],
                "sample lengths": len_samples,
                "chosen lengths": len_samples[i_popn],
                "dists": dists,
                "avg score": np.mean(scores),
                "avg length": np.mean(len_samples),
                "avg dist": np.mean(dists),
            },
        }

        popn = samples[i_popn]
        e_popn = e_samples[i_popn]


def run_on_real_points():
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
    }
    wandb.init(project="novelty", config=config)
    util.incrementally_save_jsonl(
        (d["payload"] for d in evo_search(**config) if d["kind"] == "data"),
        filename=f"../out/ns/z2-{wandb.run.id}",
    )


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
    }
    wandb.init(project="novelty", config=config)
    util.incrementally_save_jsonl(
        (d["payload"] for d in evo_search(**config) if d["kind"] == "data"),
        filename=f"../out/simple_ns/nat2-{id}-{wandb.run.id}",
    )


def lsystem_sweep():
    run_on_lsystems('configs/simple-config.yaml')


def run_on_lsystems(filename: str):
    with open(filename) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.init(project="novelty", config=config)
    config = wandb.config
    featurizer = ResnetFeaturizer(**config.featurizer)
    lang = lindenmayer.LSys(
        kind="deterministic",
        featurizer=featurizer,
        **config.render,
    )
    train_data = [lang.parse(x) for x in config.train_data]
    # holdout_data = [lang.parse(x) for x in config.holdout_data]
    with open(f"../out/ns/lsys-{wandb.run.id}", "w") as f, util.Timing(f"LSys-NS") as timer:
        for d in evo_search(
                **config.search,
                L=lang,
                init_popn=train_data,
                d=hausdorff,
                simplify=config.simplify
        ):
            kind = d["kind"]
            payload = d["payload"]

            if kind == "data":
                s = json.dumps(payload, indent=None)
                print(s)
                f.write(s + "\n")
            elif kind == "log":
                log = {
                    "scores": wandb.Histogram(payload["scores"]),
                    "chosen scores": wandb.Histogram(payload["chosen scores"]),
                    "sample lengths": wandb.Histogram(payload["sample lengths"]),
                    "chosen lengths": wandb.Histogram(payload["chosen lengths"]),
                    "dists": wandb.Histogram(payload["dists"]),
                    "avg score": payload["avg score"],
                    "avg length": payload["avg length"],
                    "avg dist": payload["avg dist"],
                }
                if payload["t"] > 0:
                    log.update(log_best_and_worst(5, lang, payload["samples"], payload["scores"]))
                wandb.log(log)

                # exit early if best samples are actually terrible
                MIN_LENGTH = 10  # length of F;F~F
                i_best = np.argsort(-payload["scores"])[:5]
                if np.all(payload["sample lengths"][i_best] == MIN_LENGTH):
                    break
            else:
                raise ValueError(f"Unknown kind: {kind} with payload: {payload}")


def viz_real_points_results(path: str):
    data = pd.read_csv(path)
    lang = point.RealPoint()
    print(data)
    outputs = np.array([lang.eval(lang.parse(x)) for x in data["program"]])
    data["x"] = outputs[:, 0]
    data["y"] = outputs[:, 1]
    sns.scatterplot(data, x="x", y="y", hue="step", markers="kind")
    plt.show()


def run_on_arc():
    wandb.init(project="arc-novelty")
    feat = ResnetFeaturizer(sigma=1)
    lang = arc.Blocks(featurizer=feat, gram=2)
    seed = [
        "(rect (point 1 2) (point 1 2) 1)",
        "(rect (point 1 1) (point xmax ymax) 1)",
        "(line (point 1 2) (point 3 4) 1)",
        "(seq (line (point 1 2) (point 3 4) 1) "
        "     (rect (point 1 2) (point 1 2) 1))",
        "(apply hflip (line (point 1 2) (point 1 4) 1))",
    ]
    with open(f"../out/ns/arc", "w") as f, util.Timing(f"ARC-NS") as timer:
        for d in evo_search(
                L=lang,
                init_popn=[lang.parse(s) for s in seed],
                d="minkowski",
                samples_per_program=1,
                iters=10,
                select="strict",
                alpha=0.01,
                max_popn_size=100,
                samples_ratio=3,
                keep_per_iter=10,
        ):
            kind = d["kind"]
            payload = d["payload"]

            print(payload)


if __name__ == '__main__':
    # run_id = run_on_real_points()
    # viz_real_points_results(f"../out/simple_ns/{run_id}.csv")
    # viz_real_points_results(f"../out/simple_ns/7hea21on.csv")
    # run_on_nat_points()
    # run_on_lsystems(filename="configs/tiny-lsys-config.yaml")
    run_on_arc()

# lsystem_sweep()

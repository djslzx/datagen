"""
Test out evolutionary search algorithms for data augmentation.
"""
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import *
from os import mkdir
from pprint import pp
import time
import Levenshtein

from grammar import Grammar
from zoo import zoo
from lindenmayer import S0LSystem
from featurizers import ResnetFeaturizer, Featurizer, RawFeaturizer
import parse
from util import Timing
import random_baseline
from param_tester import ParamTester

# Hyper-parameters
DRAW_ARGS = {
    'd': 3,
    'theta': 45,
    'n_rows': 128,
    'n_cols': 128,
}
ROLLOUT_DEPTH = 3
START = "LSystem"
OUTDIR = "../out/evo"


def next_gen(metagrammar: Grammar, n_next_gen: int, p_arkv: float, simplify: bool) -> Tuple[np.ndarray, Set]:
    popn = np.empty(n_next_gen, dtype=object)
    arkv = set()
    n_simplified = 0
    for i in range(n_next_gen):  # parallel
        while True:  # retry until we get non-nil
            ttree = metagrammar.sample(START)
            stree = parse.eval_ttree_as_str(ttree)
            if simplify:
                try:
                    n_old = len(stree)
                    stree = parse.simplify(stree)
                    n_simplified += n_old - len(stree)
                    break
                except parse.NilError:  # retry on nil
                    pass
        popn[i] = stree
        if np.random.random() < p_arkv:
            arkv.add(stree)
    print(f"Simplified {n_simplified/n_next_gen:.3e} tokens on avg")
    return popn, arkv


def semantic_score(prev_gen: np.ndarray, new_gen: np.ndarray,
                   featurizer: Featurizer, n_samples: int, n_neighbors: int) -> np.ndarray:
    n_features = featurizer.n_features

    # build knn data structure
    with Timing("Computing features"):
        popn_features = np.empty((len(prev_gen), n_features * n_samples))
        for i, s in enumerate(prev_gen):  # parallel
            sys = S0LSystem.from_str(s)
            for j in range(n_samples):
                bmp = sys.draw(sys.nth_expansion(ROLLOUT_DEPTH), **DRAW_ARGS)
                popn_features[i, j * n_features: (j + 1) * n_features] = featurizer.apply(bmp)

    with Timing("Building knn data structure"):
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(prev_gen))).fit(popn_features)

    # compute scores of next generation
    with Timing("Scoring instances"):
        scores = np.empty(len(new_gen))
        for i, s in enumerate(new_gen):  # parallel
            features = np.empty((1, n_features * n_samples))
            sys = S0LSystem.from_str(s)
            for j in range(n_samples):
                bmp = sys.draw(sys.nth_expansion(ROLLOUT_DEPTH), **DRAW_ARGS)
                features[0, j * n_features: (j + 1) * n_features] = featurizer.apply(bmp)
            distances, indices = knn.kneighbors(features)
            scores[i] = distances.sum(axis=1).item()
            # scores[i] = distances.mean(axis=1).item() ** 2 / len(s)  # prioritize shorter agents

    return scores


def syntactic_semantic_score(popn: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
    n_popn = len(popn)
    scores = np.empty(n_popn)
    for i, s in enumerate(popn):
        syntactic_score = sum(Levenshtein.distance(s, x) for x in popn) / n_popn
        scores[i] = semantic_scores[i] * syntactic_score
    return scores


def novelty_search(init_popn: List[str],
                   max_popn_size: int,
                   iters: int,
                   featurizer: Featurizer,
                   n_samples: int,
                   arkv_growth_rate: float,
                   n_neighbors: int,
                   next_gen_ratio: int,
                   simplify: bool,       # use egg to simplify expressions between generations
                   ingen_novelty: bool,  # measure novelty wrt current gen, not archive/past gens
                   out_dir: str) -> Set[Iterable[str]]:  # store outputs here
    log_params = locals()  # Pull local variables so that we can log the args that were passed to this function
    p_arkv = arkv_growth_rate / max_popn_size

    arkv = set()
    popn = np.array(init_popn, dtype=object)
    n_next_gen = max_popn_size * next_gen_ratio
    metagrammar = Grammar.from_components(parse.rule_types, gram=2)

    for iter in range(iters):
        print(f"[Novelty search: iter {iter}]")
        t_start = time.time()

        # generate next gen
        with Timing("Fitting metagrammar"):
            corpus = [parse.parse_lsys(x) for x in popn]
            counts = parse.bigram_scans(corpus)
            metagrammar.from_bigram_counts_(counts, alpha=1)

        with Timing("Generating next gen"):
            new_gen, new_arkv = next_gen(metagrammar=metagrammar,
                                         n_next_gen=n_next_gen,
                                         p_arkv=p_arkv,
                                         simplify=simplify)
            arkv |= new_arkv

        # compute popn features, build knn data structure, and score next_gen
        with Timing("Scoring population"):
            prev_gen = new_gen if ingen_novelty else np.concatenate((np.array(popn), np.array(list(arkv), dtype=object)), axis=0)
            scores = semantic_score(prev_gen=prev_gen,
                                    new_gen=new_gen,
                                    featurizer=featurizer,
                                    n_neighbors=n_neighbors,
                                    n_samples=n_samples)

        # cull popn
        with Timing("Culling popn"):
            indices = np.argsort(-scores)  # sort descending: higher mean distances first
            new_gen = new_gen[indices]
            popn = new_gen[:max_popn_size]  # take indices of top `max_popn_size` agents

        # make labels
        with Timing("Logging"):
            scores = scores[indices]
            min_score = scores[max_popn_size - 1]
            labels = [f"{score:.2e}" + ("*" if score >= min_score else "")
                      for score in scores]

        # printing/saving
        print(f"[Completed iteration {iter} in {time.time() - t_start:.2f}s.]")

        # save gen
        with open(f"{out_dir}/gen-{iter}.txt", "w") as f:
            f.write(f"# {log_params}\n")
            for agent, label in zip(new_gen, labels):
                f.write("".join(agent) + f" : {label}\n")

        # save arkv
        with open(f"{out_dir}/arkv.txt", "a") as f:
            for agent in new_arkv:
                f.write(''.join(agent) + "\n")

    return arkv


def try_mkdir(dir: str):
    try:
        f = open(dir, "r")
        f.close()
    except FileNotFoundError:
        print(f"{dir} directory not found, making dir...", file=sys.stderr)
        mkdir(dir)
    except IsADirectoryError:
        pass


def main(name: str):
    t = int(time.time())
    random_seed = [random_baseline.sample_mg() for _ in range(len(zoo))]
    zoo_seed = [x.to_str() for x in zoo]
    simple_seed = [
        "F;F~F+F,F~F-F",
        "F;F~FF,F~F-F",
        "F;F~F",
        "F;F~FF",
        "F+F;F~FF",
        "F-F;F~FF",
    ]
    p = ParamTester({
        'init_popn': [zoo_seed, random_seed, simple_seed],
        'simplify': [True, False],
        'max_popn_size': [100, 1000],
        'n_neighbors': [10, 100],
        'arkv_growth_rate': [2, 4],
        'iters': 1000,
        'next_gen_ratio': 10,
        'ingen_novelty': False,
        'featurizer': ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True),
        'n_samples': 3,
    })
    for i, params in enumerate(p):
        out_dir = f"{OUTDIR}/{t}-{name}-{i}"
        try_mkdir(out_dir)
        params |= {
            "out_dir": out_dir,
        }
        print("****************")
        print(f"Running {i}-th novelty search with parameters:\n{pp(params)}")
        novelty_search(**params)


if __name__ == '__main__':
    main('test')


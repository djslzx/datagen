"""
Test out evolutionary search algorithms for data augmentation.
"""
import pdb
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import *
from os import mkdir
from pprint import pp
import time
import Levenshtein
import itertools as it

from grammar import Grammar
from zoo import zoo
from lindenmayer import S0LSystem
from featurizers import ResnetFeaturizer, Featurizer, RawFeaturizer
import parse
from util import Timing

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


def next_gen(metagrammar: Grammar, n_next_gen: int, p_arkv: float) -> Tuple[np.ndarray, Set]:
    popn = np.empty(n_next_gen, dtype=object)
    arkv = set()
    for i in range(n_next_gen):  # parallel
        ttree = metagrammar.sample(START)
        sentence = parse.eval_ttree_as_str(ttree)
        popn[i] = sentence
        if np.random.random() < p_arkv:
            arkv.add(sentence)
    return popn, arkv


def semantic_score(prev_gen: np.ndarray, new_gen: np.ndarray,
                   featurizer: Featurizer, n_samples: int, n_neighbors: int) -> np.ndarray:
    n_features = featurizer.n_features

    # build knn data structure
    with Timing("Computing features"):
        popn_features = np.empty((len(prev_gen), n_features * n_samples))
        for i, s in enumerate(prev_gen):  # parallel
            sys = S0LSystem.from_sentence(list(s))
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
            sys = S0LSystem.from_sentence(list(s))
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
                   p_arkv: float,
                   n_neighbors: int,
                   next_gen_ratio: int,
                   sentence_limit: int,
                   measure_novelty_within_generation: bool,
                   out_dir: str) -> Set[Iterable[str]]:
    # todo: use tensorboard via lightning?
    log_params = locals()  # Pull local variables so we can log the args that were passed to this function

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
            counts = parse.multi_count_bigram(corpus)
            metagrammar.from_bigram_counts_(counts)
        with Timing("Generating next gen"):
            new_gen, new_arkv = next_gen(metagrammar=metagrammar, n_next_gen=n_next_gen, p_arkv=p_arkv)
            arkv |= new_arkv

        # compute popn features, build knn data structure, and score next_gen
        with Timing("Scoring population"):
            if measure_novelty_within_generation:
                prev_gen = new_gen
            else:
                prev_gen = np.concatenate((np.array(popn), np.array(list(arkv), dtype=object)), axis=0)
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

        print("====================")
        print(f"Completed iteration {iter} in {time.time() - t_start:.2f}s.")
        print(f"New generation ({n_next_gen}):")
        for agent, label in zip(new_gen, labels):
            print(f"  {''.join(agent)} - {label}")
        print(f"Population ({len(popn)}):")
        pp([''.join(x) for x in popn])
        print(f"Archive: ({len(arkv)})")
        pp([''.join(x) for x in arkv])
        print("====================")

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


def main(name: str):
    seed_popn = [x.to_str() for x in zoo]
    t = int(time.time())

    # choices for each param
    popn_sizes = [10, 100, 1000, 10000]
    arkv_growth_rates = [1, 2, 4, 8]
    iterations = [10, 100, 1000]
    neighborhood_sizes = [1, 10, 100]
    novelty_within_gen = [False, True]

    for i, args in enumerate(it.product(popn_sizes, arkv_growth_rates, iterations,
                                        neighborhood_sizes, novelty_within_gen)):
        popn_size, arkv_growth_rate, iters, neighborhood_size, novelty_within = args
        out_dir = f"{OUTDIR}/{t}-{name}-{i}"

        try:
            f = open(out_dir, "r")
            f.close()
        except FileNotFoundError:
            print(f"{out_dir} directory not found, making dir...", file=sys.stderr)
            mkdir(out_dir)
        except IsADirectoryError:
            pass

        params = {
            'out_dir': out_dir,
            'init_popn': seed_popn,
            'iters': iters,
            'featurizer': ResnetFeaturizer(disable_last_layer=False,
                                           softmax_outputs=True),
            'max_popn_size': popn_size,
            'n_neighbors': neighborhood_size,
            'n_samples': 3,
            'next_gen_ratio': 10,
            'sentence_limit': 30,
            'p_arkv': arkv_growth_rate / popn_size,
            'measure_novelty_within_generation': novelty_within,
        }
        print("****************")
        print(f"Running {i}-th novelty search with parameters: {params}")
        novelty_search(**params)


if __name__ == '__main__':
    # simple_seed_systems = [
    #     S0LSystem("F", {"F": ["F+F", "F-F"]}),
    #     S0LSystem("F", {"F": ["FF", "F-F"]}),
    #     S0LSystem("F", {"F": ["F"]}),
    #     S0LSystem("F", {"F": ["FF"]}),
    #     S0LSystem("F", {"F": ["FFF"]}),
    #     S0LSystem("F+F", {"F": ["FF"]}),
    #     S0LSystem("F-F", {"F": ["FF"]}),
    # ]
    main('test')


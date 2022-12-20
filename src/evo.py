"""
Test out evolutionary search algorithms for data augmentation.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import *
from os import mkdir
from pprint import pp
import time
import Levenshtein

from cfg import CFG, PCFG
from book_zoo import simple_zoo_systems
from lindenmayer import S0LSystem, trained_bigram_metagrammar
from featurizers import ResnetFeaturizer, Featurizer, RawFeaturizer

# Set up file paths
PCFG_CACHE_PREFIX = ".cache/pcfg-"
IMG_CACHE_PREFIX = ".cache/imgs/"

for directory in [".cache/", ".cache/imgs/"]:
    try:
        open(directory, "r")
    except FileNotFoundError:
        print(f"{directory} directory not found, making dir...")
        mkdir(directory)
    except IsADirectoryError:
        pass

# Hyper-parameters
DRAW_ARGS = {
    'd': 3,
    'theta': 45,
    'n_rows': 128,
    'n_cols': 128,
}
ROLLOUT_DEPTH = 3


def gen_next_gen(metagrammar: PCFG, n_next_gen: int, p_arkv: float, sentence_limit: int) -> Tuple[np.ndarray, Set]:
    next_gen = np.empty(n_next_gen, dtype=object)
    arkv = set()
    for i in range(n_next_gen):  # parallel
        retried = False
        while True:
            sentence = tuple(metagrammar.iterate_fully())
            if len(sentence) <= sentence_limit and \
               all(sentence != stored for stored in next_gen[:i]):
                break
            print(i if not retried else '.', end='')
            retried = True
        if retried: print(', ', end='')

        next_gen[i] = sentence
        if np.random.random() < p_arkv:
            arkv.add(sentence)
    print()
    return next_gen, arkv


def semantic_score(popn: np.ndarray, featurizer: Featurizer, n_samples: int, n_neighbors: int) -> np.ndarray:
    n_popn = len(popn)
    n_features = featurizer.n_features

    # build knn data structure
    popn_features = np.empty((len(popn), n_features * n_samples))
    for i, s in enumerate(popn):  # parallel
        sys = S0LSystem.from_sentence(s)
        for j in range(n_samples):
            bmp = sys.draw(sys.nth_expansion(ROLLOUT_DEPTH), **DRAW_ARGS)
            popn_features[i, j * n_features: (j + 1) * n_features] = featurizer.apply(bmp)
    knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(popn))).fit(popn_features)

    # compute scores using knn data structure
    scores = np.empty(n_popn)
    for i, s in enumerate(popn):  # parallel
        features = np.empty((1, n_features * n_samples))
        sys = S0LSystem.from_sentence(s)
        for j in range(n_samples):
            bmp = sys.draw(sys.nth_expansion(ROLLOUT_DEPTH), **DRAW_ARGS)
            features[0, j * n_features: (j + 1) * n_features] = featurizer.apply(bmp)
        distances, indices = knn.kneighbors(features)
        scores[i] = distances.mean(axis=1).item()
        # scores[i] /= LENGTH_PADDING + len(s)  # prioritize shorter agents
    return scores


def syntactic_semantic_score(popn: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
    n_popn = len(popn)
    scores = np.empty(n_popn)
    for i, s in enumerate(popn):
        syntactic_score = sum(Levenshtein.distance(s, x) for x in popn) / n_popn
        scores[i] = semantic_scores[i] * syntactic_score
    return scores


def novelty_search(init_popn: List[CFG.Sentence],
                   max_popn_size: int,
                   iters: int,
                   featurizer: Featurizer,
                   n_samples: int,
                   p_arkv: float,
                   n_neighbors: int,
                   next_gen_ratio: int,
                   sentence_limit: int,
                   id: str = "") -> Set[Iterable[str]]:
    arkv = set()
    popn = np.array(init_popn, dtype=object)
    n_next_gen = max_popn_size * next_gen_ratio
    t = f"{int(time.time())}-{id}"

    for iter in range(iters):
        print(f"[Novelty search: iter {iter}]")
        t_start = time.time()

        # generate next gen
        print("Fitting metagrammar...")
        t_io = time.time()
        metagrammar = trained_bigram_metagrammar(["".join(x) for x in popn], alpha=0.1)
        print(f"Fitting took {time.time() - t_io}s.")

        print("Generating next gen...")
        t_gen = time.time()
        next_gen, next_arkv = gen_next_gen(metagrammar=metagrammar, n_next_gen=n_next_gen,
                                           p_arkv=p_arkv, sentence_limit=sentence_limit)
        arkv |= next_arkv
        print(f"Generating took {time.time() - t_gen}s.")

        # compute popn features, build knn data structure, and score next_gen
        print("Scoring population...")
        t_score = time.time()
        sem_scores = semantic_score(popn=next_gen, featurizer=featurizer, n_neighbors=n_neighbors, n_samples=n_samples)
        scores = syntactic_semantic_score(popn=next_gen, semantic_scores=sem_scores)
        print(f"Scoring took {time.time() - t_score}s.")

        # cull popn
        print("Culling popn...")
        t_cull = time.time()
        indices = np.argsort(-scores)  # sort descending: higher mean distances first
        next_gen = next_gen[indices]
        popn = next_gen[:max_popn_size]  # take indices of top `max_popn_size` agents
        print(f"Culling took {time.time() - t_cull}s.")

        # make labels
        print("Logging...")
        t_log = time.time()
        scores = scores[indices]
        min_score = scores[max_popn_size - 1]
        labels = [f"{score:.2e}" + ("*" if score >= min_score else "")
                  for score in scores]
        t_taken = time.time() - t_start
        print(f"Logging took {time.time() - t_log}s.")

        print("====================")
        print(f"Completed iteration {iter} in {t_taken:.2f}s.")
        print(f"New generation ({n_next_gen}):")
        for agent, label in zip(next_gen, labels):
            print(f"  {''.join(agent)} - {label}")
        print(f"Population ({len(popn)}):")
        pp([''.join(x) for x in popn])
        print(f"Archive: ({len(arkv)})")
        pp([''.join(x) for x in arkv])
        print("====================")

        # save gen
        with open(f"{PCFG_CACHE_PREFIX}{t}-gen-{iter}.txt", "w") as f:
            for agent, label in zip(next_gen, labels):
                f.write("".join(agent) + f" : {label}\n")

        # save arkv
        with open(f"{PCFG_CACHE_PREFIX}{t}-arkv.txt", "a") as f:
            for agent in next_arkv:
                f.write(''.join(agent) + "\n")

    return arkv


if __name__ == '__main__':
    seed = [
        x.to_sentence()
        for x in
        simple_zoo_systems
        # [
        #     S0LSystem("F", {"F": ["F+F", "F-F"]}),
        #     S0LSystem("F", {"F": ["FF", "F-F"]}),
        #     S0LSystem("F", {"F": ["F"]}),
        #     S0LSystem("F", {"F": ["FF"]}),
        #     S0LSystem("F", {"F": ["FFF"]}),
        #     S0LSystem("F+F", {"F": ["FF"]}),
        #     S0LSystem("F-F", {"F": ["FF"]}),
        # ]
    ]
    popn_size = 100
    arkv_growth_rate = 2
    params = {
        'id': 'gen-training-exs',
        'init_popn': seed,
        'iters': 100,
        'featurizer':
            # RawFeaturizer(DRAW_ARGS['n_rows'], DRAW_ARGS['n_cols']),
            ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True),
        'max_popn_size': popn_size,
        'n_neighbors': 20,
        'n_samples': 3,
        'next_gen_ratio': 10,
        'sentence_limit': 25,
        'p_arkv': arkv_growth_rate / popn_size,
    }
    print(f"Running novelty search with parameters: {params}")
    novelty_search(**params)

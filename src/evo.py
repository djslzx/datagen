"""
Test out evolutionary search algorithms for data augmentation.
"""
from typing import List, Dict, Set, Tuple, Iterator, Iterable
import pickle
import time
import pdb
from sklearn.neighbors import NearestNeighbors
import torch as T
import numpy as np
import random
import os

from cfg import PCFG
from lindenmayer import S0LSystem
from inout import log_io
from resnet import featurize_images
from datagen import GENERAL_MG
from book_zoo import zoo_systems
import util

# Set up file paths
PCFG_CACHE_PREFIX = ".cache/pcfg-"
IMG_CACHE_PREFIX = ".cache/imgs/"

for dir in [".cache/", ".cache/imgs/"]:
    try:
        open(dir, "r")
    except FileNotFoundError:
        print(f"{dir} directory not found, making dir...")
        os.mkdir(dir)
    except IsADirectoryError:
        pass


def mutate_agents(specimens: Iterable[S0LSystem], metagrammar, n_samples: int, smoothing=0.5) -> Iterator[S0LSystem]:
    """
    Produce the next generation of L-systems from a set of L-system specimens.

    Fit a PCFG to the specimens using inside-outside with smoothing, then sample
    from the PCFG to get 'mutated' L-systems.
    """
    # check cached PCFG
    genomes = [specimen.to_sentence() for specimen in specimens]
    hash_str = "\n".join(" ".join(genome) for genome in genomes) + str(smoothing)
    hash_val = util.md5_hash(hash_str)
    cache_file = f"{PCFG_CACHE_PREFIX}{hash_val}"

    try:
        with open(cache_file, "rb") as f:
            g_fit = pickle.load(f)
        print(f"Found cached file, loaded fitted PCFG from {cache_file}: {g_fit}")

    except FileNotFoundError:
        print("No cached file found, running inside-outside...")
        g = metagrammar.to_CNF().normalized().log()
        g_fit = log_io(g, genomes, smoothing, verbose=True)
        print(f"Fitted PCFG: {g_fit}")

        # cache pcfg
        print(f"Fitted PCFG, saving to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(g_fit, f)

    # sample from fitted PCFG
    # TODO: try getting n_samples with n_samples * k tries, where k >= 1
    # TODO: add a time limit param to iteration of fitted grammar
    for i in range(n_samples):
        sentence = g_fit.exp().iterate_fully()
        sys = S0LSystem.from_sentence(sentence)
        yield sys


def novelty(indiv: S0LSystem, popn: Set[S0LSystem], k: int, n_samples: int, rollout_length: int = 100) -> float:
    """Measures the novelty of a stochastic L-system relative to a population of L-systems."""
    # Sample `n_samples` images from S0LSystems in popn
    popn_paths = []
    for i, lsystem in enumerate(popn):
        for j in range(n_samples):
            _, rollout = lsystem.expand_until(rollout_length)
            path = f"{IMG_CACHE_PREFIX}popn-{i}-{j}"
            popn_paths.append(f"{path}.png")
            S0LSystem.to_png(s=rollout, d=1, theta=43, filename=path)

    # Sample images from individual
    indiv_paths = []
    for j in range(n_samples):
        _, rollout = indiv.expand_until(rollout_length)
        path = f"{IMG_CACHE_PREFIX}indiv-{j}"
        indiv_paths.append(f"{path}.png")
        S0LSystem.to_png(s=rollout, d=1, theta=43, filename=path)

    # Get feature vectors from images
    popn_features_dict = featurize_images(popn_paths)
    popn_feature_vecs = T.stack([v for v in popn_features_dict.values()]).detach().numpy()
    indiv_features_dict = featurize_images(indiv_paths)
    indiv_feature_vecs = T.stack([v for v in indiv_features_dict.values()]).detach().numpy()

    # Take kNN distance
    knn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(popn_feature_vecs)
    distances, _ = knn.kneighbors(indiv_feature_vecs)
    scores = distances.sum(axis=1)
    return scores.max(axis=0)  # max or min?


def novelty_search(init_popn: Set[S0LSystem], grammar: PCFG, iters: int,
                   smoothing: float, p_arkv: float) -> Set[S0LSystem]:
    """Runs novelty search."""
    popn: Set[S0LSystem] = init_popn
    arkv: Set[S0LSystem] = set()
    popn_size = len(popn)
    for i in range(iters):
        # generate next gen
        next_gen = mutate_agents(popn, grammar, n_samples=popn_size, smoothing=smoothing)

        # evaluate next gen
        agents_with_scores = []
        for agent in next_gen:
            # store a subset of the popn in the arkv
            if random.random() < p_arkv:
                arkv.add(agent)
            score = novelty(agent, popn | arkv, k=5, n_samples=5)
            agents_with_scores.append((score, agent))

        # cull popn
        popn = {agent for score, agent in sorted(agents_with_scores, key=lambda x: x[0])[:popn_size]}
    return popn | arkv


def demo_mutate_agents():
    agents = mutate_agents(
        specimens=[S0LSystem("F", {"F": ["F+F", "F-F"]})],
        metagrammar=GENERAL_MG,
        n_samples=3,
        smoothing=0.01
    )
    for agent in agents:
        print(agent)


if __name__ == '__main__':
    indiv = S0LSystem("F", {"F": ["F+F", "F-F"]})
    popn = {
        S0LSystem("F", {"F": ["F+F", "F-F"]}),
        S0LSystem("F", {"F": ["FF"]}),
        S0LSystem("F", {"F": ["F++F", "FF"]}),
    }
    n = novelty(indiv, popn, k=2, n_samples=3, rollout_length=30)
    print(n)

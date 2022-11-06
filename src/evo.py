"""
Test out evolutionary search algorithms for data augmentation.
"""
import pickle
import numpy as np
import torch as T
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple, Iterator, Iterable, Callable
import math
import time
import random
import os
from pprint import pp
import pdb

from cfg import PCFG
from lindenmayer import S0LSystem
from inout import log_dirio_step, log_io
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


def img_featurizer() -> Callable[[np.ndarray], np.ndarray]:
    """
    Sets up ResNet50 and returns it as a function that maps an image to a 2048-feature vector.
    """
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    # model = T.nn.Sequential(*list(resnet.children())[:-1])  # disable last layer in resnet
    model = resnet
    model.eval()
    preprocess = weights.transforms()

    def featurizer(img: np.ndarray) -> np.ndarray:
        tensor = T.from_numpy(np.repeat(img[None, ...], 3, axis=0))  # stack array over RGB channels
        batch = preprocess(tensor).unsqueeze(0)
        features = model(batch).squeeze().softmax(0)
        return features.detach().numpy()

    return featurizer


def sample_lsystem(system: S0LSystem, n_samples: int, d: int, theta: float,
                   rollout_limit: int, n_rows: int, n_cols: int) -> List[np.ndarray]:
    bmps = []
    for _ in range(n_samples):
        _, rollout = system.expand_until(rollout_limit)
        bmp = S0LSystem.draw(rollout, d, theta, n_rows, n_cols)
        bmps.append(bmp)
    return bmps


def mutate_agents(specimens: Iterable[S0LSystem], metagrammar, n_samples: int, smoothing=0.5) -> Iterator[S0LSystem]:
    """
    Produce the next generation of L-systems from a set of L-system specimens.

    Fit a PCFG to the specimens using inside-outside with smoothing, then sample
    from the PCFG to get 'mutated' L-systems.
    """
    # check cached PCFG
    print(f"Making genomes...")
    genomes = [specimen.to_sentence() for specimen in specimens]
    g = metagrammar.to_CNF().normalized().log()
    # print(f"Fitting PCFG via Dirichlet IO...")
    # g_fit = log_dirio_step(g, genomes, smoothing)
    print(f"Fitting PCFG via IO...")
    g_fit = log_io(g, genomes, smoothing)
    print(f"Fitted PCFG: {g_fit}")

    # sample from fitted PCFG
    # TODO: try getting n_samples with n_samples * k tries, where k >= 1
    # TODO: add a time limit param to iteration of fitted grammar
    for i in range(n_samples):
        sentence = g_fit.exp().iterate_until(1000)
        sys = S0LSystem.from_sentence(sentence)
        yield sys


def novelty(indiv: S0LSystem, popn: Set[S0LSystem], featurizer: Callable[[np.ndarray], np.ndarray],
            k: int, n_samples: int, rollout_limit: int) -> float:
    """Measures the novelty of a stochastic L-system relative to a population of L-systems."""
    def sample(lsystem: S0LSystem):
        return sample_lsystem(lsystem, n_samples, d=3, theta=90, rollout_limit=rollout_limit, n_rows=300, n_cols=300)

    # Sample images from S0LSystems in popn, individual
    popn_bmps = [bmp for sys in popn for bmp in sample(sys)]
    indiv_bmps = sample(indiv)

    # Get feature vectors from images
    popn_features = np.stack([featurizer(bmp) for bmp in popn_bmps])
    indiv_features = np.stack([featurizer(bmp) for bmp in indiv_bmps])

    # Take kNN distance
    knn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(popn_features)
    distances, _ = knn.kneighbors(indiv_features)
    scores = distances.sum(axis=1)
    return scores.max(axis=0)  # max or min?


def novelty_search(init_popn: Set[S0LSystem], max_popn_size: int, grammar: PCFG, iters: int,
                   smoothing: float, p_arkv: float, rollout_limit: int, verbose=False) -> Set[S0LSystem]:
    """Runs novelty search."""
    popn: Set[S0LSystem] = init_popn
    arkv: Set[S0LSystem] = set()
    featurizer = img_featurizer()
    id = int(time.time())

    for i in range(iters):
        if verbose: print(f"[NS iter {i}]")

        # generate next gen
        if verbose: print("Generating next gen...")
        next_gen = list(mutate_agents(popn, grammar, n_samples=max_popn_size, smoothing=smoothing))  # fixme

        # evaluate next gen
        if verbose: print("Scoring agents...")
        agents_with_scores = []
        for agent in next_gen:
            # store a subset of the popn in the arkv
            if random.random() < p_arkv:
                arkv.add(agent)
            score = novelty(agent, popn | arkv, featurizer, k=5, n_samples=5, rollout_limit=rollout_limit)
            agents_with_scores.append((score, agent))

        # plot next gen
        plot_agents(next_gen, len(next_gen), 2, f"{IMG_CACHE_PREFIX}/{id}-gen-{i}.png")

        # cull popn
        if verbose: print("Culling popn...")
        popn = {agent for score, agent in sorted(agents_with_scores, reverse=True, key=lambda x: x[0])[:max_popn_size]}
        pp([(score, "".join(agent.to_sentence())) for score, agent in agents_with_scores])

        # plot popn
        plot_agents(popn, len(popn), 2, f"{IMG_CACHE_PREFIX}/{id}-popn-{i}.png")

        if verbose:
            print("Completed iteration.")
            pp(popn)
            pp(arkv)
            print("====================")

    plot_agents(arkv, len(arkv), 2, f"{IMG_CACHE_PREFIX}/{id}-arkv.png")
    return arkv


def plot_agents(agents: Iterable[S0LSystem], n_agents: int, n_samples_per_agent: int, saveto: str):
    n_bmps = n_samples_per_agent * n_agents
    n_rows = int(math.sqrt(n_bmps))
    n_cols = n_bmps // n_rows
    if n_rows * n_cols < n_bmps:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, n_cols)
    faxes = axes.flat
    i = 0
    for agent in agents:
        # title = "".join(agent.to_sentence())
        for bmp in sample_lsystem(agent, n_samples_per_agent, 3, 90, 100, 64, 64):
            faxes[i].imshow(bmp)
            # faxes[i].set_title(title)
            i += 1
    plt.savefig(saveto)
    plt.close()


def demo_mutate_agents():
    agents = mutate_agents(
        specimens=[S0LSystem("F", {"F": ["F+F", "F-F"]})],
        metagrammar=GENERAL_MG,
        n_samples=3,
        smoothing=0.01
    )
    for agent in agents:
        print(agent)


def demo_measure_novelty():
    indiv = S0LSystem("+", {"F": ["F"]})
    popn = {
        # S0LSystem("F", {"F": ["F+F", "F-F"]}),
        # S0LSystem("F", {"F": ["F+F", "F-F"]}),
        S0LSystem(
            "F-F-F-F",
            {"F": ["F+FF-FF-F-F+F+FF-F-F+F+FF+FF-F"]}
        ),
        # S0LSystem("F", {"F": ["FF"]}),
        # S0LSystem("F", {"F": ["F++F", "FF"]}),
    }
    featurizer = img_featurizer()
    print(novelty(indiv, popn, featurizer, k=1, n_samples=1, rollout_limit=1000))


def demo_ns():
    popn = {
        S0LSystem("F", {"F": ["F+F", "F-F"]}),
        S0LSystem("F", {"F": ["FF"]}),
        S0LSystem("F", {"F": ["F++F", "FF"]}),
    }
    systems = novelty_search(init_popn=popn, grammar=GENERAL_MG, iters=10,
                             max_popn_size=9, smoothing=1, p_arkv=1, rollout_limit=100, verbose=True)

    for system in systems:
        print(system)


if __name__ == '__main__':
    demo_ns()
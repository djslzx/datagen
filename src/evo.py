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
from lindenmayer import S0LSystem, LSYSTEM_MG
from inout import log_dirio_step, log_io
from book_zoo import zoo_systems, simple_zoo_systems
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

# Hyperparameters
D=3
THETA=43
NROWS=64
NCOLS=64
ROLLOUT_LIMIT=100


def img_featurizer() -> Callable[[np.ndarray], np.ndarray]:
    """
    Sets up ResNet50 and returns it as a function that maps an image to a 2048-feature vector.
    """
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    model = T.nn.Sequential(*list(resnet.children())[:-1])  # disable last layer in resnet
    model.eval()
    preprocess = weights.transforms()

    def featurizer(img: np.ndarray) -> np.ndarray:
        tensor = T.from_numpy(np.repeat(img[None, ...], 3, axis=0))  # stack array over RGB channels
        batch = preprocess(tensor).unsqueeze(0)
        features = model(batch).squeeze().softmax(0)
        return features.detach().numpy()

    return featurizer


def sample_images(system: S0LSystem, n_samples: int, d: int, theta: float,
                  rollout_limit: int, n_rows: int, n_cols: int) -> List[np.ndarray]:
    bmps = []
    for _ in range(n_samples):
        _, rollout = system.expand_until(rollout_limit)
        bmp = S0LSystem.draw(rollout, d, theta, n_rows, n_cols)
        bmps.append(bmp)
    return bmps


def random_lsystems(n_systems: int) -> List[S0LSystem]:
    try:
        with open(f"{PCFG_CACHE_PREFIX}-zoo.pcfg", "rb") as f:
            mg = pickle.load(f)
        print(f"Loaded pickled zoo meta-grammar from {PCFG_CACHE_PREFIX}-zoo.pcfg")
    except FileNotFoundError:
        print(f"Could not find pickled zoo meta-grammar, fitting...")
        mg = LSYSTEM_MG.apply_to_weights(lambda x: x)
        corpus = [sys.to_sentence() for sys in zoo_systems]
        mg = log_io(mg, corpus, smoothing=1)
        with open(f"{PCFG_CACHE_PREFIX}-zoo.pcfg", "wb") as f:
            pickle.dump(mg, f)

    return [S0LSystem.from_sentence(mg.iterate_fully())
            for _ in range(n_systems)]


def mutate_agents(specimens: Iterable[S0LSystem], n_samples: int, smoothing=0.5) -> Iterator[S0LSystem]:
    """
    Produce the next generation of L-systems from a set of L-system specimens.

    Fit a PCFG to the specimens using inside-outside with smoothing, then sample
    from the PCFG to get 'mutated' L-systems.
    """
    # check cached PCFG
    print(f"Making genomes...")
    genomes = [specimen.to_sentence() for specimen in specimens]
    g = LSYSTEM_MG.to_CNF().normalized().log()
    # print(f"Fitting PCFG via Dirichlet IO...")
    # g_fit = log_dirio_step(g, genomes, smoothing)
    print(f"Fitting PCFG via IO...")
    g_fit = log_io(g, genomes, smoothing).exp()
    print(f"Fitted PCFG: {g_fit}")

    # sample from fitted PCFG
    for i in range(n_samples):
        sentence = g_fit.iterate_until(1000)
        sys = S0LSystem.from_sentence(sentence)
        yield sys


def novelty(indiv: S0LSystem, popn: Set[S0LSystem], featurizer: Callable[[np.ndarray], np.ndarray],
            k: int, n_samples: int) -> float:
    """Measures the novelty of a stochastic L-system relative to a population of L-systems."""
    def sample(lsystem: S0LSystem):
        return sample_images(lsystem, n_samples, d=D, theta=THETA,
                             rollout_limit=ROLLOUT_LIMIT, n_rows=NROWS, n_cols=NCOLS)

    # Sample images from S0LSystems in popn, individual; map to feature vectors
    popn_features = np.stack([featurizer(bmp) for sys in popn for bmp in sample(sys)])
    indiv_features = np.stack([featurizer(bmp) for bmp in sample(indiv)])

    # Take kNN distance
    knn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(popn_features)
    distances, _ = knn.kneighbors(indiv_features)
    scores = distances.sum(axis=1)
    return scores.max(axis=0)  # max or min or mean?


def novelty_search(init_popn: Set[S0LSystem], max_popn_size: int, iters: int,
                   smoothing: float, p_arkv: float, verbose=False) -> Set[S0LSystem]:
    """Runs novelty search."""
    popn: Set[S0LSystem] = init_popn
    arkv: Set[S0LSystem] = set()
    featurizer = img_featurizer()
    id = int(time.time())

    for i in range(iters):
        if verbose:
            print(f"[NS iter {i}]")
            t_start = time.time()

        # generate next gen
        if verbose: print("Generating next gen...")
        next_gen = list(mutate_agents(popn, n_samples=max_popn_size * 2, smoothing=smoothing))  # fixme

        # evaluate next gen
        if verbose: print("Scoring agents...")
        scored_agents = []
        for agent in next_gen:
            # store a subset of the popn in the arkv
            if random.random() < p_arkv:
                arkv.add(agent)
            score = novelty(agent, popn | arkv, featurizer, k=5, n_samples=5)
            scored_agents.append((score, agent))

        # plot next gen
        plot_agents(next_gen, len(next_gen), 2, f"{IMG_CACHE_PREFIX}/{id}-gen-{i}.png")

        # cull popn
        if verbose: print("Culling popn...")
        scored_agents.sort(reverse=True, key=lambda x: x[0])
        popn = {agent for score, agent in scored_agents[:max_popn_size]}

        # plot popn
        plot_agents(popn, len(popn), 2, f"{IMG_CACHE_PREFIX}/{id}-popn-{i}.png")

        if verbose:
            t_taken = time.time() - t_start
            print("====================")
            print(f"Completed iteration {i} in {t_taken:.2f}s.")
            print("New generation:")
            pp([(score, agent.to_code()) for score, agent in scored_agents])
            print("Population:")
            pp([x.to_code() for x in popn])
            print("Archive:")
            pp([x.to_code() for x in arkv])
            print("====================")

    plot_agents(arkv, len(arkv), 2, f"{IMG_CACHE_PREFIX}/{id}-arkv.png")
    save_agents(arkv, f"{PCFG_CACHE_PREFIX}{id}.txt")
    return arkv


def plot_agents(agents: Iterable[S0LSystem], n_agents: int, n_samples_per_agent: int, saveto: str):
    n_bmps = n_samples_per_agent * n_agents
    n_rows = int(math.sqrt(n_bmps))
    n_cols = n_bmps // n_rows
    if n_rows * n_cols < n_bmps:
        n_rows += 1

    fig, ax = plt.subplots(n_rows, n_cols)
    # clear axis ticks
    for axis in ax.flat:
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    # plot bitmaps
    i = 0
    axes: List[plt.Axes] = ax.flat
    for agent in agents:
        for bmp in sample_images(agent, n_samples_per_agent, d=D, theta=THETA,
                                 rollout_limit=ROLLOUT_LIMIT, n_rows=NROWS, n_cols=NCOLS):
            axis = axes[i]
            axis.imshow(bmp)
            i += 1
    plt.savefig(saveto)
    plt.close()


def save_agents(agents: Iterable[S0LSystem], saveto: str):
    with open(saveto, "w") as f:
        for agent in agents:
            f.write(agent.to_code() + "\n")


def demo_mutate_agents():
    agents = mutate_agents(
        specimens=[S0LSystem("F", {"F": ["F+F", "F-F"]})],
        n_samples=3,
        smoothing=0.01
    )
    for agent in agents:
        print(agent)


def demo_measure_novelty():
    indiv = S0LSystem("+", {"F": ["F"]})
    popn = {
        S0LSystem(
            "F-F-F-F",
            {"F": ["F+FF-FF-F-F+F+FF-F-F+F+FF+FF-F"]}
        ),
        # S0LSystem("F", {"F": ["F+F", "F-F"]}),
        # S0LSystem("F", {"F": ["FF"]}),
        # S0LSystem("F", {"F": ["F++F", "FF"]}),
    }
    featurizer = img_featurizer()
    print(novelty(indiv, popn, featurizer, k=1, n_samples=1))


def demo_ns():
    popn = {
        S0LSystem("F", {"F": ["F+F", "F-F"]}),
        S0LSystem("F", {"F": ["FF"]}),
        S0LSystem("F", {"F": ["F++F", "FF"]}),
    }
    novelty_search(
        init_popn=popn,
        iters=1,
        max_popn_size=4,
        smoothing=1,
        p_arkv=1,
        verbose=True
    )


if __name__ == '__main__':
    # demo_mutate_agents()
    # demo_measure_novelty()
    demo_ns()

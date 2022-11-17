"""
Test out evolutionary search algorithms for data augmentation.
"""
import pickle
import itertools as it
import numpy as np
import torch as T
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from typing import *
import joblib
import time
import os
from pprint import pp
import pdb

from cfg import CFG, PCFG
from lindenmayer import S0LSystem, LSYSTEM_MG
from inout import log_io, autograd_outside, inside_outside_step, log_io_step
from book_zoo import zoo_systems

# Set up file paths
PCFG_CACHE_PREFIX = ".cache/pcfg-"
IMG_CACHE_PREFIX = ".cache/imgs/"

for directory in [".cache/", ".cache/imgs/"]:
    try:
        open(directory, "r")
    except FileNotFoundError:
        print(f"{directory} directory not found, making dir...")
        os.mkdir(directory)
    except IsADirectoryError:
        pass

# Hyper-parameters
D = 3
THETA = 43
N_ROWS = 64
N_COLS = 64
ROLLOUT_DEPTH = 3


class Featurizer:
    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError


class ResnetFeaturizer(Featurizer):

    def __init__(self, disable_last_layer=False):
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.disable_last_layer = disable_last_layer
        if disable_last_layer:
            # disable last layer in resnet
            self.model = T.nn.Sequential(*list(resnet.children())[:-1])
        else:
            self.model = resnet
        self.model.eval()
        self.preprocess = weights.transforms()

    def apply(self, img: np.ndarray) -> np.ndarray:
        tensor = T.from_numpy(np.repeat(img[None, ...], 3, axis=0))  # stack array over RGB channels
        batch = self.preprocess(tensor).unsqueeze(0)
        features = self.model(batch).squeeze().softmax(0)
        return features.detach().numpy()

    @property
    def n_features(self) -> int:
        return 2048 if self.disable_last_layer else 1000


class DummyFeaturizer(Featurizer):

    def __init__(self):
        pass

    def apply(self, img: np.ndarray) -> np.ndarray:
        return np.array([np.mean(img), np.var(img)])

    @property
    def n_features(self) -> int:
        return 2


class SyntacticSemanticFeaturizer(Featurizer):

    def __init__(self):
        raise NotImplementedError

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError


def sample_images(system: S0LSystem, n_samples: int, d: int, theta: float,
                  rollout_depth: int, n_rows: int, n_cols: int) -> Iterator[np.ndarray]:
    for _ in range(n_samples):
        rollout = system.nth_expansion(rollout_depth)
        yield S0LSystem.draw(rollout, d, theta, n_rows, n_cols)


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


def novelty_search(init_popn: Collection[S0LSystem], max_popn_size: int, iters: int,
                   featurizer: Featurizer, smoothing: float, p_arkv: float, n_neighbors: int,
                   verbose=False) -> Set[Iterable[str]]:
    """
    Runs novelty search.

    Uses multiple renders of each stochastic L-system and concatenates them together.
    """
    popn = [x.to_sentence() for x in init_popn]
    arkv = set()
    t = int(time.time())
    n_next_gen = max_popn_size * 4
    n_samples = 2
    n_features = featurizer.n_features
    meta_PCFG = PCFG.from_CFG(LSYSTEM_MG.to_CNF())
    draw_kvs = {'d': 5, 'theta': 43, 'n_rows': 256, 'n_cols': 256}

    for iter in range(iters):
        if verbose:
            print(f"[NS iter {iter}]")
            t_start = time.time()

        # generate next gen
        if verbose: print("Fitting metagrammar...")
        # metagrammar = log_io_step(meta_PCFG.copy().log(), popn).exp()
        # metagrammar = inside_outside_step(meta_PCFG.copy(), popn, smoothing)
        metagrammar = autograd_outside(meta_PCFG, popn, iters=10)

        if verbose: print("Generating next gen...")
        next_gen = np.empty(n_next_gen, dtype=object)
        for i in range(n_next_gen):  # parallel
            sentence = tuple(metagrammar.iterate_fully())
            next_gen[i] = sentence
            if np.random.random() < p_arkv:
                arkv.add(sentence)

        # compute popn features and build knn data structure
        if verbose: print("Featurizing population...")
        popn_features = np.empty((len(popn), n_features * n_samples))
        for i, s in enumerate(popn):  # parallel
            sys = S0LSystem.from_sentence(s)
            for j in range(n_samples):
                bmp = sys.draw(sys.nth_expansion(ROLLOUT_DEPTH), **draw_kvs)
                popn_features[i, j * n_features: (j+1) * n_features] = featurizer.apply(bmp)
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(init_popn))).fit(popn_features)

        # score next gen via novelty
        if verbose: print("Scoring next generation...")
        scores = np.empty(n_next_gen)
        for i, s in enumerate(next_gen):  # parallel
            sys = S0LSystem.from_sentence(s)
            features = np.empty((1, n_features * n_samples))
            for j in range(n_samples):
                bmp = sys.draw(sys.nth_expansion(ROLLOUT_DEPTH), **draw_kvs)
                features[0, j * n_features: (j+1) * n_features] = featurizer.apply(bmp)
            distances, _ = knn.kneighbors(features)
            scores[i] = distances.sum(axis=1).item()

        # cull popn
        if verbose: print("Culling popn...")
        indices = np.argsort(-scores)  # sort descending
        next_gen = next_gen[indices]
        popn = next_gen[:max_popn_size]  # take indices of top `max_popn_size` agents

        # plot generation with selection markings
        if verbose: print("Plotting...")
        scores = scores[indices]
        min_score = scores[max_popn_size - 1]
        labels = [f"{score:.2e}" + ("*" if score >= min_score else "")
                  for score in scores]
        plot_agents(next_gen, labels, 2, f"{IMG_CACHE_PREFIX}/{t}-popn-{iter}.png")

        if verbose:
            t_taken = time.time() - t_start
            print("====================")
            print(f"Completed iteration {iter} in {t_taken:.2f}s.")
            print("New generation:")
            for agent, label in zip(next_gen, labels):
                print(f"  {''.join(agent)} - {label}")
            print(f"Population ({len(popn)}):")
            pp([''.join(x) for x in popn])
            print(f"Archive: ({len(arkv)})")
            pp([''.join(x) for x in arkv])
            print("====================")

    plot_agents(arkv, ["" for _ in range(len(arkv))], 2, f"{IMG_CACHE_PREFIX}/{t}-arkv.png")
    save_agents(arkv, f"{PCFG_CACHE_PREFIX}{t}.txt")
    return arkv


def plot_agents(agents: Collection[CFG.Sentence], labels: Collection[str], n_samples_per_agent: int, saveto: str):
    assert len(agents) == len(labels), \
        f"Found mismatched lengths of agents ({len(agents)}) and labels ({len(labels)})"

    n_bmps = n_samples_per_agent * len(agents)
    n_rows = int(np.sqrt(n_bmps))
    n_cols = n_bmps // n_rows
    if n_rows * n_cols < n_bmps:
        n_rows += 1
    dpi = 96 * (n_cols // 3 + 1)

    fig, ax = plt.subplots(n_rows, n_cols)
    # clear axis ticks
    for axis in ax.flat:
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    # plot bitmaps
    i = 0
    axes: List[plt.Axes] = ax.flat
    for agent, label in zip(agents, labels):
        sys = S0LSystem.from_sentence(agent)
        for bmp in sample_images(sys, n_samples_per_agent, d=D, theta=THETA,
                                 rollout_depth=ROLLOUT_DEPTH, n_rows=N_ROWS, n_cols=N_COLS):
            axis = axes[i]
            axis.imshow(bmp)
            axis.set_title(label, fontsize=4, pad=4)
            i += 1
    plt.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.1)
    plt.savefig(saveto, dpi=dpi)
    plt.close()


def save_agents(agents: Iterable[CFG.Sentence], saveto: str):
    with open(saveto, "w") as f:
        for agent in agents:
            f.write(''.join(agent) + "\n")


def demo_plot():
    agents = [S0LSystem("F", {"F": ["F+F", "F-F"]})] * 36
    plot_agents(agents, labels=[agent.to_code() for agent in agents], n_samples_per_agent=2,
                saveto=f"{IMG_CACHE_PREFIX}test-plot.png")


def demo_ns():
    popn = {
        S0LSystem("F", {"F": ["F+F", "F-F"]}),
        S0LSystem("F", {"F": ["FF"]}),
        S0LSystem("F", {"F": ["F++F", "FF"]}),
        # S0LSystem(
        #     "F-F-F-F",
        #     {"F": ["F+FF-FF-F-F+F+FF-F-F+F+FF+FF-F"]}
        # ),
    }
    params = {
        'init_popn': popn,
        'iters': 2,
        f'featurizer': DummyFeaturizer(),
        'max_popn_size': 16,
        'n_neighbors': 5,
        'smoothing': 1,
        'p_arkv': 1/4,
        'verbose': True,
    }
    print(f"Running novelty search with parameters: {params}")
    novelty_search(**params)


if __name__ == '__main__':
    # demo_mutate_agents()
    # demo_measure_novelty()
    # demo_plot()
    demo_ns()

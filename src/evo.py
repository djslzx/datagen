"""
Test out evolutionary search algorithms for data augmentation.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from math import ceil
from typing import *
from os import mkdir
from pprint import pp
import time
import pdb

from cfg import CFG, PCFG
from lindenmayer import S0LSystem, LSYSTEM_MG
from inout import autograd_outside
from featurizers import DummyFeaturizer, ResnetFeaturizer, Featurizer
from book_zoo import zoo_systems, simple_zoo_systems

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
D = 3
THETA = 45
N_ROWS = 128
N_COLS = 128
ROLLOUT_DEPTH = 3
SENTENCE_LEN_LIMIT = 50
N_AGENTS_PER_PLOT = 9


def gen_next_gen(metagrammar: PCFG, n_next_gen: int, p_arkv: float) -> Tuple[np.ndarray, Set]:
    next_gen = np.empty(n_next_gen, dtype=object)
    arkv = set()
    for i in range(n_next_gen):  # parallel
        retried = False
        j = 0
        while True:
            sentence = tuple(metagrammar.iterate_fully())
            if len(sentence) <= SENTENCE_LEN_LIMIT and \
                all(sentence != stored for stored in next_gen[:i]):
                break
            j += 1
            if retried: print(j, end=', ')
            else: print(f"[next gen {i}] {j}, ", end='')
            retried = True
        if retried: print()

        next_gen[i] = sentence
        if np.random.random() < p_arkv:
            arkv.add(sentence)
    return next_gen, arkv


def novelty_search(init_popn: List[S0LSystem], max_popn_size: int, iters: int, io_iters: int,
                   featurizer: Featurizer, smoothing: float, p_arkv: float, n_neighbors: int,
                   verbose=False) -> Set[Iterable[str]]:
    popn = np.array(init_popn, dtype=object)
    arkv = set()
    n_next_gen = max_popn_size * 4
    n_samples = 3
    n_features = featurizer.n_features
    meta_PCFG = PCFG.from_CFG(LSYSTEM_MG.to_CNF())
    t = int(time.time())

    # TODO: to avoid extraneous computation w/ archives,
    #  - store the feature vectors of elts in the archive
    #  - measure the effect of more sampling (should be cheap in theory)
    #  - reconsider archiving entirely
    #  - log all generated specimens on disk instead of keeping in RAM

    for iter in range(iters):
        if verbose: print(f"[NS iter {iter}]")
        t_start = time.time()

        # generate next gen
        if verbose: print("Fitting metagrammar...")
        metagrammar = autograd_outside(meta_PCFG, popn, iters=io_iters)

        if verbose: print("Generating next gen...")
        next_gen, next_arkv = gen_next_gen(metagrammar, n_next_gen, p_arkv)
        arkv |= next_arkv

        # compute popn features and build knn data structure
        if verbose: print("Featurizing population...")
        popn_features = np.empty((len(popn), n_features * n_samples))
        for i, s in enumerate(popn):  # parallel
            sys = S0LSystem.from_sentence(s)
            for j in range(n_samples):
                bmp = sys.draw(sys.nth_expansion(ROLLOUT_DEPTH),
                               d=D, theta=THETA, n_rows=N_ROWS, n_cols=N_COLS)
                popn_features[i, j * n_features: (j+1) * n_features] = featurizer.apply(bmp)
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(popn))).fit(popn_features)

        # score next gen via novelty
        if verbose: print("Scoring next generation...")
        scores = np.empty(n_next_gen)
        for i, s in enumerate(next_gen):  # parallel
            features = np.empty((1, n_features * n_samples))
            sys = S0LSystem.from_sentence(s)
            for j in range(n_samples):
                bmp = sys.draw(sys.nth_expansion(ROLLOUT_DEPTH),
                               d=D, theta=THETA, n_rows=N_ROWS, n_cols=N_COLS)
                features[0, j * n_features: (j+1) * n_features] = featurizer.apply(bmp)
            distances, indices = knn.kneighbors(features)
            # neighbors = popn[indices[0]]
            # labels = ["".join(x) for x in neighbors]
            # print(f"Closest neighbors to {''.join(s)} are {labels}")
            # util.plot([sys.draw(S0LSystem.from_sentence(x).nth_expansion(ROLLOUT_DEPTH), D, THETA, N_ROWS, N_COLS)
            #            for x in neighbors],
            #           shape=(1, len(neighbors)),
            #           labels=labels)
            scores[i] = distances.mean(axis=1).item()
            # scores[i] /= len(s)  # prioritize shorter agents

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
        plot_agents_batched(next_gen, labels,
                            n_samples_per_agent=2,
                            n_agents_per_plot=N_AGENTS_PER_PLOT,
                            save_prefix=f"{IMG_CACHE_PREFIX}/{t}-popn-{iter}")

        if verbose:
            t_taken = time.time() - t_start
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

    save_agents(arkv, f"{PCFG_CACHE_PREFIX}{t}.txt")
    plot_agents_batched(list(arkv), None,
                        n_samples_per_agent=2,
                        n_agents_per_plot=N_AGENTS_PER_PLOT,
                        save_prefix=f"{IMG_CACHE_PREFIX}/{t}-arkv")

    return arkv


def plot_agents_batched(agents: Collection[CFG.Sentence],
                        labels: Optional[Collection[str]],
                        n_samples_per_agent: int,
                        n_agents_per_plot: int,
                        save_prefix: str):
    if not labels:
        labels = [""] * len(agents)

    assert len(agents) == len(labels), \
        f"Found mismatched lengths of agents ({len(agents)}) and labels ({len(labels)})"

    n_agents = len(agents)
    n_iters = ceil(n_agents / n_agents_per_plot)
    for i in range(n_iters):
        agent_batch = agents[i * n_agents_per_plot: (i + 1) * n_agents_per_plot]
        label_batch = labels[i * n_agents_per_plot: (i + 1) * n_agents_per_plot]
        plot_agents(agents=agent_batch,
                    labels=label_batch,
                    n_samples_per_agent=n_samples_per_agent,
                    saveto=f"{save_prefix}-{i}.png")


def plot_agents(agents: Collection[CFG.Sentence],
                labels: Optional[Collection[str]],
                n_samples_per_agent: int,
                saveto: str):
    if not labels:
        labels = [""] * len(agents)

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
        for _ in range(n_samples_per_agent):
            expansion = sys.nth_expansion(ROLLOUT_DEPTH)
            bmp = S0LSystem.draw(expansion, d=D, theta=THETA, n_rows=N_ROWS, n_cols=N_COLS)
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


if __name__ == '__main__':
    # demo_plot()
    popn = [
        x.to_sentence()
        for x in  # simple_zoo_systems
        [
            S0LSystem("F", {"F": ["F+F", "F-F"]}),
            S0LSystem("F", {"F": ["FF", "F-F"]}),
            S0LSystem("F", {"F": ["F"]}),
            S0LSystem("F", {"F": ["FF"]}),
            S0LSystem("F", {"F": ["FFF"]}),
            S0LSystem("F+F", {"F": ["FF"]}),
            S0LSystem("F-F", {"F": ["FF"]}),
        ]
    ]
    popn_size = 25
    arkv_growth_rate = 5
    n_neighbors = 10
    params = {
        'init_popn': popn,
        'iters': 10,
        'io_iters': 10,
        'featurizer': ResnetFeaturizer(disable_last_layer=True, softmax=True),
        'max_popn_size': popn_size,
        'n_neighbors': n_neighbors,
        'smoothing': 1,
        'p_arkv': arkv_growth_rate/popn_size,
        'verbose': True,
    }
    print(f"Running novelty search with parameters: {params}")
    novelty_search(**params)


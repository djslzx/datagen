from typing import List, Iterator, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
# import matplotlib;matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from lang.tree import Language, Tree
from lang import lindenmayer, point, arc
import util


def dpp_points(
        lang: point.RealPoint,
        n: int,
        n_samples: int,
        n_steps: int,
):
    # assume uniform initial distribution
    coords = np.random.uniform(size=(n, 2)) * 10
    x: np.ndarray[Tree] = np.array([lang.make_point(a, b) for a, b in coords], dtype=object)
    yield coords

    for t in range(1, n_steps):
        # singleton sliding window
        i = t % n

        # sample from proposal distribution
        lang.fit(x[None, i])
        samples = [lang.sample() for _ in range(n_samples)]

        # choose sample with best novelty score
        # (1) compute novelty score
        x_features = lang.extract_features(x)
        sample_features = lang.extract_features(samples)
        dists = novelty_scores(queries=sample_features, data=x_features)

        # (2) choose best sample wrt novelty score
        best_i = np.argmax(dists)
        best_sample = samples[best_i]

        # compute accept probability
        # A(x *, xt) = min(1, f(x') / f(x) * q(x|x') / q(x'|x))
        # (1) log f(x') - log f(x)
        #   = log det((B^T B)_x') - log det((B^T B)_x)
        L_x = np.matmul(x_features, x_features.transpose())
        updated_features = x_features.copy()
        updated_features[best_i] = sample_features[best_i]
        L_updated = np.matmul(updated_features, updated_features.transpose())
        log_f = np.log(np.linalg.det(L_x)) - np.log(np.linalg.det(L_updated))

        # (2) log q(x|x') - log q(x'|x)
        #   = log nov(xi|x\i) - log nov(xi'|x\i) + log p(xi|G(xi')) - log p(xi'|G(xi))
        log_q = (
                np.log(novelty_scores(x_features[None, i], x_features)[0])  # log nov(xi|x)
                - np.log(dists[best_i])  # log nov(xi'|x)
                + lang_log_pr(lang, query=x[i], data=best_sample)  # log p(xi|G(xi'))
                - lang_log_pr(lang, query=best_sample, data=x[i])  # log p(xi'|G(xi))
        )

        # (3) A = min(1, .) => log A = min(0, .)
        log_accept = min(0, log_f + log_q)

        # stochastically accept/reject
        if np.random.uniform(low=0, high=1) < np.exp(log_accept):
            # accept
            x[best_i] = best_sample
        else:
            # reject
            pass

        # yield current step point coords for animation
        yield np.stack([lang.eval(point) for point in x])


def lang_log_pr(lang: Language, query: Tree, data: Tree) -> float:
    lang.fit([data], alpha=1.0)
    return lang.log_probability(query)


def novelty_scores(queries: np.ndarray, data: np.ndarray, n_neighbors=5) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data)
    dists, indices = knn.kneighbors(queries)
    dists = np.sum(dists, axis=1)
    return dists


def animate_points(data_gen: Iterator, title: str, xlim: Tuple[int, int], ylim: Tuple[int, int]):
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    def update(frame):
        i, data = frame
        ax.set_title(f"{title}, frame: {i}")
        scatter.set_offsets(data)
        return scatter,

    return FuncAnimation(fig, update, frames=enumerate(data_gen), blit=False)


if __name__ == "__main__":
    N_STEPS = 1000
    generator = dpp_points(
        lang=point.RealPoint(xlim=10, ylim=10),
        n=10,
        n_samples=3,
        n_steps=N_STEPS,
    )
    anim = animate_points(
        generator,
        title=f"n_samples=3",
        xlim=(-10, 10),
        ylim=(-10, 10)
    )
    ts = util.timestamp()
    anim.save(f"../out/anim/ddp-points-{ts}.mp4")

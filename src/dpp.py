from typing import List, Iterator, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from lang.tree import Language, Tree
from lang import lindenmayer, point, arc
import util


def dpp_points_roundrobin_multisample(
        lang: point.RealPoint,
        n: int,
        accept_policy: str,
        n_samples: int,
        n_steps: int,
):
    # assume uniform initial distribution
    coords = np.random.uniform(size=(n, 2))
    x: np.ndarray[Tree] = np.array([lang.make_point(a, b) for a, b in coords])

    for t in range(n_steps):
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

        record = {}
        if accept_policy == 'dpp':
            # compute accept probability
            # A(x *, xt) = min(1, f(x') / f(x) * q(x|x') / q(x'|x))
            # (1) log f(x') - log f(x)
            #   = log det((B^T B)_x') - log det((B^T B)_x)
            L_x = np.matmul(x_features, x_features.transpose())
            updated_features = x_features.copy()
            updated_features[i] = sample_features[best_i]
            L_updated = np.matmul(updated_features, updated_features.transpose())
            det_x = np.linalg.det(L_x)
            det_s = np.linalg.det(L_updated)

            # assert det_x > 0 and det_s > 0, f"det_x: {det_x}, det_s: {det_s}"
            log_f = np.log(det_x) - np.log(det_s)

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
            p_accept = np.exp(log_accept)

            # log stuff
            record.update({
                "det(L_x)": det_x,
                "det(L_x')": det_s,
                "det < 0": int(det_x <= 0 or det_s <= 0),
                "log f(x') - log f(x)": log_f,
                "log q(x|x') - log q(x'|x)": log_q,
                "log A(x'|x)": log_accept,
            })
        elif accept_policy == 'all':
            p_accept = 1
        else:
            raise ValueError(f"Unknown accept policy: {accept_policy}")

        # stochastically accept/reject
        if np.random.uniform(low=0, high=1) < p_accept:
            # accept: update x[i] with best sample
            x[i] = best_sample
        else:
            # reject: keep x[i] as is
            pass

        # yield current step point coords for animation
        yield {
            "points": [lang.eval(p) for p in x],
            "samples": [lang.eval(p) for p in samples],
            "best_i": best_i,
            "i": i,
            "t": t,
            **record,
        }


def dpp_points_roundrobin(
        lang: point.RealPoint,
        n: int,
        fit_policy: str,
        accept_policy: str,
        n_steps: int,
):
    # assume uniform initial distribution
    coords = np.random.uniform(size=(n, 2))
    x = [lang.make_point(a, b) for a, b in coords]

    for t in range(n_steps):
        # singleton sliding window
        i = t % n

        # sample from proposal distribution
        if fit_policy == 'all':
            lang.fit(x)
        elif fit_policy == 'single':
            lang.fit([x[i]])

        s = lang.sample()
        x_feat = lang.extract_features(x)
        s_feat = lang.extract_features([s])[0]

        record = {}
        if accept_policy == 'dpp':
            # compute accept probability
            # (1) log f(x') - log f(x) = log det((B^T B)_x') - log det((B^T B)_x)
            L_x = np.matmul(x_feat, x_feat.transpose())
            updated_features = x_feat.copy()
            updated_features[i] = s_feat
            L_up = np.matmul(updated_features, updated_features.transpose())
            s_x, log_det_x = np.linalg.slogdet(L_x)
            s_s, log_det_s = np.linalg.slogdet(L_up)
            log_f = s_x * log_det_x - s_s * log_det_s

            # (2) log q(x|x') - log q(x'|x)
            log_q = lang_log_pr(lang, query=x[i], data=s) - lang_log_pr(lang, query=s, data=x[i])

            # (3) A = min(1, .) => log A = min(0, .)
            log_accept = min(0, log_f + log_q)
            p_accept = np.exp(log_accept)

            record.update({
                "L_x": L_x,
                "L_up": L_up,
            })
        elif accept_policy == 'all':
            p_accept = 1
        else:
            raise ValueError(f"Unknown accept policy: {accept_policy}")

        # stochastically accept/reject
        if np.random.uniform(low=0, high=1) < p_accept:
            x[i] = s

        yield {
            "i": i,
            "t": t,
            "points": [lang.eval(p) for p in x],
            **record,
        }


def lang_log_pr(lang: Language, query: Tree, data: Tree) -> float:
    lang.fit([data], alpha=1.0)
    return lang.log_probability(query)


def novelty_scores(queries: np.ndarray, data: np.ndarray, n_neighbors=5) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data)
    dists, indices = knn.kneighbors(queries)
    dists = np.sum(dists, axis=1)
    return dists


def animate_points_v1(data_gen: Iterator, title: str, xlim: Tuple[int, int], ylim: Tuple[int, int], delay=200):
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    def update(frame):
        i = frame["i"]
        t = frame["t"]
        points = frame["points"]
        samples = frame["samples"]
        best_i = frame["best_i"]

        ax.set_title(f"{title}, frame: {t}")
        scatter.set_offsets(points + samples)
        colors = ["blue"] * len(points) + ["gray"] * len(samples)
        colors[i] = "red"
        colors[best_i + len(points)] = "green"
        scatter.set_color(colors)
        return scatter,

    return FuncAnimation(fig, update, frames=data_gen, blit=False, interval=delay)


def animate_points_v2(data_gen: Iterator, title: str, xlim: Tuple[int, int], ylim: Tuple[int, int], delay=200):
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    def update(frame):
        i = frame["i"]
        t = frame["t"]
        points = frame["points"]

        ax.set_title(f"{title}, frame: {t}")
        scatter.set_offsets(points)
        scatter.set_color(["red" if j == i else "blue" for j in range(len(points))])
        return scatter,

    return FuncAnimation(fig, update, frames=data_gen, blit=False, interval=delay)


def plot_subplots(data: List[dict], keys: List[str]):
    num_keys = len(keys)
    num_rows = int(num_keys ** 0.5)
    num_cols = (num_keys + num_rows - 1) // num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    for i, key in enumerate(keys):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.set_title(key)
        ax.plot([x[key] for x in data], label=key)

    plt.tight_layout()
    plt.show()


def transform_data(data: List[dict]) -> List[dict]:
    thresholds = [1e-5, 1e-10, 1e-20]

    def map_fn(d: dict) -> dict:
        L_x, L_up = d["L_x"], d["L_up"]
        sgn_x, logdet_x = np.linalg.slogdet(L_x)
        sgn_up, logdet_up = np.linalg.slogdet(L_up)

        sparsities = {}
        for thresh in thresholds:
            sparsities[f"sparsity(L_x, {thresh})"] = np.sum(L_x < thresh)
            sparsities[f"sparsity(L_x', {thresh})"] = np.sum(L_up < thresh)

        return {
            "log det(L_x)": sgn_x * logdet_x,
            "log det(L_x')": sgn_up * logdet_up,
            **sparsities,
        }

    return list(map(map_fn, data))


def test_large_mat_dets():
    with util.Timing("large_mat_dets"):
        B = np.random.rand(10_000, 10_000)
        M = np.matmul(B.transpose(), B)
        det = np.linalg.det(M)
    print(det)


if __name__ == "__main__":
    N_STEPS = 2000
    N_SAMPLES = 1
    POPN_SIZE = 1000
    FIT_POLICY = "single"
    ACCEPT_POLICY = "dpp"

    generator = tqdm(dpp_points_single_sample(
        lang=point.RealPoint(xlim=10, ylim=10),
        n=POPN_SIZE,
        fit_policy=FIT_POLICY,
        accept_policy=ACCEPT_POLICY,
        n_steps=N_STEPS,
    )
    points = list(tqdm(generator, total=N_STEPS))
    data = transform_data(points)
    plot_subplots(data, keys=sorted(data[0].keys() - {"i", "t", "points", "L_x", "L_up"}))
    anim = animate_points_v2(
        points,
        title=f"N={POPN_SIZE}, samples={N_SAMPLES}, fit_policy={FIT_POLICY}, accept={ACCEPT_POLICY}, steps={N_STEPS}",
        xlim=(-10, 10),
        ylim=(-10, 10),
        # delay=100,
    )
    ts = util.timestamp()
    anim.save(f"../out/anim/ddp-v2-{ts}.mp4")

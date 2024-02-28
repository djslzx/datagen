from typing import List, Iterator, Tuple, Iterable, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from sklearn.manifold import MDS

from lang.tree import Language, Tree
from lang import lindenmayer, point, arc
import featurizers as feat
import util


def mcmc_points_roundrobin_multisample(
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
            L_x = np.matmul(x_features, x_features.T)
            updated_features = x_features.copy()
            updated_features[i] = sample_features[best_i]
            L_updated = np.matmul(updated_features, updated_features.T)
            det_x = np.linalg.det(L_x)
            det_s = np.linalg.det(L_updated)

            # assert det_x > 0 and det_s > 0, f"det_x: {det_x}, det_s: {det_s}"
            log_f = np.log(det_s) - np.log(det_x)

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
                "log q(x|x')/(x'|x)": log_q,
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


def mcmc_points_roundrobin(
        lang: point.RealPoint,
        x_init: np.ndarray[Tree],
        n: int,
        fit_policy: str,
        accept_policy: str,
        kernel_type: str,
        n_steps: int,
        gamma=1,
):
    x = x_init
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
        up_feat = x_feat.copy()
        up_feat[i] = s_feat

        record = {}
        if accept_policy == 'dpp':
            # compute accept probability
            # (1) log f(x') - log f(x) = log det(L_x') - log det(L_x)
            if kernel_type == 'linear':
                # L_x[i,j] = <phi_i, phi_j>
                L_x = np.matmul(x_feat, x_feat.T)
                L_up = np.matmul(up_feat, up_feat.T)
            elif kernel_type == 'rbf':
                # L_x[i,j] = exp(-gamma * ||phi_i - phi_j||^2)
                L_x = np.exp(-gamma * np.linalg.norm(x_feat[:, None] - x_feat[None], axis=-1) ** 2)
                L_up = np.exp(-gamma * np.linalg.norm(up_feat[:, None] - up_feat[None], axis=-1) ** 2)
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")

            logdet_up = logdet(L_up)
            logdet_x = logdet(L_x)
            log_f = logdet_up - logdet_x

            # (2) log q(x|x') - log q(x'|x)
            log_q = lang_log_pr(lang, x[i], s) - \
                    lang_log_pr(lang, s, x[i])

            # (3) A = min(1, .) => log A = min(0, .)
            log_accept = np.min([0, log_f + log_q])

            record.update({
                "L_x": L_x,
                "L_up": L_up,
                "log det L_x": logdet_x,
                "log det L_up": logdet_up,
                "log f(x')/f(x)": log_f,
                "log q(x|x')/q(x'|x)": log_q,
                "log A(x',x)": log_accept,
            })
        elif accept_policy == 'energy':
            # energy-based acceptance probability:
            # log P(x) = -sum_i,j exp -d(x_i, x_j)
            log_f = fast_energy_update(x_feat, up_feat, i)
            # assert np.isclose(log_f, fast_log_f), f"log_f: {log_f}, fast_log_f: {fast_log_f}"

            # log q(x|x') - log q(x'|x)
            log_q = lang_log_pr(lang, x[i], s) - \
                    lang_log_pr(lang, s, x[i])

            # log A = min(1, log f(x') - log f(x) + log q(x|x') - log q(x'|x))
            log_accept = np.min([0, log_f + log_q])

            record.update({
                "log f(x')/f(x)": log_f,
                "log q(x|x')/q(x'|x)": log_q,
                "log A(x',x)": log_accept,
            })
        elif accept_policy == 'all':
            log_accept = 0
        else:
            raise ValueError(f"Unknown accept policy: {accept_policy}")

        # stochastically accept/reject
        u = np.random.uniform()
        while u == 0:
            u = np.random.uniform()
        if np.log(u) < log_accept:
            x[i] = s

        yield {
            "i": i,
            "t": t,
            "points": [lang.eval(p) for p in x],
            "x_feat": x_feat,
            "s_feat": s_feat,
            **record,
        }


def mcmc_lang_rr(
        lang: Language,
        x_init: List[Tree],
        n: int,
        n_steps: int,
        fit_policy: str,
        accept_policy: str,
        gamma=1,
):
    x = x_init
    x_feat = lang.extract_features(x)

    for t in range(n_steps):
        i = t % n

        if fit_policy == 'all':
            lang.fit(x, alpha=1.0)
        elif fit_policy == 'single':
            lang.fit([x[i]], alpha=1.0)
        else:
            raise ValueError(f"Unknown fit policy: {fit_policy}")

        # sample and featurize
        s = lang.sample()
        s_feat = lang.extract_features([s])[0]
        up_feat = x_feat.copy()
        up_feat[i] = s_feat

        # accept policy only affects log_f
        if accept_policy == "dpp":
            log_f = dpp_update(x_feat, up_feat, gamma)
        elif accept_policy == "energy":
            log_f = fast_energy_update(x_feat, up_feat, i)
        elif accept_policy == "all":
            log_f = 0
        else:
            raise ValueError(f"Unknown accept policy: {accept_policy}")

        log_q = lang_log_pr(lang, x[i], s) - lang_log_pr(lang, s, x[i])
        log_accept = np.min([0, log_f + log_q])

        # stochastically accept/reject
        u = np.random.uniform()
        while u == 0:
            u = np.random.uniform()
        if np.log(u) < log_accept:
            x[i] = s
            x_feat[i] = s_feat

        yield {
            "i": i,
            "t": t,
            "x": x,
            "s": s,
            "x_feat": x_feat.copy(),
            "s_feat": s_feat.copy(),
            "log f(x')/f(x)": log_f,
            "log q(x|x')/q(x'|x)": log_q,
            "log A(x',x)": log_accept,
        }


def logdet(m: np.ndarray) -> float:
    return np.prod(np.linalg.slogdet(m))


def lang_log_pr(lang: Language, query: Tree, data: Tree) -> float:
    lang.fit([data], alpha=1.0)
    return lang.log_probability(query)


def lang_log_pr_multi(lang: Language, query: List[Tree], data: List[Tree]) -> float:
    lang.fit(data, alpha=1.0)
    return sum(lang.log_probability(q) for q in query)


def novelty_scores(queries: np.ndarray, data: np.ndarray, n_neighbors=5) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data)
    dists, indices = knn.kneighbors(queries)
    dists = np.sum(dists, axis=1)
    return dists


def slow_energy_update(x_feat: np.ndarray, up_feat: np.ndarray) -> float:
    log_p_up = -np.sum(np.exp(-np.linalg.norm(up_feat[:, None] - up_feat[None], axis=-1)))
    log_p_x = -np.sum(np.exp(-np.linalg.norm(x_feat[:, None] - x_feat[None], axis=-1)))
    return log_p_up - log_p_x


def fast_energy_update(x_feat: np.ndarray, up_feat: np.ndarray, i: int) -> float:
    # log f(x') - log f(x) =
    #     sum_{i != k} exp -d(x_i', x_k') - exp -d(x_i, x_k)
    #   + sum_{j != k} exp -d(x_k', x_j') - exp -d(x_k, x_j)
    # = 2 * sum_{i != k} exp -d(x_i', x_k') - exp -d(x_i, x_k) by symmetry
    return 2 * np.sum(-np.exp(-np.linalg.norm(up_feat - up_feat[i], axis=-1))
                      + np.exp(-np.linalg.norm(x_feat - x_feat[i], axis=-1)))


def dpp_update(x_feat: np.ndarray, up_feat: np.ndarray, gamma: float) -> float:
    L_x = np.exp(-gamma * np.linalg.norm(x_feat[:, None] - x_feat[None], axis=-1) ** 2)
    L_up = np.exp(-gamma * np.linalg.norm(up_feat[:, None] - up_feat[None], axis=-1) ** 2)
    return logdet(L_up) - logdet(L_x)


def animate_points(data_gen: Iterable, title: str, xlim: Optional[Tuple[int, int]], ylim: Optional[Tuple[int, int]],
                   delay=200):
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    def update(frame):
        t = frame["t"]
        points = frame["points"]

        # check that points are in 2d
        assert points.shape[1] == 2, f"points.shape: {points.shape}"

        ax.set_title(f"{title}, frame: {t}")
        ax.title.set_fontsize(8)
        scatter.set_offsets(points)

        if xlim is None:
            ax.set_xlim(min(p[0] for p in points), max(p[0] for p in points))
        if ylim is None:
            ax.set_ylim(min(p[1] for p in points), max(p[1] for p in points))

        return scatter,

    return FuncAnimation(fig, update, frames=data_gen, blit=False, interval=delay)


def animate_matrix_spy(data_gen: Iterable, precision, delay=200):
    # animation where each frame is a spy plot of a matrix
    fig = plt.figure(figsize=(10, 5))

    def update(frame):
        t = frame["t"]
        L_x = frame["L_x"]
        L_up = frame["L_up"]

        plt.clf()  # Clear current plot

        # Plot the first matrix
        plt.subplot(1, 2, 1)
        plt.spy(L_x, precision=precision)
        plt.title(f'L_x @ {t}')

        # Plot the second matrix
        plt.subplot(1, 2, 2)
        plt.spy(L_up, precision=precision)
        plt.title(f'L_up @ {t}')

    return FuncAnimation(fig, update, frames=data_gen, blit=False, interval=delay)


def plot_v_subplots(data: List[dict], keys: List[str]):
    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, 1, figsize=(12, 2 * n_keys))

    for ax, key in zip(axes, keys):
        ax.set_title(key)
        ax.plot([x[key] for x in data], label=key)
        if key.startswith("log"):
            ax.set_yscale("symlog")
        if key.startswith("sparsity"):
            ax.set_ylim(0, 1)
            ax.set_ylabel("sparsity")

    plt.tight_layout()
    return fig


def transform_data(data: List[dict]) -> List[dict]:
    threshold = 1e-10
    rm_keys = {
        "i",
        "t",
        "x", "s",
        "x_feat", "s_feat",
        "points",
        "L_x", "L_up",
        "log q(x|x')/q(x'|x)",
        "log det L_x",
        "log A(x',x)"
    }

    def map_fn(t: int, d: dict) -> dict:
        # compute sparsities
        sparsities = {}
        try:
            L_x, L_up = d["L_x"], d["L_up"]
            # sparsities[f"sparsity(L_x, {threshold})"] = np.sum(L_x < threshold) / L_x.size
            sparsities[f"sparsity(L_up, {threshold})"] = np.sum(L_up < threshold) / L_up.size
        except KeyError:
            pass

        # knn of new sample point
        x_feat = d["x_feat"]
        s_feat = d["s_feat"]
        dists = novelty_scores(queries=s_feat[None], data=x_feat)
        d["knn_dist"] = dists[0]

        # measure overlap of i-th embedding wrt rest of embeddings
        i = d["i"]
        d["overlap"] = np.sum(np.isclose(np.delete(x_feat, i, axis=0), x_feat[i], atol=1e-5))

        # rename embeddings as points
        d["points"] = d["x_feat"]

        # measure avg radius of all embeddings in d["points"]
        d["mean radius"] = np.mean(np.linalg.norm(np.array(d["points"])[:, None] -
                                                  np.array(d["points"]),
                                                  axis=-1))

        # accept probability
        d["A(x',x)"] = np.exp(d["log A(x',x)"])

        # average accept probability over time
        # show how mean changes over time
        d["mean A(x',x)"] = np.sum([x["A(x',x)"] for x in data[:t + 1]]) / (t + 1)

        # filter keys
        d = {k: v for k, v in d.items() if k not in rm_keys}

        return {
            **d,
            **sparsities,
        }

    return [map_fn(i, d) for i, d in enumerate(data)]


def transform_points(data: List[dict]) -> List[dict]:
    # transform points from n dimensions to 2 dimensions
    n_dim = len(data[0]["points"][0])
    if n_dim < 2:
        raise ValueError(f"Expected n_dim >= 2, got n_dim: {n_dim}")
    elif n_dim == 2:
        return data
    else:
        # use MDS to reduce dimensionality
        mds = MDS(n_components=2)
        n_points = len(data[0]["points"])
        all_points = [p for d in data for p in d["points"]]
        points_2d = mds.fit_transform(all_points)
        return [{**d, "points": points_2d[i * n_points:(i + 1) * n_points]}
                for i, d in enumerate(data)]


def test_large_mat_dets():
    with util.Timing("large_mat_dets"):
        B = np.random.rand(10_000, 10_000)
        M = np.matmul(B.T, B)
        det = np.linalg.det(M)
    print(det)


def main(
        id: int,
        n_steps: int,
        popn_size: int,
        fit_policy: str,
        accept_policy: str,
        run: int,
        spread=1.0,
        animate=True,
        spy=False,
):
    lim = None
    # # point domain
    # lang = point.RealPoint(lim=lim, std=1)
    # coords = np.random.uniform(size=(n, 2)) * spread
    # x_init = [lang.make_point(a, b) for a, b in coords]

    # lsystem domain
    lang = lindenmayer.LSys(
        kind="deterministic",
        featurizer=feat.ResnetFeaturizer(),
        step_length=3,
        render_depth=4,
        n_rows=128,
        n_cols=128,
        aa=True,
        vary_color=False,
    )
    lsystems = [
        "20;F;F~F",
        "90;F;F~FF",
        "45;F[+F][-F]FF;F~FF",
        "60;F+F-F;F~F+FF",
        "60;F;F~F[+F][-F]F",
        "90;F-F-F-F;F~F+FF-FF-F-F+F+FF-F-F+F+FF+FF-F",
        "90;-F;F~F+F-F-F+F",
        "90;F-F-F-F;F~FF-F-F-F-F-F+F",
        "90;F-F-F-F;F~FF-F-F-F-FF",
        "90;F-F-F-F;F~FF-F+F-F-FF",
        "90;F-F-F-F;F~FF-F--F-F",
        "90;F-F-F-F;F~F-FF--F-F",
        "90;F-F-F-F;F~F-F+F-F-F",
        "20;F;F~F[+F]F[-F]F",
        "20;F;F~F[+F]F[-F][F]",
        "20;F;F~FF-[-F+F+F]+[+F-F-F]",
        # "90;F;F~F[+F]F[-F]F",
        # "90;F;F~F[+F]F[-F][F]",
        # "20;F;F~FF-[-F+F+F]+[+F-F-F]",
        # "20;F;F~FF+[+F-F-F]-[-F+F+F]",
        # "60;F;F~F[+F]F[-F]F",
        # "60;F;F~F[+F]F[-F][F]",
        # "45;F;F~FF-[-F+F+F]+[+F-F-F]",
        # "45;F;F~FF+[+F-F-F]-[-F+F+F]",
    ]
    x_init = [lang.parse(lsys) for lsys in lsystems]
    # # draw x_init
    # for lsys in lsystems:
    #     img = lang.eval(lang.parse(lsys))
    #     plt.imshow(img)
    #     plt.show()

    # init generator
    # generator = mcmc_points_roundrobin(
    generator = mcmc_lang_rr(
        lang=lang,
        x_init=x_init,
        n=popn_size,
        n_steps=n_steps,
        fit_policy=fit_policy,
        accept_policy=accept_policy,
    )
    raw_data = list(tqdm(generator, total=n_steps))

    title = (f"N={popn_size}"
             f",fit={fit_policy}"
             f",accept={accept_policy}"
             f",steps={n_steps}"
             f",spread={spread}"
             f",run={run}")

    # make run directory
    try:
        util.mkdir(f"../out/dpp/{id}/")
    except FileExistsError:
        pass

    util.mkdir(f"../out/dpp/{id}/{title}")
    dirname = f"../out/dpp/{id}/{title}"

    if accept_policy in {"dpp", "energy"}:
        data = transform_data(raw_data)
        keys = sorted(data[0].keys() - {"i", "t", "points", "L_x", "L_up", "s_feat", "x_feat"})
        fig = plot_v_subplots(data, keys)
        fig.savefig(f"{dirname}/plot.png")
        plt.cla()

    # Save data
    print("Saving data...")
    np.save(f"{dirname}/data.npy", np.array(raw_data))

    # Save animation
    if animate:
        if lim is not None:
            xlim = lang.lim
            ylim = lang.lim
        else:
            xlim = None
            ylim = None
        points = transform_points(raw_data)
        anim = animate_points(
            points,
            title=title,
            xlim=xlim,
            ylim=ylim,
            delay=100,
        )
        print("Saving animation...")
        anim.save(f"{dirname}/anim.mp4")

    # Save spy animation
    if spy:
        spy_anim = animate_matrix_spy(raw_data, delay=100, precision=1e-10)
        print("Saving spy animation...")
        spy_anim.save(f"{dirname}/spy.mp4")


if __name__ == "__main__":
    N_STEPS = [10 * 2]
    POPN_SIZE = [10]
    ACCEPT_POLICY = ["energy", "dpp"]
    FIT_POLICY = ["all", "single"]
    N_RUNS = 1
    SPREAD = [1]

    ts = util.timestamp()
    for t in N_STEPS:
        for n in POPN_SIZE:
            for accept in ACCEPT_POLICY:
                for fit in FIT_POLICY:
                    for spread in SPREAD:
                        for run in range(N_RUNS):
                            main(
                                id=ts,
                                n_steps=t,
                                popn_size=n,
                                fit_policy=fit,
                                accept_policy=accept,
                                run=run,
                                spread=spread,
                                animate=True,
                                spy=False,
                            )

import os
import pdb
from typing import List, Iterator, Tuple, Iterable, Optional, Set
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection
import featurizers as feat
import argparse

from lang.tree import Language, Tree
from lang import lindenmayer, point, arc
import util


def mcmc_lang_rr(
        lang: Language,
        x_init: List[Tree],
        n: int,
        n_steps: int,
        fit_policy: str,
        accept_policy: str,
        gamma=1,
        length_cap=50,
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
        s = lang.samples(n_samples=1, length_cap=length_cap)[0]
        s_feat = lang.extract_features([s])[0]
        up_feat = x_feat.copy()
        up_feat[i] = s_feat

        # accept policy only affects log_f
        if accept_policy == "dpp":
            log_f = dpp_rbf_update(x_feat, up_feat, gamma)
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
            "x": [lang.to_str(p) for p in x],
            "s": lang.to_str(s),
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


def knn_dist_sum(queries: np.ndarray, data: np.ndarray, n_neighbors=5) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data)
    dists, indices = knn.kneighbors(queries)
    dists = np.sum(dists, axis=1)
    return dists


def slow_energy_update(x_feat: np.ndarray, up_feat: np.ndarray) -> float:
    log_p_up = -np.sum(np.exp(-np.linalg.norm(up_feat[:, None] - up_feat[None], axis=-1)))
    log_p_x = -np.sum(np.exp(-np.linalg.norm(x_feat[:, None] - x_feat[None], axis=-1)))
    return log_p_up - log_p_x


def fast_energy_update(x_feat: np.ndarray, up_feat: np.ndarray, k: int) -> float:
    # log f(x') - log f(x) =
    #     sum_{i != k} exp -d(x_i', x_k') - exp -d(x_i, x_k)
    #   + sum_{j != k} exp -d(x_k', x_j') - exp -d(x_k, x_j)
    # = 2 * sum_{i != k} exp -d(x_i', x_k') - exp -d(x_i, x_k) by symmetry
    return 2 * np.sum(-np.exp(-np.linalg.norm(up_feat - up_feat[k], axis=-1))
                      + np.exp(-np.linalg.norm(x_feat - x_feat[k], axis=-1)))


def dpp_rbf_update(x_feat: np.ndarray, up_feat: np.ndarray, gamma: float) -> float:
    L_x = np.exp(-gamma * np.linalg.norm(x_feat[:, None] - x_feat[None], axis=-1) ** 2)
    L_up = np.exp(-gamma * np.linalg.norm(up_feat[:, None] - up_feat[None], axis=-1) ** 2)
    return logdet(L_up) - logdet(L_x)


def animate_points(
        data_gen: Iterable[np.ndarray],
        title: str,
        xlim: Optional[Tuple[int, int]],
        ylim: Optional[Tuple[int, int]],
        delay=200
):
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_box_aspect(1)

    @util.count_calls
    def update(points):
        # check that points are in 2d
        assert points.shape[1] == 2, f"points.shape: {points.shape}"

        ax.set_title(f"{title}, frame: {update.calls}")
        ax.title.set_fontsize(8)
        scatter.set_offsets(points)

        if xlim is None:
            ax.set_xlim(min(p[0] for p in points), max(p[0] for p in points))
        if ylim is None:
            ax.set_ylim(min(p[1] for p in points), max(p[1] for p in points))

        return scatter,

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


def plot_square_subplots(images: np.ndarray, title: str):
    n_images = len(images)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def analyze_run_iteration(d: dict, threshold: float) -> dict:
    # compute sparsity
    sparsity = {}
    try:
        L_up = d["L_up"]
        sparsity[f"sparsity(L_up, {threshold})"] = np.sum(L_up < threshold) / L_up.size
    except KeyError:
        pass

    # knn of new sample point
    x_feat = d["x_feat"]
    s_feat = d["s_feat"]
    dists = knn_dist_sum(queries=s_feat[None], data=x_feat, n_neighbors=min(5, len(x_feat)))
    d["knn_dist"] = dists[0]

    # measure overlap of i-th embedding wrt rest of embeddings
    i = d["i"]
    d["overlap"] = np.sum(np.all(np.isclose(x_feat[i],
                                            np.delete(x_feat, i, axis=0),
                                            atol=1e-5),
                                 axis=0))

    # measure avg radius of all embeddings in d["points"]
    d["mean radius"] = np.mean(np.linalg.norm(np.array(d["x_feat"])[:, None] -
                                              np.array(d["x_feat"]),
                                              axis=-1))

    # program sample length
    d["sample length"] = len(d["s"])

    # accept probability
    d["A(x',x)"] = np.exp(d["log A(x',x)"])

    return {
        **d,
        **sparsity,
    }


def transform_data(data: List[dict], verbose=False) -> Iterator[dict]:
    threshold = 1e-10
    rm_keys = {
        "i",
        "t",
        "x", "s",
        "x_feat", "s_feat",
        # "log q(x|x')/q(x'|x)",
        "log det L_x",
        "log A(x',x)"
    }

    if verbose:
        return (analyze_run_iteration(d, threshold, rm_keys) for i, d in
                enumerate(tqdm(data, desc="Transforming data")))
    else:
        return (analyze_run_iteration(d, threshold, rm_keys) for i, d in enumerate(data))


def test_large_mat_dets():
    with util.Timing("large_mat_dets"):
        B = np.random.rand(10_000, 10_000)
        M = np.matmul(B.T, B)
        det = np.linalg.det(M)
    print(det)


def npy_to_images(lang: Language, npy_dir: str, img_dir: str):
    """
    Read npy data files in `npy_dir` directory and write image renders
    of npy files to `img_dir`.
    """
    # iterate through .npy files in npy_dir, checking extension
    seen = set()
    for filename in tqdm(sorted(os.listdir(npy_dir))):
        if filename.endswith(".npy"):
            # load npy file
            frame = np.load(os.path.join(npy_dir, filename), allow_pickle=True).tolist()

            # assume that data is a dict -- a single frame generated by mcmc_lang_rr
            assert isinstance(frame, dict), f"Expected dict, got {type(frame)}"

            # parse n from part-n
            basename = os.path.splitext(filename)[0]
            n = int(basename.split("-")[1])

            # render all programs in x for gen 0, then only render the i-th program if we haven't seen it before
            if n == 0:
                for i, p in enumerate(frame["x"]):
                    seen.add(p)
                    tree = lang.parse(p)
                    img = lang.eval(tree)
                    plt.imshow(img)
                    plt.title(p)
                    plt.savefig(os.path.join(img_dir, f"{n}-{i:06d}.png"))
                    plt.clf()
            else:
                i = frame["i"]
                p = frame["x"][i]
                if p not in seen:
                    seen.add(p)
                    tree = lang.parse(frame["x"][i])
                    img = lang.eval(tree)
                    plt.imshow(img)
                    plt.title(p)
                    plt.savefig(os.path.join(img_dir, f"{n}-{i:06d}.png"))
                    plt.clf()


def parse_part_n(filename: str) -> int:
    basename = os.path.splitext(filename)[0]
    return int(basename.split("-")[1])


def npy_to_batched_images(lang: Language, npy_dir: str, img_dir: str):
    """
    Read npy data files in `npy_dir` directory and write image renders
    at every N-th iteration, where N is the size of the MCMC matrix.
    """
    filenames = [name for name in os.listdir(npy_dir) if name.endswith(".npy")]
    filenames_with_n = sorted([(parse_part_n(name), name) for name in filenames])

    # extract N from part-0.npy
    data_0 = np.load(os.path.join(npy_dir, filenames_with_n[0][1]), allow_pickle=True).tolist()
    N = len(data_0["x"])

    # get every N-th iteration, starting from 0
    for n, filename in tqdm(filenames_with_n):
        if n % N == 0:
            frame = np.load(os.path.join(npy_dir, filename), allow_pickle=True).tolist()
            save_path = os.path.join(img_dir, f"{n}.png")
            plot_batched_images(lang,
                                programs=frame["x"],
                                save_path=save_path,
                                title=f"gen-{n}")


def plot_batched_images(lang: Language, programs: List[str], save_path: str, title: str):
    """
    Plot all programs in `programs` in a single plot and save to `save_path` with title `title`.
    """
    images = []
    for i, p in enumerate(programs):
        tree = lang.parse(p)
        img = lang.eval(tree)
        images.append(img)
    fig = plot_square_subplots(np.stack(images), title=title)
    fig.savefig(save_path)
    plt.close(fig)


def run_search_iter(
        id: int,
        domain: str,
        n_steps: int,
        popn_size: int,
        fit_policy: str,
        accept_policy: str,
        run: int,
        save_data=True,
        spread=1.0,
        sigma=0.,
        animate_embeddings=True,
        anim_stride=1,
        analysis_stride=1,
        plot=False,
):
    lim = None

    if domain == "point":
        lang = point.RealPoint(lim=lim, std=1)
        coords = np.random.uniform(size=(popn_size, 2)) * spread
        x_init = [lang.make_point(a, b) for a, b in coords]
    elif domain == "lsystem":
        lang = lindenmayer.LSys(
            kind="deterministic",
            featurizer=feat.ResnetFeaturizer(sigma=sigma),
            step_length=3,
            render_depth=4,
            n_rows=128,
            n_cols=128,
            aa=True,
            vary_color=False,
        )
        # lsystem initial popn: x0 + samples from G(x0)
        lsystem_strs = [
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
        ]
        lsystems = [lang.parse(lsys) for lsys in lsystem_strs]
        lang.fit(lsystems, alpha=1.0)
        x_init = lsystems + lang.samples(popn_size - len(lsystems), length_cap=50)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # init generator
    generator = mcmc_lang_rr(
        lang=lang,
        x_init=x_init,
        n=popn_size,
        n_steps=n_steps,
        fit_policy=fit_policy,
        accept_policy=accept_policy,
    )
    title = (f"N={popn_size}"
             f",fit={fit_policy}"
             f",accept={accept_policy}"
             f",steps={n_steps}"
             f",spread={spread}"
             f",run={run}"
             f",sigma={sigma}")

    # make run directory
    try:
        util.mkdir(f"../out/dpp/{id}/")
    except FileExistsError:
        pass

    util.mkdir(f"../out/dpp/{id}/{title}")
    dirname = f"../out/dpp/{id}/{title}"
    util.mkdir(f"{dirname}/data/")
    util.mkdir(f"{dirname}/images/")

    anim_coords = []  # save 2d embeddings for animation
    srp = SparseRandomProjection(n_components=2)
    srp.fit(np.random.rand(popn_size, lang.featurizer.n_features))

    analysis_data = []

    for i, d in enumerate(tqdm(generator, total=n_steps, desc="Generating data")):
        # Save data
        if save_data:
            np.save(f"{dirname}/data/part-{i:06d}.npy", d, allow_pickle=True)

        # Plot images
        if plot and domain == "lsystem" and i % analysis_stride == 0:
            plot_batched_images(lang, d["x"], f"{dirname}/images/{i:06d}.png", title=f"gen-{i}")

        # Analysis
        if i % analysis_stride == 0:
            analysis_data.append(analyze_run_iteration(d, threshold=1e-10))

        # Animation
        if animate_embeddings and i % anim_stride == 0:
            x_feat = d["x_feat"]
            dim = x_feat.shape[1]
            if dim < 2:
                # if x_feat is 1d, add a dimension to make it 2d
                coords = np.stack([x_feat, np.zeros_like(x_feat)], axis=-1)
            elif dim == 2:
                coords = x_feat
            else:
                coords = srp.transform(d["x_feat"])

            anim_coords.append(coords)

    if animate_embeddings:
        anim = animate_points(
            anim_coords,
            title=title,
            xlim=lim,
            ylim=lim,
            delay=100,
        )
        print("Saving animation...")
        anim.save(f"{dirname}/embed.mp4")

    # Plot analysis
    for i, d in enumerate(analysis_data):
        d["mean A(x',x)"] = np.sum([x["A(x',x)"] for x in analysis_data[:i + 1]]) / (i + 1)

    keys = sorted(analysis_data[0].keys() - {"i", "t", "s", "x", "s_feat", "x_feat"})
    fig = plot_v_subplots(analysis_data, keys)
    fig.savefig(f"{dirname}/plot.png")
    plt.cla()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["search", "npy-to-images"])
    p.add_argument("--domain", type=str, required=True, choices=["point", "lsystem"])
    p.add_argument("--npy-dir", type=str)
    p.add_argument("--img-dir", type=str)
    p.add_argument("--batched", action="store_true", default=False)
    p.add_argument("--debug", action="store_true", default=False)
    args = p.parse_args()

    if args.mode == "search":
        n_steps = 100 * 10 * 100  # 100 iters * 10x samples * 100 popn size
        popn_size = 100

        if args.debug:
            n_steps = 100

        ts = util.timestamp()
        for fit in ["all", "single"]:
            for sigma in [0., 3.]:
                run_search_iter(
                    id=ts,
                    domain=args.domain,
                    n_steps=n_steps,
                    popn_size=popn_size,
                    fit_policy=fit,
                    accept_policy="energy",
                    run=0,
                    spread=1,
                    save_data=True,
                    animate_embeddings=True,
                    sigma=sigma,
                    plot=True,
                    analysis_stride=1000,
                    anim_stride=popn_size,
                )
    elif args.mode == "npy-to-images":
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
        if args.batched:
            npy_to_batched_images(lang, args.npy_dir, args.img_dir)
        else:
            npy_to_images(lang, args.npy_dir, args.img_dir)

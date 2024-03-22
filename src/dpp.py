import os
import sys
from typing import List, Iterator, Tuple, Iterable, Optional, Set, Union
import numpy as np
import yaml
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection
import featurizers as feat
import wandb

from lang.tree import Language, Tree
from lang import lindenmayer, point, arc
import util


def mcmc_lang_rr(
        lang: Language,
        x_init: List[Tree],
        popn_size: int,
        n_epochs: int,
        fit_policy: str,
        accept_policy: str,
        gamma=1,
        length_cap=50,
):
    """
    MCMC with target distribution f(x) and proposal distribution q(x'|x),
    chosen via f=accept_policy and q=fit_policy.

    We update x in a round-robin fashion, where x' differs from x at a
    single position i.  If fit_policy is "all", we fit the model to x;
    if it is "single", we fit the model to x[i].
    """

    assert fit_policy in {"all", "single"}
    assert accept_policy in {"dpp", "energy", "all"}

    x = x_init
    x_feat = lang.extract_features(x)

    for t in range(n_epochs):
        samples = []
        samples_feat = []

        for i in range(popn_size):
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

            # save samples
            samples.append(s)
            samples_feat.append(s_feat)

            # compute log f(x')/f(x)
            if accept_policy == "dpp":
                log_f = dpp_rbf_update(x_feat, up_feat, gamma)
            elif accept_policy == "energy":
                log_f = fast_energy_update(x_feat, up_feat, i)
            elif accept_policy == "all":
                log_f = 0
            else:
                raise ValueError(f"Unknown accept policy: {accept_policy}")

            # compute log q(x|x')/q(x'|x)
            if fit_policy == 'all':
                up = x.copy()
                up[i] = s
                log_q = lang_log_pr(lang, x[i], up) - lang_log_pr(lang, s, x)
            elif fit_policy == 'single':
                log_q = lang_log_pr(lang, x[i], s) - lang_log_pr(lang, s, x[i])
            else:
                raise ValueError(f"Unknown fit policy: {fit_policy}")

            log_accept = np.min([0, log_f + log_q])

            # stochastically accept/reject
            u = np.random.uniform()
            while u == 0:
                u = np.random.uniform()
            if np.log(u) < log_accept:
                x[i] = s
                x_feat[i] = s_feat

        yield {
            "t": t,
            "x": [lang.to_str(p) for p in x],
            "x'": [lang.to_str(p) for p in samples],
            "x_feat": x_feat.copy(),
            "x'_feat": samples_feat.copy(),
            "log f(x')/f(x)": log_f,
            "log q(x|x')/q(x'|x)": log_q,
            "log A(x',x)": log_accept,
        }


def mcmc_lang_full_step(
        lang: Language,
        x_init: List[Tree],
        popn_size: int,
        n_epochs: int,
        fit_policy: str,
        accept_policy: str,
        gamma=1,
        length_cap=50,
  ):
    """
    MCMC with target distribution f(x) and proposal distribution q(x'|x),
    chosen via f=accept_policy and q=fit_policy.

    At each iteration, x' consists of |x| independent samples from G(x).
    """
    assert fit_policy in {"all"}
    assert accept_policy in {"dpp", "energy"}

    x = x_init
    x_feat = lang.extract_features(x)

    for t in range(n_epochs):
        lang.fit(x, alpha=1.0)
        x_new = lang.samples(n_samples=popn_size, length_cap=length_cap)
        x_new_feat = lang.extract_features(x_new)

        # compute log f(x')/f(x)
        if accept_policy == "dpp":
            log_f = dpp_rbf_update(x_feat, x_new_feat, gamma)
        elif accept_policy == "energy":
            log_f = slow_energy_update(x_feat, x_new_feat)
        else:
            raise ValueError(f"Unknown accept policy: {accept_policy}")

        # compute log q(x|x')/q(x'|x)
        log_q_x_given_new = lang_log_prs(lang, x, x_new)
        log_q_new_given_x = lang_log_prs(lang, x_new, x)
        log_q = log_q_x_given_new.sum() - log_q_new_given_x.sum()
        wandb.log({
            "log q(x|x')": wandb.Histogram(log_q_x_given_new),
            "log q(x'|x)": wandb.Histogram(log_q_new_given_x),
        }, commit=False)

        log_accept = np.min([0, log_f + log_q])
        u = np.random.uniform()
        while u == 0:
            u = np.random.uniform()
        if np.log(u) < log_accept:
            x = x_new
            x_feat = x_new_feat

        yield {
            "t": t,
            "x": [lang.to_str(p) for p in x],
            "x_feat": x_feat.copy(),
            "x'": [lang.to_str(p) for p in x_new],
            "x'_feat": x_new_feat.copy(),
            "log f(x')/f(x)": log_f,
            "log q(x|x')/q(x'|x)": log_q,
            "log A(x',x)": log_accept,
        }


def logdet(m: np.ndarray) -> float:
    return np.prod(np.linalg.slogdet(m))


def lang_log_pr(lang: Language, query: Tree, data: Union[List[Tree], Tree]) -> float:
    data = [data] if not isinstance(data, list) else data
    lang.fit(data, alpha=1.0)
    return lang.log_probability(query)


def lang_log_prs(lang: Language, query: List[Tree], data: List[Tree]) -> np.ndarray:
    lang.fit(data, alpha=1.0)
    log_probs = np.array([lang.log_probability(q) for q in query])
    return log_probs


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


def analyzer_iter(d: dict, threshold: float) -> dict:
    # compute sparsity
    sparsity = {}
    try:
        L_up = d["L_up"]
        sparsity[f"sparsity(L_up, {threshold})"] = np.sum(L_up < threshold) / L_up.size
    except KeyError:
        pass

    # avg knn of new sample points
    x_feat = d["x_feat"]
    x_new_feat = d["x'_feat"]
    dists = knn_dist_sum(queries=x_new_feat, data=x_feat, n_neighbors=min(5, len(x_feat)))
    d["mean knn dist"] = dists.mean()

    # measure average overlap of embeddings within epsilon ball
    popn_size = len(x_feat)
    d["epsilon overlap"] = 1 / popn_size * np.sum(np.all(np.isclose(x_feat, x_feat[:, None], atol=1e-5), axis=-1))

    # embedding distances
    dists = np.linalg.norm(x_feat[:, None] - x_feat, axis=-1)
    d["mean dist"] = np.mean(dists)
    d["std dist"] = np.std(dists)
    d["max dist"] = np.max(dists)

    # average program length
    d["mean length"] = np.mean([len(p) for p in d["x"]])

    # accept probability
    d["A(x',x)"] = np.exp(d["log A(x',x)"])

    return {
        **d,
        **sparsity,
    }


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
            plot_batched_images(lang, frame["x"], save_path, title=f"gen-{n}")

            images = render_program_batch(lang, frame["x"])
            fig = util.plot_square_subplots(images, title=f"gen-{n}")
            fig.savefig(save_path)
            plt.close(fig)


def render_program_batch(lang: Language, programs: List[str]) -> np.ndarray:
    """
    Render all programs in `programs` and return a batch of images.
    """
    images = []
    for p in programs:
        tree = lang.parse(p)
        img = lang.eval(tree)
        images.append(img)
    return np.stack(images)


def render_program_batch_as_wandb_image(lang: Language, programs: List[str], caption: str = "") -> wandb.Image:
    """
    Render all programs in `programs` and return as a single wandb.Image.
    """
    images = render_program_batch(lang, programs)
    image = util.combine_images(images)
    return wandb.Image(image, caption=caption)


def plot_batched_images(lang: Language, programs: List[str], save_path: str, title: str):
    """
    Plot all programs in `programs` in a single plot and save to `save_path` with title `title`.
    """
    images = render_program_batch(lang, programs)
    fig = util.plot_square_subplots(images, title=title)
    fig.savefig(save_path)
    plt.close(fig)


def run_point_search(popn_size: int, spread: float, lim=None):
    lang = point.RealPoint(lim=lim, std=1)
    coords = np.random.uniform(size=(popn_size, 2)) * spread
    x_init = [lang.make_point(a, b) for a, b in coords]


def run_lsys_search(config):
    expected_keys = {"x_init", "search", "featurizer", "render"}
    assert all(k in config for k in expected_keys), f"Expected {expected_keys}, got {set(config.keys())}"

    featurizer = feat.ResnetFeaturizer(**config.featurizer)
    lang = lindenmayer.LSys(
        kind="deterministic",
        featurizer=featurizer,
        **config.render,
    )
    lsystems = [lang.parse(lsys) for lsys in config.x_init]

    length_cap = config.search["length_cap"]
    popn_size = config.search["popn_size"]
    if popn_size < len(lsystems):
        x_init = lsystems[:popn_size]
    elif popn_size > len(lsystems):
        lang.fit(lsystems, alpha=1.0)
        x_init = lsystems + lang.samples(popn_size - len(lsystems), length_cap=length_cap)
    else:
        x_init = lsystems

    # init generator
    update_policy = config.search["update_policy"]
    epochs = config.search["epochs"]
    if update_policy == "rr":
        generator_fn = mcmc_lang_rr
    elif update_policy == "full_step":
        generator_fn = mcmc_lang_full_step
    else:
        raise ValueError(f"Unknown update policy: {update_policy}")

    fit_policy = config.search["fit_policy"]
    accept_policy = config.search["accept_policy"]
    generator = generator_fn(
        lang=lang,
        x_init=x_init,
        popn_size=popn_size,
        n_epochs=epochs,
        fit_policy=fit_policy,
        accept_policy=accept_policy,
        length_cap=length_cap,
    )

    # make run directory
    save_dir = f"../out/dpp/{wandb.run.id}/"
    try:
        util.mkdir(save_dir)
    except FileExistsError:
        pass

    util.mkdir(f"{save_dir}/data/")
    util.mkdir(f"{save_dir}/images/")

    wandb_process_data_epochs(lang=lang, generator=generator, popn_size=popn_size, n_epochs=epochs,
                              save=True, save_dir=save_dir)


def reduce_dim(x: np.ndarray, srp: SparseRandomProjection) -> np.ndarray:
    dim = x.shape[1]
    if dim < 2:
        # if x_feat is 1d, add a dimension to make it 2d
        coords = np.stack([x, np.zeros_like(x)], axis=-1)
    elif dim == 2:
        coords = x
    else:
        coords = srp.transform(x)
    return coords


def wandb_process_data_epochs(
        lang: Language,
        generator: Iterator[dict],
        popn_size: int,
        n_epochs: int,
        save: bool,
        save_dir: str,
):
    srp = SparseRandomProjection(n_components=2)
    srp.fit(np.random.rand(popn_size, lang.featurizer.n_features))
    for i, d in enumerate(tqdm(generator, total=n_epochs, desc="Generating data")):
        if save:
            np.save(f"{save_dir}/data/part-{i:06d}.npy", d, allow_pickle=True)
        analysis_data = analyzer_iter(d, threshold=1e-10)
        coords = reduce_dim(d["x_feat"], srp)
        coord_image = util.scatterplot_image(coords, figsize=3)
        log = {
            **d,
            **analysis_data,
            "step": i,
            "renders": render_program_batch_as_wandb_image(lang, d["x"]),
            "scatter": wandb.Image(coord_image),
        }
        rm_keys = {"x", "x'"}
        log = {k: v for k, v in log.items()
               if k not in rm_keys and not k.endswith("_feat")}
        wandb.log(log)


def local_process_data(
        lang: Language,
        generator: Iterator[dict],
        popn_size: int,
        n_steps: int,
        save_data: bool,
        dirname: str,
        plot: bool,
        domain: str,
        analysis_stride: int,
        anim_stride: int,
        animate_embeddings: bool,
        title: str,
        debug: bool,
):
    anim_coords = []  # save 2d embeddings for animation
    srp = SparseRandomProjection(n_components=2)
    srp.fit(np.random.rand(popn_size, lang.featurizer.n_features))

    analysis_data = []

    for i, d in enumerate(tqdm(generator, total=n_steps, desc="Generating data")):
        # Save data
        if save_data:
            np.save(f"{dirname}/data/part-{i:06d}.npy", d, allow_pickle=True)

        # Plot images
        if debug and domain == "lsystem" and i % analysis_stride == 0:
            plot_batched_images(lang, d["x"], f"{dirname}/images/{i:06d}.png", title=f"gen-{i}")

        # Analysis
        if i % analysis_stride == 0:
            analysis_data.append(analyzer_iter(d, threshold=1e-10))

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
        anim = util.animate_points(
            anim_coords,
            title=title,
            delay=100,
        )
        print("Saving animation...")
        anim.save(f"{dirname}/embed.mp4")

    # Plot analysis
    for i, d in enumerate(analysis_data):
        d["mean A(x',x)"] = np.sum([x["A(x',x)"] for x in analysis_data[:i + 1]]) / (i + 1)

    keys = sorted(analysis_data[0].keys() - {"i", "t", "s", "x", "s_feat", "x_feat"})
    fig = util.plot_v_subplots(analysis_data, keys)
    fig.savefig(f"{dirname}/plot.png")
    plt.cla()


if __name__ == "__main__":
    # p = argparse.ArgumentParser()
    # p.add_argument("--mode", type=str, required=True, choices=["search", "npy-to-images"])
    # p.add_argument("--domain", type=str, required=True, choices=["point", "lsystem"])
    # p.add_argument("--npy-dir", type=str)
    # p.add_argument("--img-dir", type=str)
    # p.add_argument("--wandb-sweep-config", type=str)
    # p.add_argument("--batched", action="store_true", default=False)
    # args = p.parse_args()

    sweep_config = "./configs/mcmc-lsystem.yaml"
    with open(sweep_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    wandb.init(project="dpp", config=config)
    config = wandb.config
    run_lsys_search(config)

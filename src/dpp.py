import os
import sys
from typing import List, Iterator, Tuple, Iterable, Optional, Set, Union, Generator, Any
import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection
import featurizers as feat
import wandb
import einops as ein

from lang.tree import Language, Tree
from lang import lindenmayer, point, arc
import util


class VectorArchive:
    def __init__(self, n: int, d: int):
        assert n > 0
        assert d > 0

        self.capacity = n
        self.dim = d
        self._vecs = np.zeros((n, d))
        self._tags = np.empty(n, dtype=object)
        self.n_entries = 0
        self.step = 0

    @staticmethod
    def from_vecs(n: int, vecs: np.ndarray, tags: List[Any]) -> "VectorArchive":
        assert vecs.ndim == 2, f"Expected 2D vector, but got {vecs.shape}"
        m, d = vecs.shape
        assert m == len(tags)

        archive = VectorArchive(n, d)
        for vec, tag in zip(vecs[:, None], tags):
            archive.add(vec, tag)
        return archive

    def add(self, vec: np.ndarray, tag: Any):
        """Extend archive by a single entry. Returns True if the vec was added to the archive."""
        assert vec.ndim == 2, f"Expected 2D vector, but got {vec.shape}"
        assert vec.shape[0] == 1
        assert vec.shape[1] == self.dim

        if self.n_entries < self.capacity:
            # append
            self._vecs[self.n_entries] = vec
            self._tags[self.n_entries] = tag
            self.n_entries += 1
        else:
            # probabilistically replace
            m = np.random.randint(0, self.step)
            if m < self.capacity:
                self._vecs[m] = vec
                self._tags[m] = tag
        self.step += 1

    def data(self) -> np.ndarray:
        return self._vecs[:self.n_entries]

    def metadata(self) -> np.ndarray:
        return self._tags[:self.n_entries]


def mcmc_lang_rr(
        lang: Language,
        x_init: List[Tree],
        popn_size: int,
        n_epochs: int,
        fit_policy: str,
        accept_policy: str,
        distance_metric: str,
        archive_size: int,
        archive_beta: float,
        length_cap=50,
        gamma=1,
) -> Iterator[dict]:
    """
    MCMC with target distribution f(x) and proposal distribution q(x'|x),
    chosen via f=accept_policy and q=fit_policy.

    We update x in a round-robin fashion, where x' differs from x at a
    single position i.  If fit_policy is "all", we fit the model to x;
    if it is "single", we fit the model to x[i].
    """

    assert fit_policy in {"all", "single", "none", "first"}
    assert accept_policy in {"dpp", "energy", "moment", "all"}
    assert distance_metric in {"cosine", "dot", "euclidean"}

    x = x_init.copy()
    x_feat = lang.extract_features(x)
    archive = VectorArchive.from_vecs(
        n=archive_size,
        vecs=x_feat,
        tags=[lang.to_str(tree) for tree in x]
    )

    if fit_policy == "first":
        lang.fit(x_init, alpha=1.0)

    for t in range(n_epochs):
        samples = []
        samples_feat = []
        sum_log_f = 0
        sum_log_q = 0
        sum_log_accept = 0

        for i in range(popn_size):
            if fit_policy == "all":
                lang.fit(x, alpha=1.0)
            elif fit_policy == "single":
                lang.fit([x[i]], alpha=1.0)
            elif isinstance(lang, point.RealMaze):
                # maze: allow points to spread based on current positions
                lang.update_allowed(x_feat)

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
            elif accept_policy == "moment":
                log_f = slow_fom_update(x_feat, up_feat)
            elif accept_policy == "all":
                log_f = np.inf
            else:
                raise ValueError(f"Unknown accept policy: {accept_policy}")

            # add archive correction term to target distribution
            if archive_beta > 0:
                archive.add(s_feat[None, :], tag=lang.to_str(s))
                log_f += archive_beta * archive_correction_update(x_feat, up_feat, archive.data(), i)

            # compute log q(x|x')/q(x'|x)
            if fit_policy == "all":
                up = x.copy()
                up[i] = s
                log_q = lang_log_pr(lang, x[i], up) - lang_log_pr(lang, s, x)
            elif fit_policy == "single":
                log_q = lang_log_pr(lang, x[i], s) - lang_log_pr(lang, s, x[i])
            else:
                log_q = 0

            log_accept = np.min([0, log_f + log_q])

            # stochastically accept/reject
            u = uniform_nonzero()
            if np.log(u) < log_accept:
                x[i] = s
                x_feat[i] = s_feat

            # track log probabilities
            sum_log_f += log_f
            sum_log_q += log_q
            sum_log_accept += log_accept

        yield {
            "t": t,
            "x": [lang.to_str(p) for p in x],
            "x'": [lang.to_str(p) for p in samples],
            "x_feat": x_feat.copy(),
            "x'_feat": samples_feat.copy(),
            "archive": archive.metadata(),
            "log f(x')/f(x)": sum_log_f / popn_size,
            "log q(x|x')/q(x'|x)": sum_log_q / popn_size,
            "log A(x',x)": sum_log_accept / popn_size,
        }


def uniform_nonzero() -> float:
    u = np.random.uniform()
    while u == 0:
        u = np.random.uniform()
    return u


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


def archive_correction_update(x_feat: np.ndarray, up_feat: np.ndarray, archive_feat: np.ndarray, k: int) -> float:
    # C(X', A) - C(X, A)
    #   = sum_{x in X'} min_{a in A} d(x, a)
    #     - sum_{x in X} min_{a in A} d(x, a)
    #   = min_{a in A} d(x_i', a) - min_{a in A} d(x_i, a)
    return (np.min(np.linalg.norm(archive_feat - up_feat[k], axis=-1))
            - np.min(np.linalg.norm(archive_feat - x_feat[k], axis=-1)))


def dpp_rbf_update(x_feat: np.ndarray, up_feat: np.ndarray, gamma: float) -> float:
    L_x = np.exp(-gamma * np.linalg.norm(x_feat[:, None] - x_feat[None], axis=-1) ** 2)
    L_up = np.exp(-gamma * np.linalg.norm(up_feat[:, None] - up_feat[None], axis=-1) ** 2)
    return logdet(L_up) - logdet(L_x)


def slow_fom_update(x_feat: np.ndarray, up_feat: np.ndarray) -> float:
    # update using first order moment:
    # log f(x) = sum_i d(x_i, mean(x))
    return np.sum(np.linalg.norm(up_feat - np.mean(up_feat, axis=0), axis=-1)) \
        - np.sum(np.linalg.norm(x_feat - np.mean(x_feat, axis=0), axis=-1))


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


def render_program_batch(lang: Language, programs: Iterable[str]) -> np.ndarray:
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
    image = util.combine_images_square(images)
    return wandb.Image(image, caption=caption)


def log_best_and_worst(
        k: int,
        lang: Language,
        current_programs: List[str],
        current_features: np.ndarray,
        prev_features: np.ndarray,
) -> dict:
    def summarize(samples: np.ndarray, scores: np.ndarray) -> wandb.Image:
        images = render_program_batch(lang, samples)
        image = util.combine_images_row(images)
        caption = "LTR: " + ", ".join(f"{score:.2e}" for score in scores)
        return wandb.Image(image, caption=caption)

    current_programs = np.array(current_programs)
    current_features = np.array(current_features)
    prev_features = np.array(prev_features)

    scores = knn_dist_sum(queries=current_features, data=prev_features, n_neighbors=5)

    i_best = np.argpartition(scores, -k)[-k:][::-1]
    i_worst = np.argpartition(scores, k)[:k]

    return {"best": summarize(current_programs[i_best], scores[i_best]),
            "worst": summarize(current_programs[i_worst], scores[i_worst])}


def plot_batched_images(lang: Language, programs: List[str], save_path: str, title: str):
    """
    Plot all programs in `programs` in a single plot and save to `save_path` with title `title`.
    """
    images = render_program_batch(lang, programs)
    fig = util.plot_square_subplots(images, title=title)
    fig.savefig(save_path)
    plt.close(fig)


def run_point_search(
        popn_size: int,
        save_dir: str,
        n_epochs: int,
        fit_policies: List[str],
        accept_policies: List[str],
        archive_beta: float,
        archive_size: int,
        spread=1,
):
    parse_lang = point.RealPoint()
    coords = (np.random.uniform(size=(popn_size, 2)) * spread) + 1
    x_init = [parse_lang.make_point(a, b) for a, b in coords]
    for fit_policy in fit_policies:
        for accept_policy in accept_policies:
            title = f"fit={fit_policy},accept={accept_policy}"
            local_dir = f"{save_dir}/{title}"
            util.mkdir(local_dir)

            lang = point.RealPoint(std=1)
            generator = mcmc_lang_rr(
                lang=lang,
                x_init=x_init,
                popn_size=popn_size,
                n_epochs=n_epochs,
                fit_policy=fit_policy,
                accept_policy=accept_policy,
                distance_metric="euclidean",
                archive_beta=archive_beta,
                archive_size=archive_size,
            )
            data = list(tqdm(generator, total=n_epochs))

            # plots
            data = [analyzer_iter(d, threshold=1e-10) for d in data]
            plot_keys = list(sorted(data[0].keys() - {"x", "x'", "t", "x_feat", "x'_feat", "archive"}))
            fig = util.plot_v_subplots(data, keys=plot_keys)
            fig.savefig(f"{local_dir}/plot.png")
            plt.cla()

            # save plot data
            df = pd.DataFrame(data)
            df.drop(columns=["x", "x'", "x_feat", "x'_feat", "archive"], inplace=True)
            df["fit policy"] = fit_policy
            df["accept policy"] = accept_policy
            df.to_json(f"{local_dir}/data.json")

            # animation
            init_feat = lang.extract_features(x_init)
            embeddings = [(0, init_feat)]
            embeddings += [(d["t"] + 1, d["x_feat"]) for d in data]
            anim = util.animate_points(
                embeddings,
                title=title,
                xlim=lang.xlim,
                ylim=lang.ylim,
            )
            anim.save(f"{local_dir}/embed.mp4")


def summarize_point_data(save_dir: str):
    # plot together
    dfs = []
    for dir in util.ls_subdirs(save_dir):
        df = pd.read_json(f"{save_dir}/{dir}/data.json")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(df)
    df["config"] = "fit=" + df["fit policy"] + ", accept=" + df["accept policy"]

    # plot summary
    cols = set(df.columns) - {"t", "config", "fit policy", "accept policy"}
    for col in cols:
        sns.relplot(x='t', y=col, hue='config', kind='line', data=df)
        plt.show()
        plt.close()

    return df


def run_maze_search(
        popn_size: int,
        save_dir: str,
        n_epochs: int,
        fit_policies: List[str],
        accept_policies: List[str],
        archive_size: int,
        archive_beta: float,
        spread=1,
):
    # lang = point.RealPoint(lim=lim, std=1)
    str_mask = [
        "####################",
        "#                  #",
        "####   ######      #",
        "#  ## ##           #",
        "#   ###            #",
        "###   #  ##        #",
        "# ##  ##  ###      #",
        "#  ## ###   ##     #",
        "#     # ###  ##    #",
        "#     #   ##  ##   #",
        "#     #    ##  #####",
        "#     #     ##  ####",
        "#                  #",
        "#          ##      #",
        "#         ##       #",
        "#        ##        #",
        "####################",
        # "#####################################",
        # "#_#_______#_______#_____#_________#_#",
        # "#_#_#####_#_###_#####_###_###_###_#_#",
        # "#_______#___#_#_____#_____#_#_#___#_#",
        # "#####_#_#####_#####_###_#_#_#_#####_#",
        # "#___#_#_______#_____#_#_#_#_#_____#_#",
        # "#_#_#######_#_#_#####_###_#_#####_#_#",
        # "#_#_______#_#_#___#_____#_____#___#_#",
        # "#_#######_###_###_#_###_#####_#_###_#",
        # "#_____#___#_#___#_#___#_____#_#_____#",
        # "#_###_###_#_###_#_#####_#_#_#_#######",
        # "#___#___#_#_#___#___#___#_#_#___#___#",
        # "#######_#_#_#_#####_#_###_#_###_###_#",
        # "#_____#_#_____#___#_#___#_#___#_____#",
        # "#_###_#_#####_###_#_###_###_#######_#",
        # "#_#___#_____#_____#___#_#_#_______#_#",
        # "#_#_#####_#_###_#####_#_#_#######_#_#",
        # "#_#_____#_#_#_#_#_____#_______#_#___#",
        # "#_#####_#_#_#_###_#####_#####_#_#####",
        # "#_#___#_#_#_____#_____#_#___#_______#",
        # "#_#_###_###_###_#####_###_#_#####_#_#",
        # "#_#_________#_____#_______#_______#_#",
        # "#####################################",
        # "##########",
        # "#________#",
        # "#_######_#",
        # "#_#______#",
        # "#_#____#_#",
        # "#_#_####_#",
        # "#_#____#_#",
        # "#_###__#_#",
        # "#___#__#_#",
        # "#__##__#_#",
        # "##########",
    ]
    point_lang = point.RealPoint()
    coords = (np.random.uniform(size=(popn_size, 2)) * spread) + 1
    x_init = [point_lang.make_point(a, b) for a, b in coords]
    for fit_policy in fit_policies:
        for accept_policy in accept_policies:
            title = f"fit={fit_policy},accept={accept_policy}"
            local_dir = f"{save_dir}/{title}"
            util.mkdir(local_dir)

            maze = point.RealMaze(str_mask, std=1)
            generator = mcmc_lang_rr(
                lang=maze,
                x_init=x_init,
                popn_size=popn_size,
                n_epochs=n_epochs,
                fit_policy=fit_policy,
                accept_policy=accept_policy,
                distance_metric="euclidean",
                archive_beta=archive_beta,
                archive_size=archive_size,
            )
            data = list(tqdm(generator, total=n_epochs))

            # plots
            data = [analyzer_iter(d, threshold=1e-10) for d in data]
            plot_keys = list(sorted(data[0].keys() - {"x", "x'", "t", "x_feat", "x'_feat", "archive"}))
            fig = util.plot_v_subplots(data, keys=plot_keys)
            fig.savefig(f"{local_dir}/plot.png")
            plt.cla()

            # animation
            init_feat = maze.extract_features(x_init)
            embeddings = [(0, init_feat)]
            embeddings += [(d["t"] + 1, d["x_feat"]) for d in data]
            anim = util.animate_points(
                embeddings,
                title=title,
                xlim=maze.xlim,
                ylim=maze.ylim,
                background=maze.background,
            )
            anim.save(f"{local_dir}/embed.mp4")


def run_lsys_search(config):
    expected_keys = {"x_init", "search", "featurizer", "render"}
    assert all(k in config for k in expected_keys), f"Expected {expected_keys}, got {set(config.keys())}"

    seed = config.search["random_seed"]
    np.random.seed(seed)

    # choose featurizer from config
    feat_kind = config.featurizer["kind"]
    if feat_kind == "ViT":
        featurizer = feat.ViTBase(**config.featurizer)
    elif feat_kind == "resnet":
        featurizer = feat.ResnetFeaturizer(**config.featurizer)
    else:
        raise ValueError(f"Unexpected featurizer kind: {feat_kind}")

    lang = lindenmayer.LSys(
        kind="deterministic",
        featurizer=featurizer,
        **config.render,
    )
    lsystems = [lang.parse(lsys) for lsys in config.x_init]

    length_cap = config.search["length_cap"]
    popn_size = config.search["popn_size"]
    keep_original = config.search["keep_original"]
    if keep_original:
        if popn_size < len(lsystems):
            x_init = lsystems[:popn_size]
        elif popn_size > len(lsystems):
            lang.fit(lsystems, alpha=1.0)
            x_init = lsystems + lang.samples(popn_size - len(lsystems), length_cap=length_cap)
        else:
            x_init = lsystems
    else:
        lang.fit(lsystems, alpha=1.0)
        x_init = lang.samples(popn_size, length_cap=length_cap)

    # init generator
    n_epochs = config.search["n_epochs"]
    generator = mcmc_lang_rr(
        lang=lang,
        x_init=x_init,
        popn_size=popn_size,
        n_epochs=n_epochs,
        fit_policy=config.search["fit_policy"],
        accept_policy=config.search["accept_policy"],
        distance_metric=config.search["distance_metric"],
        archive_size=config.search["archive_size"],
        archive_beta=config.search["archive_beta"],
        length_cap=config.search["length_cap"],
    )

    # make run directory
    save_dir = f"../out/dpp/{wandb.run.id}/"
    try:
        util.mkdir(save_dir)
    except FileExistsError:
        pass

    util.mkdir(f"{save_dir}/data/")
    util.mkdir(f"{save_dir}/images/")

    wandb_process_data_epochs(
        lang=lang,
        generator=generator,
        popn_size=popn_size,
        n_epochs=n_epochs,
        save=True,
        save_dir=save_dir
    )


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

    prev_feat = None
    for i, d in enumerate(tqdm(generator, total=n_epochs, desc="Generating data")):
        if save:
            np.save(f"{save_dir}/data/part-{i:06d}.npy", d, allow_pickle=True)
        analysis_data = analyzer_iter(d, threshold=1e-10)
        coords = reduce_dim(d["x_feat"], srp)
        coord_image = util.scatterplot_image(coords, figsize=3)
        if prev_feat is None:
            prev_feat = d["x_feat"]
        bw = log_best_and_worst(5, lang, d["x"], d["x_feat"], prev_feat)
        log = {
            **d,
            **analysis_data,
            **bw,
            "step": i,
            "renders": render_program_batch_as_wandb_image(lang, d["x"]),
            "archive": render_program_batch_as_wandb_image(lang, d["archive"]),
            "scatter": wandb.Image(coord_image),
        }
        rm_keys = {"x", "x'"}
        log = {k: v for k, v in log.items()
               if k not in rm_keys and not k.endswith("_feat")}
        wandb.log(log)

        prev_feat = d["x_feat"]


def check_fuzzballs(filename: str):
    """See what fuzzballs classify as"""

    from skimage import filters

    classifier = feat.ResnetFeaturizer(disable_last_layer=False, softmax_outputs=False, sigma=5.)
    lang = lindenmayer.LSys(kind="deterministic", featurizer=classifier, step_length=4, render_depth=3)
    sigma = 5.
    n_progs = 4
    with open(filename, "r") as f:
        for i in range(n_progs):
            program = f.readline().strip()
            tree = lang.parse(program)
            images = [
                lang.eval(tree, env={"vary_color": True}),
                lang.eval(tree, env={"vary_color": False}),
            ]

            for image in images:
                image = filters.gaussian(image, sigma=sigma, channel_axis=-1)

                # featurize and classify
                feat_vec = classifier.apply([image])
                class_id = classifier.top_k_classes(feat_vec, k=1)[0]

                plt.imshow(image)
                plt.title(f"{program}\n{class_id}")
                plt.show()


def sweep():
    sweep_config = "./configs/mcmc-lsystem.yaml"
    with open(sweep_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    wandb.init(project="dpp", config=config)
    config = wandb.config
    run_lsys_search(config)


def local_searches():
    ts = util.timestamp()
    save_dir = f"out/dpp-points/{ts}"
    util.try_mkdir(save_dir)
    run_maze_search(
        popn_size=100,
        save_dir=save_dir,
        n_epochs=100,
        fit_policies=["single", "all", "none", "first"],
        accept_policies=["energy", "moment", "all"],
        archive_beta=1,
        archive_size=100,
    )
    # run_point_search(
    #     popn_size=100,
    #     save_dir=save_dir,
    #     n_epochs=100,
    #     fit_policies=["single", "all", "none", "first"],
    #     accept_policies=["energy", "moment", "all"],
    # )


if __name__ == "__main__":
    sweep()
    # local_searches()

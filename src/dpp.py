import pdb
import os
import sys
from typing import List, Iterator, Tuple, Iterable, Optional, Set, Union, Generator, Any, Callable
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
from lang import lindenmayer, point, arc, ant, maze
import util


class VectorArchive:
    def __init__(self, n: int, d: int, debug=False):
        assert n > 0
        assert d > 0

        self.capacity = n
        self.dim = d
        self._vecs = np.zeros((n, d))
        self._tags = np.empty(n, dtype=object)
        self.n_entries = 0
        self.step = 0
        self.debug = debug
        self.log = {
            "added": 0,
            "replaced": 0,
        }

    @staticmethod
    def from_vecs(n: int, vecs: np.ndarray, tags: List[Any], debug=False) -> "VectorArchive":
        assert vecs.ndim == 2, f"Expected 2D vector, but got {vecs.shape}"
        m, d = vecs.shape
        assert m == len(tags)

        archive = VectorArchive(n, d, debug=debug)
        for vec, tag in zip(vecs[:, None], tags):
            archive.add(vec, tag)
        return archive

    def add(self, vec: np.ndarray, tag: Any):
        """Extend archive by a single entry. Returns True if the vec was added to the archive."""
        assert vec.ndim == 2, f"Expected 2D vector, but got {vec.shape}"
        assert vec.shape[0] == 1
        assert vec.shape[1] == self.dim

        if self.debug:
            self.log["added"] += 1

        if self.n_entries < self.capacity:
            # append
            self._vecs[self.n_entries] = vec
            self._tags[self.n_entries] = tag
            self.n_entries += 1
        else:
            # probabilistically replace
            m = np.random.randint(0, self.step)
            replace = m < self.capacity
            if replace:
                self._vecs[m] = vec
                self._tags[m] = tag

            if self.debug and replace:
                self.log["replaced"] += 1

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
        debug=False
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
    x_out, x_feat = lang.evaluate_features(x)
    archive = VectorArchive.from_vecs(
        n=archive_size,
        vecs=x_feat,
        tags=[lang.to_str(tree) for tree in x],
        debug=debug,
    )

    if fit_policy == "first":
        lang.fit(x_init, alpha=1.0)

    for t in range(n_epochs):
        samples = []
        samples_feat = []
        samples_out = []

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
            s_out, s_feat = lang.evaluate_features([s])
            (s_out,) = s_out
            (s_feat,) = s_feat

            up_feat = x_feat.copy()
            up_feat[i] = s_feat

            # save samples
            samples.append(s)
            samples_feat.append(s_feat)
            samples_out.append(s_out)

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
                x_out[i] = s_out

            # track log probabilities
            sum_log_f += log_f
            sum_log_q += log_q
            sum_log_accept += log_accept

        yield {
            "t": t,
            "x": [lang.to_str(p) for p in x],
            "x'": [lang.to_str(p) for p in samples],
            "x_out": x_out.copy(),
            "x'_out": samples_out.copy(),
            "x_feat": x_feat.copy(),
            "x'_feat": samples_feat.copy(),
            "archive": archive.metadata().copy(),
            "log f(x')/f(x)": sum_log_f / popn_size,
            "log q(x|x')/q(x'|x)": sum_log_q / popn_size,
            "log A(x',x)": sum_log_accept / popn_size,
        }

    print(archive.log, file=sys.stderr)


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
            plot_keys = sorted(data[0].keys() - {"x", "x'", "t", "x_feat", "x'_feat", "archive"})
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
        archive_beta: List[float],
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
    ]
    point_lang = point.RealPoint()
    coords = (np.random.uniform(size=(popn_size, 2)) * spread) + 1
    x_init = [point_lang.make_point(a, b) for a, b in coords]
    for fit_policy in fit_policies:
        for accept_policy in accept_policies:
            for beta in archive_beta:
                title = (f"fit={fit_policy},"
                         f"accept={accept_policy},"
                         f"beta={beta}")
                local_dir = f"{save_dir}/{title}"
                util.mkdir(local_dir)

                maze_lang = point.RealMaze(str_mask, std=1)
                generator = mcmc_lang_rr(
                    lang=maze_lang,
                    x_init=x_init,
                    popn_size=popn_size,
                    n_epochs=n_epochs,
                    fit_policy=fit_policy,
                    accept_policy=accept_policy,
                    distance_metric="euclidean",
                    archive_beta=beta,
                    archive_size=archive_size,
                    debug=True,
                )
                data = list(tqdm(generator, total=n_epochs))

                # plots
                data = [analyzer_iter(d, threshold=1e-10) for d in data]
                plot_keys = sorted({
                    k for k in data[0].keys()
                    if (k not in {"t", "x", "x'", "archive"} and
                        not k.endswith("_feat") and
                        not k.endswith("_out"))
                })
                fig = util.plot_v_subplots(data, keys=plot_keys)
                fig.savefig(f"{local_dir}/plot.png")
                plt.cla()

                # animate embeddings
                init_feat = maze_lang.extract_features(x_init)
                frames = [(0, init_feat, [0] * len(init_feat))]
                for d in data:
                    t = d["t"] + 1
                    popn_embeddings = d["x_feat"]

                    if archive_beta == 0:
                        frames.append((
                            t,
                            popn_embeddings,
                            [0] * len(popn_embeddings),
                        ))
                    else:
                        archive_embeddings = [
                            maze_lang.eval(maze_lang.parse(s))
                            for s in d["archive"]
                        ]
                        frames.append((
                            t,
                            np.concatenate([popn_embeddings, archive_embeddings], axis=0),
                            [0] * len(popn_embeddings) + [1] * len(archive_embeddings),
                        ))

                anim = util.animate_points(
                    frames,
                    title=title,
                    xlim=maze_lang.xlim,
                    ylim=maze_lang.ylim,
                    background=maze_lang.background,
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


def run_ant_search_from_conf(conf):
    expected_keys = {
        "maze_name",
        "featurizer",
        "random_seed",
        "popn_size",
        "n_epochs",
        "sim_steps",
        "fit_policy",
        "accept_policy",
        "distance_metric",
        "archive_beta",
        "archive_size",
        "program_depth",
        "length_cap",
        "include_orientation",
        "freeze_leaves",
    }
    assert all(k in conf for k in expected_keys), \
        f"Missing expected keys {expected_keys - set(conf.keys())}"

    run_ant_search(
        **{k: conf[k] for k in expected_keys},
        run_id=wandb.run.id,
        wandb_run=True,
    )


def run_ant_search(
        maze_name: str,
        featurizer: str,
        random_seed: int,
        popn_size: int,
        n_epochs: int,
        sim_steps: int,
        fit_policy: str,
        accept_policy: str,
        distance_metric: str,
        archive_beta: float,
        archive_size: int,
        program_depth: int,
        length_cap: int,
        include_orientation: bool,
        freeze_leaves: bool,
        run_id: str,
        wandb_run=True,
):
    np.random.seed(random_seed)

    maze_map = maze.Maze.from_saved(maze_name) # "lehman-ecj-11-hard"

    if featurizer == "trail":
        ft = ant.TrailFeaturizer(stride=1)
    elif featurizer == "end":
        ft = ant.EndFeaturizer()
    elif featurizer == "heatmap":
        ft = ant.HeatMapFeaturizer(maze)
    else:
        raise ValueError(f"Invalid featurizer type for ant domain: {featurizer}")

    lang = ant.FixedDepthAnt(
        maze=maze_map,
        include_orientation=include_orientation,
        freeze_leaves=freeze_leaves,
        low_state_dim=111,
        program_depth=program_depth,
        steps=sim_steps,
        featurizer=ft,
    )

    assert fit_policy != "all" or popn_size >= lang.n_params, \
        f"Must have popn size > param dimension if fitting to all, " \
        f"but got popn={popn_size}, dim={lang.n_params}"

    # make starting set of programs
    x_init_params = np.random.multivariate_normal(
        np.zeros(lang.n_params),
        np.eye(lang.n_params),
        size=popn_size,
    )
    x_init = [lang.make_program(p) for p in x_init_params]

    generator = mcmc_lang_rr(
        lang=lang,
        x_init=x_init,
        popn_size=popn_size,
        n_epochs=n_epochs,
        fit_policy=fit_policy,
        accept_policy=accept_policy,
        distance_metric=distance_metric,
        archive_size=archive_size,
        archive_beta=archive_beta,
        length_cap=length_cap,
    )

    # make run directory
    save_dir = f"../out/dpp/ant/{run_id}/"
    try:
        util.mkdir(save_dir)
    except FileExistsError:
        pass
    util.mkdir(f"{save_dir}/data/")

    # process data
    for i, d in enumerate(tqdm(generator, total=n_epochs, desc="Generating data")):
        np.save(f"{save_dir}/data/part-{i:06d}.npy", d, allow_pickle=True)
        analysis_data = analyzer_iter(d, threshold=1e-10)

        trails = np.array(d["x_out"])  # [n t 2]
        endpoints = trails[:, -1, :]  # [n 2]

        trail_fig = maze_map.plot_trails(trails)
        endpoint_fig = maze_map.plot_endpoints(endpoints)

        trail_plot = wandb.Image(trail_fig)
        endpoint_plot = wandb.Image(endpoint_fig)

        log = {
            **d,
            **analysis_data,
            "trail": trail_plot,
            "endpoints": endpoint_plot,
            "step": i,
        }
        log = {k: v for k, v in log.items()
               if (k not in {"x", "x'"} and
                   not k.endswith("_feat") and
                   not k.endswith("_out"))}
        if wandb_run:
            wandb.log(log)


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


def sweep(conf: str, run_fn: Callable):
    with open(conf, "r") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    wandb.init(project="dpp", config=config)
    run_fn(wandb.config)


def local_searches():
    ts = util.timestamp()
    save_dir = f"out/dpp-points/{ts}"
    util.try_mkdir(save_dir)
    run_maze_search(
        popn_size=50,
        save_dir=save_dir,
        n_epochs=100,
        fit_policies=["single", "all", "first"],
        accept_policies=["energy", "moment", "all"],
        archive_beta=[0, 1, 5],
        archive_size=10,
    )
    # run_point_search(
    #     popn_size=100,
    #     save_dir=save_dir,
    #     n_epochs=100,
    #     fit_policies=["single", "all", "none", "first"],
    #     accept_policies=["energy", "moment", "all"],
    # )


if __name__ == "__main__":
    sweep("./configs/mcmc-ant.yaml", run_ant_search_from_conf)

    # local_searches()

    # ts = util.timestamp()
    # run_ant_search(
    #     featurizer="End",
    #     random_seed=0,
    #     popn_size=20,
    #     n_epochs=2,
    #     sim_steps=2,
    #     fit_policy="single",
    #     accept_policy="energy",
    #     distance_metric="euclidean",
    #     archive_beta=0.,
    #     archive_size=10,
    #     program_depth=2,
    #     length_cap=1000,
    #     run_id=f"test-{ts}",
    #     wandb_run=False,
    # )

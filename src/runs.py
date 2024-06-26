import csv
from pprint import pp
import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import List, Dict, Union

from lang.tree import Tree
from lang.lindenmayer import LSys
from featurizers import ResnetFeaturizer
import util


def make_df(sweep_path: str, filter_same=False):
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    runs = sweep.runs
    records = []
    for run in runs:
        record = {
            "name": run.name,
            "id": run.id,
        }
        record.update(util.flatten({
            k: v for k, v in run.config.items()
            if not k.startswith('_') and not k.startswith("parameters")
        }))
        records.append(record)

    if filter_same:
        keep_keys = set()
        for k, v in records[0].items():
            if any(record[k] != v for record in records[1:]):
                keep_keys.add(k)
        records = [
            {k: record[k] for k in keep_keys}
            for record in records
        ]

    return pd.DataFrame(records)


def render_run(prefix: str, run_id: str, stride: int):
    df = pd.read_csv(f"{prefix}/{run_id}.csv")
    l = LSys(step_length=4, render_depth=3, n_rows=128, n_cols=128, kind="deterministic",
             featurizer=ResnetFeaturizer())
    # plot rows in batches by generation
    # step program kind dist length score chosen
    steps = df.step.unique()
    print(run_id)
    for step in steps[::stride]:
        print(f"  step: {step}")
        gen = df.loc[(df.step == step) & (df.chosen == True)].sort_values(by='score', ascending=False)[:100]
        imgs = [l.eval(l.parse(x.program)) for i, x in gen.iterrows()]
        # labels = [f"{x.score:.3e}{'*' if x.chosen else ''}" for i, x in gen.iterrows()]
        if len(gen) < 100:
            shape = None
        else:
            shape = (10, 10)
        util.plot_image_grid(imgs,
                             shape=shape,
                             title=f"{run_id} step={step}",
                             fontsize=3,
                             saveto=f"{prefix}/{run_id}-step{step}.png")


def viz_closest(lang: LSys, prefix: str, run_id: str, holdout: List[str], stride: int, n_neighbors=5):
    def embed(s: Union[List[Tree], np.ndarray]):
        return lang.extract_features(s, n_samples=1, load_bar=True)

    df = pd.read_csv(f"{prefix}/{run_id}.csv")
    n_holdout = len(holdout)
    holdout_trees = [lang.parse(x) for x in holdout]
    holdout_embeddings = embed(holdout_trees)

    steps = df.step.unique()
    print(run_id)
    for step in steps[::stride]:
        print(f"  step: {step}")
        gen = df.loc[(df.step <= step) & (df.chosen == True)].program
        gen = np.array([lang.parse(x) for x in gen], dtype=object)
        gen_embeddings = embed(gen)

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="minkowski")
        knn.fit(gen_embeddings)
        _, indices = knn.kneighbors(holdout_embeddings)

        # produce img matrix
        images = []
        for i in range(n_holdout):
            target = holdout_trees[i]
            neighbors = gen[indices[i]].tolist()
            images.extend([lang.eval(x) for x in [target] + neighbors])
        util.plot_image_grid(images, shape=(n_holdout, 1 + n_neighbors),
                             saveto=f"{prefix}/{run_id}-knn-step{step}.png")


def plot_avg_dist(lang: LSys, prefix: str, run_id: str, holdout: List[str], stride: int, n_neighbors=5):
    def embed(s: Union[List[Tree], np.ndarray]):
        return lang.extract_features(s, n_samples=1, load_bar=True)

    df = pd.read_csv(f"{prefix}/{run_id}.csv")
    holdout_trees = [lang.parse(x) for x in holdout]
    holdout_embeddings = embed(holdout_trees)
    rows = []

    steps = df.step.unique()
    print(run_id)
    include_last = (len(holdout) - 1) % stride == 0
    for step in list(steps[::stride]) + (steps[-1:] if include_last else []):
        print(f"  step: {step}")
        gen = df.loc[(df.step <= step) & (df.chosen == True)].program
        gen = [lang.parse(x) for x in gen]
        gen_embeddings = embed(gen)

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="minkowski")
        knn.fit(gen_embeddings)
        dists, _ = knn.kneighbors(holdout_embeddings)
        sum_dists = np.sum(dists, axis=1)
        for i, d in enumerate(sum_dists):
            rows.append([step, i, d])

    # produce line plot
    plot_df = pd.DataFrame(rows, columns=["step", "source", "dist"])
    sns.relplot(plot_df, kind="line", x="step", y="dist", hue="source")
    plt.show()


def sum_configs(configs: List[Dict]):
    """Flatten config dicts and count up each occurrence of each value"""
    def keep(key, val):
        return not key.startswith("parameters.") and \
            not key.endswith(".desc") and \
            ("ablate_mutator" in key or
             "length_penalty_type" in key or
             "softmax_outputs" in key)
    hist = {}
    for config in configs:
        for k, v in util.flatten(config).items():
            if keep(k, v):
                key = f"{k}={v}"
                hist[key] = hist.get(key, 0) + 1
    return sorted({k: v for k, v in hist.items() if 1 < v}.items())


def mock_run_csv(n_steps: int, angle: int, filename: str, max_angle=359):
    """Generate mock run data to test analyses"""
    with open(f"{filename}.csv", "w") as f:
        # write column headers
        writer = csv.writer(f)
        writer.writerow(["step", "program", "kind", "dist", "length", "score", "chosen"])

        # write rows
        for step in range(n_steps):
            for n_toks in range(1, max_angle//angle):
                program = f"{angle};F;F~{'+' * n_toks}F" + ("F" * step)
                writer.writerow([step, program, "S", 0, len(program), 0, True])


def test_evals():
    holdout = ["45;F;F~+F" + ("F" * step) for step in range(10)]
    ## std
    # mock_run_csv(n_steps=10, angle=15, filename="../out/sweeps/mock/mock")
    # render_run("../out/sweeps/mock", "mock", stride=1)
    # viz_closest("../out/sweeps/mock", "mock", holdout=holdout, stride=1, n_neighbors=6)
    # plot_avg_dist("../out/sweeps/mock", "mock", holdout=holdout, stride=1, n_neighbors=1)
    ## small
    # mock_run_csv(n_steps=10, angle=15, max_angle=45, filename="../out/sweeps/mock/small")
    # viz_closest("../out/sweeps/mock", "small", holdout=holdout, stride=2, n_neighbors=2)
    plot_avg_dist("../out/sweeps/mock", "small", holdout=holdout, stride=2, n_neighbors=1)


def get_holdout() -> List[str]:
    with open("configs/static-config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config['holdout_data']


if __name__ == '__main__':
    name = "2a5p4beb"
    df = make_df(f"djsl/novelty/{name}", filter_same=True)
    df.to_csv(f"../out/summary/{name}.csv")
    lang = LSys(kind="deterministic",
                featurizer=ResnetFeaturizer(
                    disable_last_layer=True,
                    softmax_outputs=False,
                    sigma=0),
                step_length=4,
                render_depth=3,
                n_rows=256,
                n_cols=256,
                aa=True,
                vary_color=True)
    # df = pd.read_csv(f"{name}.csv")
    path = f"../out/sweeps/{name}/"
    run_id = "jboe4u9c"
    viz_closest(lang, path, run_id, holdout=get_holdout(), stride=50, n_neighbors=10)

    # for run_id in df.id.unique():
        # render_run(path, run_id, stride=10)
        # viz_closest(lang, path, run_id, holdout=get_holdout(), stride=50, n_neighbors=10)
        # plot_avg_dist(lang, path, run_id, holdout=get_holdout(), stride=50, n_neighbors=5)
    # test_evals()

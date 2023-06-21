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

from lang import Tree
from featurizers import ResnetFeaturizer
from lindenmayer import LSys
from ns import extract_features
import util


def make_df(sweep_path: str):
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
        util.plot(imgs,
                  shape=shape,
                  title=f"{run_id} step={step}",
                  fontsize=3,
                  saveto=f"{prefix}/{run_id}-step{step}.png")

# todo:
#  - produce multiple distances by filtering points into different subsets:
#    - archive/samples/chosen/not chosen
#  - measure redundancy: number of repeated feature vectors
#    (should show a difference between additive/inverse length penalty)
#  - check relative novelty between runs
#  - viz how token probabilities change over time - max likelihood program?
def viz_closest(prefix: str, run_id: str, holdout: List[str], stride: int, n_neighbors=5):
    lang = LSys(kind="deterministic", featurizer=ResnetFeaturizer(), step_length=4, render_depth=3)
    def embed(s: Union[List[Tree], np.ndarray]): return extract_features(lang, s, n_samples=1)

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
        util.plot(images, shape=(n_holdout, 1 + n_neighbors),
                  saveto=f"{prefix}/{run_id}-knn-step{step}.png")


def plot_avg_dist(prefix: str, run_id: str, holdout: List[str], stride: int, n_neighbors=5):
    lang = LSys(kind="deterministic", featurizer=ResnetFeaturizer(), step_length=4, render_depth=3)
    def embed(s: Union[List[Tree], np.ndarray]): return extract_features(lang, s, n_samples=1)

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


if __name__ == '__main__':
    name = "2a5p4beb"
    # df = make_df(f"djsl/novelty/{name}")
    # df.to_csv(f"{name}.csv")
    with open("configs/static-config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    holdout_data = config['holdout_data']
    df = pd.read_csv(f"{name}.csv")
    for run_id in df.id.unique():
        # render_run("../out/sweeps/2a5p4beb/", run_id, stride=10)
        # viz_closest("../out/sweeps/2a5p4beb", run_id, holdout_data, stride=10, n_neighbors=10)
        plot_avg_dist(f"../out/ns/{name}/", run_id, stride=50, n_neighbors=5, holdout=holdout_data)
    # test_evals()

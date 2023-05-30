from pprint import pp
import numpy as np
import pandas as pd
import wandb
from typing import List, Dict

from featurizers import ResnetFeaturizer
from lindenmayer import LSys
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


def render_run(prefix:str, run_id: str):
    df = pd.read_csv(f"{prefix}/{run_id}.csv")
    l = LSys(step_length=4, render_depth=3, n_rows=128, n_cols=128, kind="deterministic",
             featurizer=ResnetFeaturizer())
    # plot rows in batches by generation
    # step program kind dist length score chosen
    steps = df.step.unique()
    print(run_id)
    for step in steps[::10]:
        print(f"  step: {step}")
        gen = df.loc[(df.step == step) & (df.chosen == True)].sort_values(by='score', ascending=False)[:100]
        imgs = [l.eval(l.parse(x.program)) for i, x in gen.iterrows()]
        # labels = [f"{x.score:.3e}{'*' if x.chosen else ''}" for i, x in gen.iterrows()]
        util.plot(imgs, shape=(10, 10),
                  title=f"{run_id} step={step}",
                  fontsize=3,
                  saveto=f"{prefix}/{run_id}-step{step}.png")


def eval_run(run_id: str):
    df = pd.read_csv(f"../sweeps/{run_id}.csv")
    l = LSys(step_length=4, render_depth=3, n_rows=128, n_cols=128, kind="deterministic")
    # todo:
    #  - measure knn distance (in feature space) from programs in df to target programs (book ex's)
    #  - produce multiple distances by filtering points into different subsets:
    #    - by generation
    #    - archive/samples/chosen/not chosen
    #    -

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


if __name__ == '__main__':
    name = "2a5p4beb"
    # df = make_df(f"djsl/novelty/{name}")
    # df.to_csv(f"{name}.csv")
    df = pd.read_csv(f"{name}.csv")
    for run_id in df.id.unique():
        render_run("../out/sweeps/2a5p4beb/", run_id)

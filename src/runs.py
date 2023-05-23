import pdb

import pandas
import pandas as pd

import util
import wandb
from typing import List

from lindenmayer import LSys
from util import flatten


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
        record.update(flatten({
            k: v for k, v in run.config.items()
            if not k.startswith('_') and not k.startswith("parameters")
        }))
        records.append(record)
    return pd.DataFrame(records)


def render_run(run_id: str):
    df = pd.read_csv(f"../sweeps/{run_id}/{run_id}.csv")
    l = LSys(theta=45, step_length=4, render_depth=3, n_rows=128, n_cols=128, kind="deterministic")
    # plot rows in batches by generation
    # step program kind dist length score chosen
    steps = df.step.unique()
    print(run_id)
    for step in steps[::10]:
        print(f"  step: {step}")
        gen = df.loc[df.step == step][:100]
        gen.sort_values(by='score', ascending=False, inplace=True)
        imgs = [l.eval(l.parse(x.program)) for i, x in gen.iterrows()]
        labels = [f"{x.score:.3e}{'*' if x.chosen else ''}" for i, x in gen.iterrows()]
        util.plot(imgs, shape=(10, 10),
                  # labels=labels,
                  title=f"{run_id} step={step}",
                  fontsize=3,
                  saveto=f"../sweeps/{run_id}/step{step}.png")


def render_strs(lsys: LSys, strs: List[str]):
    for s in strs:
        t = lsys.parse(s)
        img = lsys.eval(t)
        yield img


if __name__ == '__main__':
    df = make_df("djsl/novelty/tvwm5xkd")
    df.to_csv("project.csv")
    df = pandas.read_csv("project.csv")
    for run_id in df.id.unique()[160:]:
        render_run(run_id)

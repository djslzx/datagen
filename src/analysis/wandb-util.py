import pandas as pd
import wandb
from pprint import pp


def get_runs(project_name: str) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(project_name)
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append(run.config)
        name_list.append(run.name)

    return pd.DataFrame({
        "name": name_list,
        "summary": summary_list,
        "config": config_list,
    })


def get_sft_runs() -> pd.DataFrame:
    df = get_runs("djsl/sft")
    keys = [
        "model-name",
        "dataset",
        "kbit",
        "max-seq-length",
        "epochs",
        "output_dir",
    ]
    for key in keys:
        df[key] = df["config"].apply(lambda x: x.get(key, None))
        # remove rows that are missing the key
        df = df[df[key].notnull()]

    return df


if __name__ == "__main__":
    df = get_runs("djsl/sft")
    pp(df["config"].head(1).values[0])
    pp(df["summary"].head(1).values[0])

    df = get_sft_runs()
    df.to_csv("sft-runs-filtered.csv")
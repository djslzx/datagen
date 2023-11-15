import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def sample_problems_by_gen(files: Dict[str, str], n_samples: int) -> pd.DataFrame:
    rows = []
    for name, file in files.items():
        df = pd.read_json(file, lines=True)
        df = df[["id", "iter", "name", "text"]]
        df["id"] = df["id"].apply(lambda x: f"{name}:{x}")
        df = df[df["iter"] > 0]

        sample = df.groupby("iter").sample(n=n_samples, replace=False)
        rows.append(sample.to_records(index=False))
    return pd.DataFrame(np.concatenate(rows))


def annotate_samples(samples: pd.DataFrame, annot_files: Dict[str, str]) -> pd.DataFrame:
    def replace_prefix(s: str, prefix: str) -> str:
        suffix = ":".join(s.split(":")[1:])
        return f"{prefix}:{suffix}"

    annots = []
    for name, file in annot_files.items():
        df = pd.read_json(file, lines=True)

        # replace id prefix
        df["id"] = df["id"].apply(lambda x: replace_prefix(x, name))

        # keep only original and restyled problem annotations
        df = df[df["key"].isin(["restyled problem"])]

        # remove duplicate rows
        df = df.drop_duplicates(subset=["id", "key"], keep="first")

        # reshape key, value cols => each key is a column
        df = df[["id", "key", "value"]]
        df = df.pivot(index="id", columns="key", values="value")
        df["id"] = df.index
        annots.append(df.to_records(index=False))

    # merge in annotations
    annot_df = pd.DataFrame(np.concatenate(annots))
    samples = samples.merge(annot_df, how="left", on="id")

    return samples


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 20)

    problems = sample_problems_by_gen(files={
        "NSCA": "../../datasets/wiz/novel-instruct.jsonl",
        "NSE": "../../datasets/wiz/novel-instruct-euler.jsonl",
        "WD": "../../datasets/wiz/wiz-deep.jsonl",
        "WW": "../../datasets/wiz/wiz-wide.jsonl",
    }, n_samples=10)
    problems = annotate_samples(problems, annot_files={
        "NSCA": "../../datasets/wiz/all-solved-NS.jsonl",
        "NSE": "../../datasets/wiz/all-solved-NSE.jsonl",
        "WD": "../../datasets/wiz/all-solved-WD.jsonl",
        "WW": "../../datasets/wiz/all-solved-WW.jsonl",
    })
    print(problems)
    problems.to_csv("../../datasets/wiz/sampled-problems.csv", index=False)

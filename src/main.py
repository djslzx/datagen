import pdb
import sys
import numpy as np
from typing import *
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def summary_stats(arr: np.ndarray) -> str:
    return f"in: [mean: {np.mean(arr)}, std: {np.std(arr)}, max: {np.max(arr)}, min: {np.min(arr)}]"


def make_df(names: List[str], in_files: List[str], out_files: List[str]) -> pd.DataFrame:
    rows = []
    for name, in_file, out_file in zip(names, in_files, out_files):
        for in_line, out_line in zip(*read_files(in_file, out_file)):
            x, y = len(in_line), len(out_line)
            entry = (in_line, out_line, x, y, x-y, name)
            rows.append(entry)
    df = pd.DataFrame(rows, columns=["in", "out", "in length", "out length", "diff", "source"])
    return df


def pluck_egg_examples(df: pd.DataFrame, k=10):
    # remove lines that were empty after simplification (i.e. invalid lines) or which weren't simplified at all
    def view(df: pd.DataFrame) -> str:
        s = ""
        for _, row in df.iterrows():
            s += f"{row['in']} =>\n  {row['out']}, {row['diff']} tokens reduced\n"
        return s

    for source in df["source"].unique():
        fdf = df.loc[(df["diff"] > 0) &
                     (df["out length"] > 0) &
                     (df["source"] == source)].sort_values(by="diff", ascending=False)
        print(f"Filtered {source}: {(df['source'] == source).size} => {fdf.size}")
        m = len(fdf)//2
        print("Max:\n", view(fdf.head(k)))
        print("Med:\n", view(fdf[m:m + k]))
        print("Min:\n", view(fdf.tail(k)))
        print()


def plot_egg_scatter(df: pd.DataFrame) -> plt.Axes:
    counts = df.groupby(["in length", "out length", "source"]).size().reset_index(name="count")
    p = sns.relplot(data=counts, x="in length", y="out length",
                    hue="source", col="source", size="count", sizes=(10, 1000))
    x_max = df["in length"].max()
    p.map(lambda *args, **kwargs: plt.plot([0, x_max], [0, x_max], color="gray"))
    return p


def plot_egg_cdf(df: pd.DataFrame) -> plt.Axes:
    return sns.displot(df, x="diff", hue="source", kind="ecdf")


def plot_egg_pdf(df: pd.DataFrame) -> plt.Axes:
    return sns.displot(df, x="diff", hue="source", kind="kde", bw_adjust=.5)
    # return sns.displot(df, x="diff", hue="source", multiple="dodge", stat="density", common_norm=False)


def read_files(in_file: str, out_file: str) -> Tuple[List[str], List[str]]:
    x, y = [], []
    with open(in_file, 'r') as f_in, open(out_file, 'r') as f_out:
        for f, arr in [(f_in, x), (f_out, y)]:
            for line in f.readlines():
                if line.strip().startswith("#"):
                    continue
                if ":" in line:
                    arr.append(line.split(" : ")[0])
                else:
                    arr.append(line.strip())

    return x, y


if __name__ == "__main__":
    sns.set_theme()
    test_paths = [
        "ns-simpl", "../datasets/ns/ns.txt", "../datasets/ns/ns-simpl.txt",
        "ns-filt", "../datasets/ns/ns.txt", "../datasets/ns/ns-filtered.txt",
        "random", "../datasets/random/random_100k.txt", "../datasets/random/random_100k_simpl.txt",
    ]

    df = make_df(test_paths[::3], test_paths[1::3], test_paths[2::3])
    filtered_df = df.loc[(df["diff"] > 0) & (df["out length"] > 0)]
    print(df)
    print(filtered_df)

    # plot_egg_scatter(filtered_df)
    # plot_egg_cdf(filtered_df)
    # plot_egg_pdf(filtered_df)
    # plt.show()

    pluck_egg_examples(df, k=3)

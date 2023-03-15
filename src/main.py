import numpy as np
from typing import List, Tuple, Dict, Set
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import lark
import pickle

from lang import Language, ParseError


def summary_stats(arr: np.ndarray) -> str:
    return f"in: [mean: {np.mean(arr)}, std: {np.std(arr)}, max: {np.max(arr)}, min: {np.min(arr)}]"


def make_df(names: List[str], in_files: List[str], out_files: List[str]) -> pd.DataFrame:
    rows = []
    for (name, version), in_file, out_file in zip(names, in_files, out_files):
        for in_line, out_line in zip(*read_files(in_file, out_file)):
            x, y = len(in_line), len(out_line)
            entry = (in_line, out_line, x, y, x-y, f"{name}-{version}", name, version)
            rows.append(entry)
    df = pd.DataFrame(rows, columns=["in", "out", "in length", "out length", "diff", "id", "dataset", "version"])
    return df


def pluck_egg_examples(df: pd.DataFrame, k=10):
    # remove lines that were empty after simplification (i.e. invalid lines) or which weren't simplified at all
    def view(df: pd.DataFrame) -> str:
        s = ""
        for _, row in df.iterrows():
            s += f"{row['in']} =>\n  {row['out']}, {row['diff']} tokens reduced\n"
        return s

    for id in df["id"].unique():
        fdf = df.loc[(df["diff"] > 0) &
                     (df["out length"] > 0) &
                     (df["id"] == id)].sort_values(by="diff", ascending=False)
        print(f"Filtered {id}: {(df['id'] == id).size} => {fdf.size}")
        m = len(fdf)//2
        print("Max:\n", view(fdf.head(k)))
        print("Med:\n", view(fdf[m:m + k]))
        print("Min:\n", view(fdf.tail(k)))
        print()


def plot_egg_scatter(df: pd.DataFrame) -> plt.Axes:
    counts = df.groupby(["in length", "out length", "dataset", "version"]).size().reset_index(name="count")
    p = sns.relplot(data=counts, x="in length", y="out length",
                    col="dataset", row="version", size="count", sizes=(10, 1000))
    x_max = df["in length"].max()
    p.map(lambda *args, **kwargs: plt.plot([0, x_max], [0, x_max], color="gray"))
    return p


def plot_egg_cdf(df: pd.DataFrame) -> plt.Axes:
    return sns.displot(df, x="diff", col="dataset", hue="version", kind="ecdf")


def plot_egg_pdf(df: pd.DataFrame) -> plt.Axes:
    return sns.displot(df, x="diff", col="dataset", hue="version", kind="kde", bw_adjust=.5)


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


def simplify_file(lang: Language, in_path: str, out_path: str, score_thresh=None):
    print(f"Writing simplified file to {out_path}")
    n_parse_failures, n_low_score = 0, 0
    with open(in_path, 'r') as f_in, open(out_path, 'w') as f_out:
        for i, line in enumerate(f_in.readlines()):
            if line.startswith("#"):  # skip comments
                f_out.write(line)
                continue
            if ":" in line:  # split out scores
                line, score = line.split(" : ")
                if score_thresh is not None:
                    # skip lines with low score
                    score = float(score.replace("*", ""))
                    if score <= score_thresh:
                        print(f"Skipping line {i} because of low score: {score}")
                        f_out.write("\n")
                        n_low_score += 1
                        continue
            # simplify line
            try:
                t = lang.simplify(lang.parse(line))
                s = lang.to_str(t)
                print(f"{i}: {s}")
                f_out.write(s + "\n")
            except (lark.UnexpectedCharacters, lark.UnexpectedToken, ParseError):
                print(f"Skipping line {i}")
                f_out.write("\n")
                n_parse_failures += 1
    print(f"Skipped {n_parse_failures} lines b/c of parsing failure,\n"
          f"        {n_low_score} lines b/c of low score (< 0.001)")


def plot():
    sns.set_theme()
    test_paths = [
        ("ns-simpl", "1"), "../datasets/ns/ns.txt", "../datasets/ns/ns-simpl.txt",
        ("ns-filt", "1"), "../datasets/ns/ns.txt", "../datasets/ns/ns-filt.txt",
        ("ns-simpl", "2"), "../datasets/ns/ns.txt", "../datasets/ns/ns-simpl2.txt",
        ("ns-filt", "2"), "../datasets/ns/ns.txt", "../datasets/ns/ns-filt2.txt",
        ("ns-simpl", "3"), "../datasets/ns/ns.txt", "../datasets/ns/ns-simpl3.txt",
        ("ns-filt", "3"), "../datasets/ns/ns.txt", "../datasets/ns/ns-filt3.txt",
        ("random", "1"), "../datasets/random/random_100k.txt", "../datasets/random/random_100k_simpl.txt",
        ("random", "2"), "../datasets/random/random_100k.txt", "../datasets/random/random_100k_simpl2.txt",
        ("random", "3"), "../datasets/random/random_100k.txt", "../datasets/random/random_100k_simpl3.txt",
    ]
    df = make_df(test_paths[::3], test_paths[1::3], test_paths[2::3])
    # filtered_df = df.loc[(df["diff"] > 0) & (df["out length"] > 0)]
    print(df)
    # print(filtered_df)

    # plot_egg_scatter(df)
    # plot_egg_cdf(df)
    # plot_egg_pdf(df)
    # plt.show()
    pluck_egg_examples(df, k=3)


def read_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # simplify_file("../datasets/ns/ns.txt", "../datasets/ns/ns-simpl3.txt")
    # simplify_file("../datasets/ns/ns.txt", "../datasets/ns/ns-filt3.txt", 0.001)
    # simplify_file("../datasets/random/random_100k.txt", "../datasets/random/random_100k_simpl3.txt")
    # plot()
    x = read_pickle("../datasets/csv/csv.p")
    print(x)

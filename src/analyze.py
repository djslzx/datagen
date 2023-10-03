import json
from pprint import pp
from tqdm import tqdm
import itertools as it
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Dict, Tuple, Callable
import textwrap
import datetime
from langchain.chat_models import ChatOpenAI

import featurizers as feat
import util
import wizard


def load_json_as_df(filename: str) -> pd.DataFrame:
    data = util.load_jsonl(filename)
    data = pd.json_normalize(data)
    return pd.DataFrame(data)


def add_parents(df: pd.DataFrame) -> pd.DataFrame:
    df["parent"].fillna(df["id"], inplace=True)
    df["parent"] = df["parent"].astype(np.int32)
    if "source file" in df.columns:
        df["parent name"] = df.apply(lambda row: df.loc[df["id"] == row["parent"]]["name"], axis=1)
        df["parent text"] = df.apply(lambda row: df.loc[df["id"] == row["parent"]]["text"], axis=1)
    else:
        df["parent name"] = df.apply(
            lambda row: df.loc[(df["id"] == row["parent"]) & (df["source file"] == row["source file"])]["name"], axis=1)
        df["parent text"] = df.apply(
            lambda row: df.loc[(df["id"] == row["parent"]) & (df["source file"] == row["source file"])]["text"], axis=1)
    df["mutator"].fillna("self", inplace=True)
    return df


def add_pc_dist(df: pd.DataFrame) -> pd.DataFrame:
    """analyze semantic distances between parents and children"""
    assert type(df["embedding"].iloc[0]) == np.ndarray
    if "source file" in df.columns:
        df["parent embedding"] = df.apply(
            lambda row: df.loc[(df["id"] == row["parent"]) & (df["source file"] == row["source file"])]["embedding"],
            axis=1)
    else:
        df["parent embedding"] = df.apply(lambda row: df.loc[df["id"] == row["parent"]]["embedding"], axis=1)
    df["pc dist"] = df.apply(lambda row: np.linalg.norm(row["embedding"] - row["parent embedding"]), axis=1)
    return df


def avg_distance(df: pd.DataFrame, n_samples: int):
    """
    Plot average embedding distance by generation;
    use a random sample for each generation to reduce computation time.
    """
    # select a row with `n_samples` entries for each generation
    sample = df.groupby("iter").sample(n_samples)

    # compute average embedding distance within each group
    sample["avg dist"] = sample.groupby("iter").apply(
        lambda x: np.mean([np.linalg.norm(x.iloc[i]["embedding"] -
                                          x.iloc[j]["embedding"])
                           for i in range(n_samples)
                           for j in range(i + 1, n_samples)])
    )
    sample["avg pc dist"] = sample.groupby("iter")["pc dist"].mean()
    sample = sample.melt(var_name='cols', value_name='value', value_vars=["avg dist", "avg pc dist"])

    sns.lineplot(data=sample, x="iter", y="value", hue="cols")
    plt.title(f"Average embedding distance by generation ({n_samples} samples)")
    plt.gcf().set_size_inches(12, 6)
    plt.show()


def pc_dist_samples(df: pd.DataFrame):
    # sample best parent-child pairs
    samples: pd.DataFrame = (
        df
        .loc[df["iter"] % 20 == 0]
        .sort_values("pc dist", ascending=False)
        .groupby(["iter", "source file"])
        .head(3)
        .sort_values(["source file", "iter"])
        [["source file", "iter", "text", "pc dist", "mutator"]]
    )
    for i, row in samples.iterrows():
        print(
            f"{row['iter']} ({row['source file']}) w/ dist {row['pc dist']}:\n"
            f"#+begin_quote"
            f"{row['text']}"
            f"#+end_quote"
        )
    samples = (
        df
        .loc[df["source file"] == "evol-instruct-20Kx3"]
        .sort_values("pc dist", ascending=False)
        .groupby("iter")
        .head(3)
        .sort_values("iter")
        [["iter", "text", "pc dist", "mutator"]]
    )
    print("Samples from evol-instruct-20Kx3")
    for i, row in samples.iterrows():
        print(
            f"{row['iter']} w/ dist {row['pc dist']}:\n"
            f"#+begin_quote"
            f"{row['text']}"
            f"#+end_quote"
        )


def pc_dist_plots(df: pd.DataFrame, names: List[str]):
    # avg pc dist by gen
    sns.relplot(data=df, x="iter", y="pc dist", hue="source file", kind="line")
    plt.gcf().set_size_inches(12, 6)
    plt.show()

    for name in names:
        sns.relplot(data=df[df["source file"] == name],
                    x="iter", y="pc dist", hue="mutator", col="source file", kind="line")
        plt.gcf().set_size_inches(12, 6)
        plt.show()

    # plot pc dist by mutator
    sns.catplot(data=df, x="mutator", y="pc dist", kind='violin', row="source file")
    plt.gcf().set_size_inches(12, 6)
    plt.show()


def read_runs_into_df(filenames: Dict[str, str], with_embeddings=True) -> pd.DataFrame:
    full_df: Optional[pd.DataFrame] = None
    for shortname, filename in filenames.items():
        df = pd.read_json(f"{filename}.jsonl", lines=True)
        df = add_parents(df)

        if with_embeddings:
            # embeddings = wizard.embed(feat.SentenceFeaturizer(), [data["text"] for data in data], saveto=f"{file}.npy")
            embeddings = np.load(f"{filename}.npy")
            print(f"Loaded file {filename} with {len(embeddings)} embeddings")
            df["embedding"] = pd.Series([v for v in embeddings], dtype=object)
            df = add_pc_dist(df)

        df["source file"] = shortname
        full_df = df if full_df is None else pd.concat([full_df, df], ignore_index=True)
    return full_df


def chamfer_diversity(a_embeddings: np.ndarray, b_embeddings: np.ndarray, k=1) -> float:
    knn = NearestNeighbors(metric="minkowski", n_neighbors=k)
    knn.fit(b_embeddings)
    d = 0
    for a in a_embeddings:
        dists, _ = knn.kneighbors([a])
        d += dists.mean()
    return d


def avg_density(embeddings: np.ndarray, n_samples: int, n_neighbors=5) -> float:
    I = np.random.randint(low=0, high=len(embeddings), size=n_samples)
    embeddings = embeddings[I]
    knn = NearestNeighbors(metric="minkowski", n_neighbors=n_neighbors + 1)
    knn.fit(embeddings)
    d = 0
    for x in embeddings:
        dists, _ = knn.kneighbors([x])
        dists = dists[0]
        assert dists[0] < 1e-5
        d += dists[1:].mean()
    return d / n_neighbors


def plot_avg_density(data: dict, n_samples: int):
    rows = []
    for key in data.keys():
        e_k = data[key]["embeddings"]
        for k in range(1, 11):
            rows.append({"dataset": key,
                         "k": k,
                         "density": avg_density(e_k, n_samples=n_samples, n_neighbors=k)})
    df = pd.DataFrame(rows)
    sns.lineplot(df, x="k", y="density", hue="dataset")
    plt.gcf().set_size_inches(12, 6)
    plt.show()


def plot_chamfer_diversity_heatmap(data: dict, n_samples: int):
    rows = []
    for k1, k2 in it.combinations(data.keys(), r=2):
        print(f"{k1}, {k2} w/ {n_samples} samples")
        l1 = data[k1]["lines"]
        l2 = data[k2]["lines"]
        I1 = [i for i, line in enumerate(l1) if line["iter"] > 0]
        I2 = [i for i, line in enumerate(l2) if line["iter"] > 0]
        e1 = data[k1]["embeddings"][I1]
        e2 = data[k2]["embeddings"][I2]
        print(f"Number of embeddings: {k1}: {len(I1)}, {k2}: {len(I2)}")
        e1 = e1[np.random.randint(low=0, high=len(e1), size=n_samples)]
        e2 = e2[np.random.randint(low=0, high=len(e2), size=n_samples)]
        d12 = chamfer_diversity(e1, e2, k=1)
        d21 = chamfer_diversity(e2, e1, k=1)
        print(f"D({k1}, {k2}) = {d12}")
        print(f"D({k2}, {k1}) = {d21}")
        rows.extend([
            {"src": k1, "dst": k2, "dist": d12 - d21},
            {"src": k2, "dst": k1, "dist": d21 - d12},
        ])
    for k in data.keys():
        rows.append({"src": k, "dst": k, "dist": 0})
    df = pd.DataFrame(rows)
    df = df.pivot(columns="dst", index="src")
    sns.heatmap(df, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
    plt.gcf().set_size_inches(12, 6)
    plt.show()


def det_diversity(embeddings: np.ndarray, n_samples: int) -> float:
    """Use determinant (volume of parallelepiped) to measure diversity"""
    pass


def plot_embedding_stats(data: dict):
    def l2_norm(mat: np.ndarray):
        return np.sum(np.abs(mat) ** 2, axis=1) ** 0.5

    rows = []
    for k in data:
        e_k = data[k]["embeddings"]
        max_sum = np.linalg.norm(e_k, ord=np.inf)
        min_sum = np.linalg.norm(e_k, ord=-np.inf)
        max_val = np.linalg.norm(e_k, ord=2)
        min_val = np.linalg.norm(e_k, ord=-2)
        max_l2 = np.max(l2_norm(e_k))
        min_l2 = np.min(l2_norm(e_k))
        rows.append({
            "dataset": k,
            "max sum": max_sum,
            "min sum": min_sum,
            "max val": max_val,
            "min val": min_val,
            "max l2": max_l2,
            "min l2": min_l2,
        })
    df = pd.melt(pd.DataFrame(rows),
                 id_vars=["dataset"],
                 value_vars=["max sum", "min sum",
                             "max val", "min val",
                             "max l2", "min l2"])
    sns.catplot(df, x="variable", y="value", hue="dataset", kind="bar")
    plt.show()


def load_embeddings(filename: str, expected_length=None) -> np.ndarray:
    try:
        embeddings = np.load(f"{filename}.npy")
        if expected_length:
            if expected_length > len(embeddings):
                print(f"Expected {expected_length} embeddings, got {len(embeddings)}. Generating more embeddings...")
                lines = util.load_jsonl(f"{filename}.jsonl")
                return wizard.embed(feat.SentenceFeaturizer(),
                                    [line["text"] for line in lines],
                                    saveto=f"{filename}.npy")
            else:
                return embeddings[:expected_length]
        else:
            return embeddings
    except FileNotFoundError:
        print(f"Cached embeddings for {filename} not found, generating embeddings...")
        lines = util.load_jsonl(f"{filename}.jsonl")
        return wizard.embed(feat.SentenceFeaturizer(),
                            [line["text"] for line in lines],
                            saveto=f"{filename}.npy")


def add_extras(df: pd.DataFrame, extras: List[Tuple[str, Callable]], saveto: str) -> pd.DataFrame:
    jsonl_fname = saveto + ".jsonl"
    with wizard.get_openai_callback() as cb, open(jsonl_fname, "w") as f:
        for i, row in df.iterrows():
            for extra, fn in extras:
                row[extra] = fn(row)
            line = json.dumps(row.to_dict(), indent=None)
            print(f"Adding line: {line}")
            f.write(line + "\n")
            if i % 100 == 0:
                print(f"Cost: {cb.total_cost}")
    df = pd.read_json(jsonl_fname, lines=True)
    df.to_csv(f"{saveto}.csv")
    return df


def add_predicate_cols(chat: ChatOpenAI, df: pd.DataFrame, saveto: str) -> pd.DataFrame:
    return add_extras(
        df=df,
        extras=[
            ("solvable?", lambda row: wizard.check_problem_solvable(chat, row['text'])),
            ("novel?", lambda row: wizard.check_problem_novel(chat, row['text'], row['parent text'])),
        ],
        saveto=saveto,
    )


def add_solution_cols(chat: ChatOpenAI, df: pd.DataFrame, saveto: str, n=1):
    return add_extras(
        df=df,
        extras=[(f"solution-{i}", lambda row: wizard.propose_solution(chat, row["text"]))
                for i in range(n)],
        saveto=saveto,
    )


def add_entry_point_col(chat: ChatOpenAI, df: pd.DataFrame, saveto: str):
    assert "solution" in df.columns, f"Missing 'solution' column in columns={df.columns}"
    assert "text" in df.columns, f"Missing 'text' column in columns={df.columns}"
    return add_extras(
        df=df,
        extras=[("entry_point", lambda row: wizard.propose_entry_point(chat, row["text"], row["solution"]))],
        saveto=saveto,
    )


def analyze_datasets(filenames: Dict[str, str]):
    data = {
        shortname: {
            "lines": util.load_jsonl(f"{filename}.jsonl"),
            "embeddings": load_embeddings(f"{filename}"),
        }
        for shortname, filename in filenames.items()
    }
    print("Loaded data:")
    for name in data.keys():
        n_embeddings = len(data[name]['embeddings'])
        print(f"  {name}: {n_embeddings} embeddings")

    plot_avg_density(data, n_samples=1000)
    plot_chamfer_diversity_heatmap(data, n_samples=1000)
    plot_embedding_stats(data)

    # todo: child-child dist - how similar are children of a given parent to one another over time?

    df = read_runs_into_df(filenames, with_embeddings=False)[:10]

    # add solvable column
    chat = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")
    df = add_extras(chat, df, "../datasets/all-extras.jsonl")

    # pc_dist_plots(df, names=list(filenames.keys()))
    # pc_dist_samples(df)


def load_annotations(annot_file: str, filename_map: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_json(annot_file, lines=True)
    # df = df[df["id"] < 100]  # use a subset of the dataset for testing
    print(df["source file"].value_counts())

    for shortname, filename in filename_map.items():
        n_entries = len(df[df["source file"] == shortname])
        if n_entries == 0:
            continue

        # add ids if not present
        if "id" not in df.columns:
            df.loc[df["source file"] == shortname, "id"] = np.arange(n_entries)

        # check that ids were added properly:
        # - if mutator is self, id should match parent
        # - if mutator is not self, id should be different from parent

    # df["rank"] = df["rank"].astype(int)
    df["solvable?"] = df["solvable?"] == "True"
    df["novel?"] = df["novel?"] == "True"
    # df["non-identical?"] = df["pc dist"] > 1e-5
    df["chosen?"] = df["chosen?"] == 1.0
    df["both?"] = df["novel?"] & df["solvable?"]
    return df


def sample_problems(df: pd.DataFrame, n: int):
    rows = []
    headings = ["id", "iter", "source file", "mutator", "parent name", "name", "text",
                "score", "rank", "chosen?", "solvable?", "novel?"]
    for source in df["source file"].unique():
        print(f"Source: {source}")
        sample = (df[(df["chosen?"] | (source.lower().startswith("wiz")))
                     & (df["source file"] == source)
                     # & (df["solvable?"])
                     & (df["novel?"])]
                  .sample(n=n)
                  .sort_values(["iter", "source file"]))
        rows.extend(sample[headings].to_records(index=False))
    return pd.DataFrame.from_records(rows, columns=headings)


def analyze_annotations(df: pd.DataFrame):
    # # add embeddings
    # min_id = df["id"].min()  # fixme: assumes all ids start at the same number
    # df["embedding"] = df.apply(lambda row: embed_map[row["source file"]][row["id"] - min_id], axis=1)
    #
    # # check that every entry has an embedding
    # assert df["embedding"].isna().sum() == 0, f"Missing embeddings for {df[df['embedding'].isna()]['source file']}"
    # df = add_pc_dist(df)

    for y in ["solvable?", "novel?", "both?"]:
        sns.relplot(df, x="iter", y=y, hue="source file")
        plt.show()

    table = pd.melt(df,
                    id_vars=["iter", "source file"],
                    value_vars=["solvable?", "novel?", "both?"])
    print(table)

    sns.relplot(table, x="iter", y="value", hue="source file", style="variable", kind="line")
    plt.show()

    # condition on whether entries were chosen
    for y in ["solvable?", "novel?", "both?"]:
        sns.relplot(df[df["chosen?"]], x="iter", y=y, hue="source file", kind="line")
        plt.show()

    # summary heatmap
    grp = (df
           [df["iter"] > 0]
           [["source file", "solvable?", "novel?", "both?"]]
           .groupby(["source file"], group_keys=False)
           .mean())
    grp.reset_index(inplace=True)
    table = pd.melt(grp, id_vars=["source file"], value_vars=["solvable?", "novel?", "both?"])
    print(table)
    table = table.pivot(columns="variable", index="source file")
    print(table)
    sns.heatmap(table, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
    plt.show()

    def sample_table(mask):
        entries = (df[mask]
                   [["source file", "iter", "mutator", "text", "parent text", "solvable?", "novel?"]]
                   .sample(5))
        entries["text"] = entries["text"].apply(lambda x: '\n'.join(textwrap.wrap(x, width=40)))
        entries["parent text"] = entries["parent text"].apply(lambda x: '\n'.join(textwrap.wrap(x, width=40)))
        return entries.to_markdown(index=False)

    def sample_text(mask):
        entries = (df[mask]
                   [["source file", "iter", "mutator", "text", "parent text", "solvable?", "novel?"]]
                   .sample(5))
        return entries["text"].to_list()

    # display a sampling of prompts that are solvable/unsolvable, novel/unoriginal
    for text in sample_text(df["solvable?"] & df["novel?"]):
        print(text)

    # print(
    #     "Solvable:", sample_table(df["solvable?"]),
    #     "Unsolvable:", sample_table(~df["solvable?"]),
    #     "Novel:", sample_table(df["novel?"]),
    #     "Unoriginal:", sample_table(~df["novel?"]),
    #     sep="\n"
    # )


def analyze_extras(df: pd.DataFrame):
    """
    - are the solutions all the same?
    - are the tests all the same?
    - how similar are the tests in test(text) to those in test(text, soln)?
    - how many tests are runnable? how do we check this?
      - failure modes:
        - needs web connection
        - needs functions to be defined
        - needs external resources (run code in a container? but then too slow...)
        - code is malformed
        - etc
    - how many solutions pass the tests?
    """
    # how many solutions are the same?
    n_solns = max(int(c[len("solution-"):])
                  for c in df.columns if c.startswith("solution-"))
    print(f"n solns: {n_solns}")
    df["n unique solns"] = df.apply(
        lambda row: len({
            row[f"solution-{i}"] for i in range(n_solns)
        }),
        axis=1
    )
    avg_n_uniq_solns = df["n unique solns"].mean()
    print(f"Average number of unique solutions per row: {avg_n_uniq_solns}")

    # how similar (SBert distance) are the solutions?
    pass

    # how similar are the tests?
    pass


def run_extras(annot_file: str, filenames: dict, n_samples: int, n_solns: int) -> pd.DataFrame:
    df = load_annotations(annot_file, filenames)
    lo_temp_chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
    hi_temp_chat = ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo-16k-0613")
    df = sample_problems(df, n=n_samples)
    df = add_solution_cols(hi_temp_chat, df, n=n_solns, saveto=f"../datasets/sampling-solved-{timestamp}")
    # df = add_entry_point_col(chat, df, f"../datasets/sampling-epoint-{timestamp}")
    df = add_extras(
        df,
        saveto=f"../datasets/sampling-tests-{timestamp}",
        extras=[
            ("test(text)", lambda row: wizard.propose_test_from_text(lo_temp_chat, row["text"])),
            ("test(text, soln)",
             lambda row: wizard.propose_test_from_text_and_solution(lo_temp_chat, row["text"], row["solution-0"])),
        ],
    )
    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.min_rows", 50)
    pd.set_option('display.max_colwidth', 40)
    timestamp = util.timestamp()

    filenames = {
        "NS": "../datasets/novel-instruct-200x80-2023-09-01T15:50:12.708593",
        # "NS-euler": "../datasets/novel-instruct-euler-2023-09-07T13:34:54.519254",
        # "Wiz-wide": "../datasets/evol-instruct-20kx3-2023-08-29T18:39:47.555169",
        # "Wiz-deep": "../datasets/evol-instruct-1000x100-2023-08-25T12:36:17.752098",
        # "CA 1K": "../datasets/code_alpaca_1k",
        # "CA 20K": "../datasets/code_alpaca_20k",
    }

    # df = load_annotations("../datasets/annotated-sep20.jsonl", filenames)
    # analyze_annotations(df)
    # df = run_extras(annot_file="../datasets/annotated-sep20.jsonl", filenames=filenames, n_samples=10, n_solns=2)
    # df = pd.read_json("../datasets/sampling-tests-2023-10-02T22:52:58.129383.jsonl", lines=True)
    df = pd.read_json("../datasets/sampling-tests-2023-10-03T00:51:02.288607.jsonl", lines=True)
    analyze_extras(df)

    # # find outputs with "sorry"
    # sorry = df[df["sample.output.text"].str.lower().str.contains(["sorry", "apolog", "can't", "unable", "unfortunately"])]
    # print(sorry)

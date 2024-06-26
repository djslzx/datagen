import re
import json
from pprint import pp
import itertools as it
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Dict, Tuple, Callable, Set
import textwrap
from langchain.chat_models import ChatOpenAI

import sys
import featurizers as feat
import util
import wizard
import prompts


TIMESTAMP = util.timestamp()


def load_json_as_df(filename: str) -> pd.DataFrame:
    data = util.load_jsonl(filename)
    data = pd.json_normalize(data)
    return pd.DataFrame(data)


def add_parents(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("id", inplace=True, drop=False)
    df["parent"].fillna(df["id"], inplace=True)
    df["parent"] = df["parent"].astype(np.int32)
    df["parent name"] = df.apply(lambda row: df.loc[row["parent"]]["name"], axis=1)
    df["parent text"] = df.apply(lambda row: df.loc[row["parent"]]["text"], axis=1)
    df["mutator"].fillna("self", inplace=True)
    return df


def add_pc_dist(df: pd.DataFrame) -> pd.DataFrame:
    """analyze semantic distances between parents and children"""
    assert type(df["embedding"].iloc[0]) == np.ndarray
    df.set_index("id", inplace=True, drop=False)
    df["parent embedding"] = df.apply(lambda row: df.loc[row["parent"]]["embedding"], axis=1)
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


def read_runs_into_df(filenames: Dict[str, str]) -> pd.DataFrame:
    full_df: Optional[pd.DataFrame] = None
    for shortname, filename in filenames.items():
        df = pd.read_json(f"{filename}.jsonl", lines=True)
        df = add_parents(df)
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

    df = read_runs_into_df(filenames)[:10]

    # add solvable column
    chat = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")
    df = add_extras(chat, df, "../datasets/all-extras.jsonl")

    # pc_dist_plots(df, names=list(filenames.keys()))
    # pc_dist_samples(df)


def sample_problems(df: pd.DataFrame, n: int):
    rows = []
    for source in df["source file"].unique():
        print(f"Source: {source}")
        sample = (
            df[df["source file"] == source]
            .sample(n=n)
            # .sort_values(["iter", "source file"])
        )
        rows.extend(sample.to_records(index=False))
    return pd.DataFrame.from_records(rows, columns=df.columns)


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


def read_problems(filenames: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for shortname, filename in filenames.items():
        df = pd.read_json(f"{filename}.jsonl", lines=True)
        df["source file"] = shortname

        # add id column if it doesn't exist
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)

        df = df[["id", "source file", "text"]]
        rows.extend(df.to_records(index=False))
    return pd.DataFrame.from_records(rows, columns=["id", "source file", "text"])


# Analyze results from running tests on solutions
def analyze_test_results(df: pd.DataFrame):
    print("columns:", df.columns)
    print("source files:", df["source file"].unique())
    print("file sizes:", df["source file"].value_counts())

    # group together errors
    patterns = {
        r"test_.+ did not pass": "test failed",
        r"name '.+' is not defined": "name is not defined",
        r"indentation error (.+)": "indentation error",
        r"invalid syntax (.+)": "invalid syntax",
        r"'return' outside function (.+)": "return outside function",
        r"'.+' object has no attribute '.+'": "object has no attribute",
        r"No module named": "module not found",
        r"\[Errno 2\] No such file or directory: '.+'": "file not found",
        r"module not found '.+'": "module not found",
        r"expected an indented block (.+)": "indentation error",
        r"EOF while scanning triple-quoted string literal (.+)": "EOF while scanning triple-quoted string literal",
        r"import of .+ halted": "import failure",
        r"cannot import name '.+' from .+": "import failure"
    }

    # group together patterns using regex match
    for pattern, label in patterns.items():
        df["result"] = df["result"].str.replace(pattern, label, regex=True)

    def error_map(x: str) -> str:
        if x == "failed: test failed":
            return "test failed"
        elif x == "passed":
            return "passed"
        else:
            return "error"

    df["result type"] = df["result"].apply(error_map)
    df["error"] = df["result type"] == "error"
    df["failed"] = df["result type"] == "test failed"

    # how many unique solns and tests per problem?
    ndf = df.groupby("id").agg({
        "source file": "first",
        "soln": "nunique",
        "test": "nunique",
        "passed": "mean",
        "error": "mean",
        "failed": "mean",
        "rating": "mean",
    })
    ndf.rename(columns={"soln": "n_solns", "test": "n_tests"}, inplace=True)

    # plot histogram of both n solns and n tests
    sns.countplot(ndf, x="n_solns", hue="source file")
    plt.title("Number of unique solutions by dataset")
    plt.show()
    sns.countplot(ndf, x="n_tests", hue="source file")
    plt.title("Number of unique tests by dataset")
    plt.show()

    # plot histogram by difficulty rating
    sumdf = ndf.groupby(["source file", "rating"], as_index=False).agg({"n_solns": "mean", "n_tests": "mean"})
    sns.barplot(sumdf, x="rating", y="n_solns", hue="source file")
    plt.title("Number of solutions by rating")
    plt.show()

    # make a pie chart of result type
    plt.pie(df["result type"].value_counts(), labels=df["result type"].value_counts().index)
    plt.show()

    # make a bar chart of result type by source file
    sns.countplot(data=df, x="source file", hue="result type")
    plt.gcf().set_size_inches(6, 6)
    plt.tight_layout()
    plt.show()

    # plot difficulty rating by source file
    sns.displot(data=df, x="rating", hue="source file", multiple="stack", binwidth=1, discrete=True)
    plt.title("Difficulty rating by source file")
    plt.gcf().set_size_inches(6, 6)
    plt.tight_layout()
    plt.show()

    # for each rating, show a stacked bar plot of the number of problems that failed, passed, or errored
    sns.histplot(df, x="rating", hue="result type", multiple="stack", discrete=True, stat="percent")
    plt.show()

    # line plot of pass/fail/error percentages by difficulty rating
    nndf = ndf.groupby("rating", as_index=False).agg({"passed": "mean", "error": "mean", "failed": "mean"})
    nndf = nndf.melt(id_vars=["rating"], var_name="result type", value_name="percent")
    sns.lineplot(data=nndf, x="rating", y="percent", hue="result type")
    plt.show()

    sns.displot(
        df, x="rating", hue="result type", row="source file",
        multiple="stack", discrete=True, stat="percent",
    )
    plt.show()

    # evaluate solutions: what is the average number of tests passed per solution?
    pass

    # plot distribution of % passed
    pass

    # evaluate tests: what is the average number of solutions that pass each test?
    pass


def gen_solns_and_tests(chat: ChatOpenAI, df: pd.DataFrame, n_samples: Optional[int]) -> pd.DataFrame:
    # sample problems from each source file
    if n_samples is not None:
        df = df.groupby("source file").sample(n=n_samples)

    # generate solutions and tests for sample
    problems = df[["id", "text"]].to_dict(orient="records")
    df = util.incrementally_save_jsonl(
        prompts.gen_solns_and_tests_dict(chat, problems),
        filename=f"../datasets/wiz/solns-and-tests-{n_samples}-{TIMESTAMP}"
    )
    return df


def subset_by_id(all_df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    all_df["id"] = all_df.apply(
        lambda row: f"{row['source file']}:{row['id']}",
        axis=1
    )
    return all_df[all_df["id"].isin(ids)]


def rate_difficulty(chat: ChatOpenAI, df: pd.DataFrame) -> pd.DataFrame:
    assert all(col in df.columns for col in ["source file", "id", "text"]), \
        f"Expected to have columns ['source file', 'id', 'text'], got {df.columns}"

    def gen_ratings(problems):
        for problem in problems:
            rating = prompts.rate_difficulty(chat, problem["text"])
            yield {
                "id": problem["id"],
                "text": problem["text"],
                "rating": rating,
            }

    return util.incrementally_save_jsonl(
        gen_ratings(problems=df[["source file", "id", "text"]].to_dict(orient="records")),
        filename=f"../datasets/wiz/ratings-{TIMESTAMP}"
    )


def try_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return np.nan


if __name__ == "__main__":
    # pd.set_option("display.max_rows", None)
    # pd.set_option("display.min_rows", 50)
    pd.set_option("display.max_columns", None)
    # pd.set_option('display.max_colwidth', 20)

    CHAT = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
    filenames = {
        "NS": "../datasets/wiz/novel-instruct",
        "NS-euler": "../datasets/wiz/novel-instruct-euler",
        "Wiz-wide": "../datasets/wiz/wiz-wide",
        "Wiz-deep": "../datasets/wiz/wiz-deep",
        "CA-20k": "../datasets/wiz/code-alpaca",
    }
    extras = [
        "../datasets/wiz/ratings/ratings-2023-10-31T16:51:38.999519.jsonl",
        "../datasets/wiz/ratings/ratings-2023-10-31T16:51:38.999519.jsonl",
        "../datasets/wiz/solns-and-tests/all-solns-and-tests.jsonl",
    ]


    def convert_ca(s: str) -> str:
        if s.startswith("CA-"):
            _, n = s.split(":")
            return f"CA:{n}"
        else:
            return s


    df = pd.read_json("../datasets/wiz/solns-and-tests/all-solns-and-tests.jsonl", lines=True)
    df["id"] = df["id"].apply(convert_ca)

    ratings = pd.concat([
        pd.read_json(filename, lines=True)
        for filename in [
            "../datasets/wiz/ratings/ratings-2023-10-31T16:51:38.999519.jsonl",
            "../datasets/wiz/ratings/ratings-2023-10-31T16:51:38.999519.jsonl",
        ]
    ], ignore_index=True).drop_duplicates(subset="id")
    ratings = ratings.melt(id_vars=["id"], value_vars=["rating"], var_name="key", value_name="value")
    ratings["id"] = ratings["id"].apply(convert_ca)

    df = pd.concat([df, ratings], axis=0)
    df.sort_values(["id", "key"], inplace=True)
    print(df)

    # df = gen_solns_and_tests(CHAT, master, n_samples=None)
    # print(df)

    analyze_test_results(df)

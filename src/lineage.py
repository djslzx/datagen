import json
from pprint import pp
from tqdm import tqdm
import itertools as it
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Dict
import graph_tool.all as gt
from langchain.chat_models import ChatOpenAI

import featurizers as feat
import util
import wizard


def add_vertex(G: gt.Graph, name: str, text: str, iteration: int, rank: int, chosen: bool, archived: bool):
    v = G.add_vertex()
    G.vp.name[v] = name
    G.vp.text[v] = text
    G.vp.iteration[v] = iteration
    G.vp.rank[v] = rank
    G.vp.chosen[v] = chosen
    G.vp.archived[v] = archived
    return v


def add_edge(G: gt.Graph, source: gt.Vertex, target: gt.Vertex, method: str):
    e = G.add_edge(source=source, target=target)
    G.ep.method[e] = method
    return e


def build_lineage_graph(df: pd.DataFrame, chat: ChatOpenAI = None) -> gt.Graph:
    G = gt.Graph()
    G.vp["name"] = G.new_vp("string")
    G.vp["text"] = G.new_vp("string")
    G.vp["iteration"] = G.new_vp("int")
    G.vp["rank"] = G.new_vp("int")
    G.vp["chosen"] = G.new_vp("bool")
    G.vp["archived"] = G.new_vp("bool")
    G.ep["method"] = G.new_ep("string")

    # add first generation parents first
    text_to_vertex = dict()
    for _, row in df[df["iteration"] == 0].iterrows():
        if row["iteration"] == 0:
            parent_text = row["sample.input"]
            if "sample.input name" in row:
                parent_name = row["sample.input name"]
            elif chat:
                parent_name = wizard.propose_name(chat, parent_text)
            else:
                parent_name = ""

            rank = row["rank"]
            if parent_text in text_to_vertex:
                parent = text_to_vertex[parent_text]
                rank = min(rank, G.vp.rank[parent])
                G.vp.rank[parent] = rank
            else:
                v = add_vertex(G, parent_name, parent_text, 0, rank, True, False)
                text_to_vertex[parent_text] = v

    # add the rest of the nodes
    for _, row in df.iterrows():
        parent_text = row["sample.input"]
        child_text = row["sample.output"]
        child_name = row["sample.output name"] if "sample.output name" in row else ""
        iteration = row["iteration"]
        rank = row["rank"]
        chosen = row["chosen?"]
        archived = row["archived?"]
        method = wizard.EVOL_METHOD_NAMES[row["sample.evol_method"]]

        child = add_vertex(G, child_name, child_text, iteration + 1, rank, chosen, archived)
        text_to_vertex[child_text] = child
        parent = text_to_vertex[parent_text]
        add_edge(G, parent, child, method)

    return G


def genealogy_layout(G: gt.Graph) -> gt.PropertyMap:
    pos = G.new_vp("vector<double>")
    for v in G.vertices():
        pos[v] = np.array([G.vp.iteration[v], G.vp.rank[v]])
    return pos


def embedding_layout(G: gt.Graph, embeddings: np.array) -> gt.PropertyMap:
    mds = MDS(n_components=2, random_state=0)
    positions = mds.fit_transform(embeddings)
    pos = G.new_vp("vector<double>")
    for v in G.vertices():
        if G.vp.name[v] == "root":
            G.vp.pos[v] = np.array([0, 0])
        else:
            i = G.vertex_index[v]
            G.vp.pos[v] = positions[i]
    return pos


def add_root_to_sources_(G: gt.Graph):
    sources = [v for v in G.vertices() if v.in_degree() == 0]
    root = add_vertex(G, "root", "", -1, 0, True, False)
    for v in sources:
        add_edge(G, root, v, "root")
    return root


def draw_lineage_graph(G: gt.Graph, pos: gt.PropertyMap = None, outfile: str = None):
    gt.graph_draw(
        G,
        pos=pos,
        # vertex_text=G.vp.name,
        output=f"../reports/images/{outfile}.png" if outfile else None,
        output_size=(2000, 2000),
        # vertex_text_position="centered"
    )


def load_json_as_df(filename: str) -> pd.DataFrame:
    data = util.load_jsonl(filename)
    data = pd.json_normalize(data)
    return pd.DataFrame(data)


def add_ancestors(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("id", inplace=True)
    df["parent"].fillna(df.index.to_series(), inplace=True)
    df["parent"] = df["parent"].astype(np.int32)
    df["parent name"] = df.apply(lambda row: df.loc[row["parent"]]["name"], axis=1)
    df["parent text"] = df.apply(lambda row: df.loc[row["parent"]]["text"], axis=1)
    df["mutator"].fillna("self", inplace=True)

    # add root column
    roots = []
    for id, row in df.iterrows():
        if row.iter == 0:
            root_id = id
        roots.append(root_id)
    df["root"] = roots
    df["root name"] = df.apply(lambda row: df.loc[row["root"]]["name"], axis=1)
    df["root text"] = df.apply(lambda row: df.loc[row["root"]]["text"], axis=1)
    return df


def add_pc_dist(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """analyze semantic distances between parents and children"""
    df["embedding"] = pd.Series([v for v in embeddings], dtype=object)
    assert type(df["embedding"].iloc[0]) == np.ndarray
    df["parent embedding"] = df.apply(lambda row: df.loc[row["parent"]]["embedding"], axis=1)
    df["root embedding"] = df.apply(lambda row: df.loc[row["root"]]["embedding"], axis=1)
    df["pc dist"] = df.apply(lambda row: np.linalg.norm(row["embedding"] - row["parent embedding"]), axis=1)
    df["rc dist"] = df.apply(lambda row: np.linalg.norm(row["embedding"] - row["root embedding"]), axis=1)
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


def plot_by_generation(df: pd.DataFrame, y: str):
    """plot a metric by generation"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    sns.lineplot(df, x="iteration", y=y, ax=axes[1])
    sns.lineplot(df, x="iteration", y=y, hue="sample.evol_method", ax=axes[0])
    plt.suptitle(f"{y} by generation")
    plt.tight_layout()
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
        data = util.load_jsonl(f"{filename}.jsonl")
        df = pd.DataFrame(data)
        df = add_ancestors(df)

        if with_embeddings:
            # embeddings = wizard.embed(feat.SentenceFeaturizer(), [data["text"] for data in data], saveto=f"{file}.npy")
            embeddings = np.load(f"{filename}.npy")
            print(f"Loaded file {filename} with {len(embeddings)} embeddings")
            df = add_pc_dist(df, embeddings)

        df["id"] = df.index
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


def load_embeddings(filename: str) -> np.ndarray:
    try:
        return np.load(f"{filename}.npy")
    except FileNotFoundError:
        print(f"Cached embeddings for {filename} not found, generating embeddings...")
        lines = util.load_jsonl(f"{filename}.jsonl")
        return wizard.embed(feat.SentenceFeaturizer(),
                            [line["text"] for line in lines],
                            saveto=f"{filename}.npy")


def add_solvable_col(chat: ChatOpenAI, df: pd.DataFrame) -> pd.DataFrame:
    solvable = []
    with wizard.get_openai_callback() as cb:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            solvable.append(wizard.check_problem_solvable(chat, row['text']))
            if i % 100 == 0:
                print(f"Cost: {cb.total_cost}")
    df["solvable?"] = solvable
    return df


def add_novel_col(chat: ChatOpenAI, df: pd.DataFrame) -> pd.DataFrame:
    novel = []
    with wizard.get_openai_callback() as cb:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            novel.append(wizard.check_problem_novel(chat, src_problem=row['text'], dst_problem=row['parent text']))
            if i % 100 == 0:
                print(f"Cost: {cb.total_cost}")
    df["novel?"] = novel
    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.min_rows", 50)

    filenames = {
        "NS": "../datasets/novel-instruct-200x80-2023-09-01T15:50:12.708593",
        # "NS-euler": "../datasets/novel-instruct-pe-2023-09-07T13:34:54.519254",
        # "Wiz-wide": "../datasets/evol-instruct-20kx3-2023-08-29T18:39:47.555169",
        # "Wiz-deep": "../datasets/evol-instruct-1000x100-2023-08-25T12:36:17.752098",
        # "CA 1K": "../datasets/code_alpaca_1k",
        # "CA 20K": "../datasets/code_alpaca_20k",
    }
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

    # plot_avg_density(data, n_samples=1000)
    # plot_chamfer_diversity_heatmap(data, n_samples=1000)
    # plot_embedding_stats(data)

    # todo: child-child dist - how similar are children of a given parent to one another over time?

    df = read_runs_into_df(filenames, with_embeddings=False)

    # add solvable column
    df = df[:10]
    chat = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")
    df = add_solvable_col(chat, df)
    df = add_novel_col(chat, df)
    print(df)
    df.to_csv("../datasets/NI-2023-09-01-extras.csv")
    # df = pd.read_csv("../datasets/NI-2023-09-01-extras.csv")[["iter", "id", "name", "text", "solvable?"]]

    # show percentage of solvable by iter
    df = df[["iter", "solvable?", "novel?"]]
    df["solvable?"] = df["solvable?"] == "True"
    df["novel?"] = df["novel?"] == "True"
    grp = df.groupby("iter").mean()["solvable?"]
    print(grp)
    sns.lineplot(grp)
    plt.show()

    # [[""]]
    # pc_dist_plots(df, names=list(filenames.keys()))
    # pc_dist_samples(df)

    # # find outputs with "sorry"
    # sorry = df[df["sample.output.text"].str.lower().str.contains(["sorry", "apolog", "can't", "unable", "unfortunately"])]
    # print(sorry)

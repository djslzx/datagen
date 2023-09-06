from pprint import pp
from sklearn.manifold import MDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional
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


def embed_dist(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """analyze semantic distances between parents and children"""
    df.set_index("id", inplace=True)
    df["parent"].fillna(df.index.to_series(), inplace=True)
    df["parent"] = df["parent"].astype(np.int32)

    # add root column
    roots = []
    for id, row in df.iterrows():
        if row.iter == 0:
            root_id = id
        roots.append(root_id)
    df["root"] = roots

    df["mutator"].fillna("self", inplace=True)
    df["embedding"] = pd.Series([v for v in embeddings], dtype=object)
    assert type(df["embedding"].iloc[0]) == np.ndarray
    df["parent name"] = df.apply(lambda row: df.loc[row["parent"]]["name"], axis=1)
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


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.min_rows", 50)

    files = {
        "novel-instruct": "../datasets/novel-instruct-200x80-2023-09-01T15:50:12.708593",
        "evol-instruct-20Kx3": "../datasets/evol-instruct-20kx3-2023-08-29T18:39:47.555169",
        "evol-instruct-1000x100": "../datasets/evol-instruct-1000x100-2023-08-25T12:36:17.752098",
    }
    full_df: Optional[pd.DataFrame] = None
    for name, file in files.items():
        data = util.load_jsonl(f"{file}.jsonl")
        # embeddings = wizard.embed(feat.SentenceFeaturizer(), [data["text"] for data in data], saveto=f"{file}.npy")
        embeddings = np.load(f"../datasets/{file}.npy")
        print(f"Loaded file {file} with {len(data)} samples and {len(embeddings)} embeddings")

        df = pd.DataFrame(data)
        df = embed_dist(df, embeddings)
        df["id"] = df.index
        df["source file"] = name
        full_df = df if full_df is None else pd.concat([full_df, df], ignore_index=True)

    # # check how many roots are represented at each iteration
    # roots_by_iter = df.groupby("iter")["root"].nunique()
    # print(roots_by_iter.describe())

    # # avg pc dist by gen
    # sns.relplot(data=full_df, x="iter", y="pc dist", hue="source file", kind="line")
    # plt.gcf().set_size_inches(12, 6)
    # plt.show()
    #
    # for name in files.keys():
    #     sns.relplot(data=full_df[full_df["source file"] == name],
    #                 x="iter", y="pc dist", hue="mutator", col="source file", kind="line")
    #     plt.gcf().set_size_inches(12, 6)
    #     plt.show()
    #
    # # plot pc dist by mutator
    # sns.catplot(data=full_df, x="mutator", y="pc dist", kind='violin', row="source file")
    # plt.gcf().set_size_inches(12, 6)
    # plt.show()

    # sample best parent-child pairs
    samples: pd.DataFrame = (
        full_df
        .loc[full_df["iter"] % 20 == 0]
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
        full_df
        .loc[full_df["source file"] == "evol-instruct-20Kx3"]
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

    # # plot proportion of 0-distance pairs by mutator
    # df["same"] = df["pc dist"] == 0
    # sns.catplot(data=df, x="mutator", y="same", kind='bar')

    # # find outputs with "sorry"
    # sorry = df[df["sample.output.text"].str.lower().str.contains(["sorry", "apolog", "can't", "unable", "unfortunately"])]
    # print(sorry)

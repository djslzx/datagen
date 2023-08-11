from pprint import pp
from sklearn.manifold import MDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
import graph_tool.all as gt
from langchain.chat_models import ChatOpenAI

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
                parent_name = wizard.name_programming_problem(chat, parent_text)
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


def parent_embed_dist(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """analyze semantic distances between parents and children"""
    # prepend rows for starting dataset
    orig_rows = pd.DataFrame([
        {
            "iteration": 0,
            "score": None,
            "rank": None,
            "chosen?": True,
            "archived?": False,
            "sample.input": None,
            "sample.evol_method": None,
            "sample.output": x,
        }
        for x in df[df["iteration"] == 0]["sample.input"].unique()
    ])
    df["iteration"] += 1
    df = pd.concat([orig_rows, df], ignore_index=True)

    # add embedding as column of df
    df["sample.output.embedding"] = pd.Series([v for v in embeddings], dtype=object)
    assert type(df["sample.output.embedding"].iloc[0]) == np.ndarray

    # set a row's sample input index by looking for the first row whose sample output is the same as the current row's sample input
    # unless the row is a starting dataset row, in which case the sample input index is the row's index
    df["sample.input.index"] = df.apply(
        lambda row: row.name if row["iteration"] == 0 else df[df["sample.output"] == row["sample.input"]].index[0],
        axis=1
    )
    # pull the embedding of the sample input from the input's row
    df["sample.input.embedding"] = df.apply(
        lambda row: df["sample.output.embedding"].iloc[row["sample.input.index"]],
        axis=1
    )
    # take distance between input and output embeddings
    df["parent-child dist"] = df.apply(
        lambda row: np.linalg.norm(row["sample.input.embedding"] - row["sample.output.embedding"]),
        axis=1
    )

    df["sample.evol_method"] = df["sample.evol_method"].apply(lambda x: wizard.EVOL_METHOD_NAMES[x] if x else None)

    # drop embedding columns
    df = df.drop(columns=["sample.output.embedding", "sample.input.embedding"])

    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.min_rows", 50)

    # chat = ChatOpenAI(temperature=0.9)
    # df = load_json_as_df("../datasets/code_alpaca_100_nov_100xAll.jsonl")
    embeddings = np.load("../datasets/code_alpaca_tiny_nov_10x20_1-embeddings.npy")
    df = load_json_as_df("../datasets/code_alpaca_tiny_nov_10x20_1.jsonl")
    df = parent_embed_dist(df, embeddings)
    print(df)

    sns.relplot(df, x="iteration", y="parent-child dist", hue="sample.evol_method", kind="line")
    plt.show()


    # G = build_lineage_graph(df, chat=None)
    # # root = add_root_to_sources_(G)
    # # print(G)
    #
    # embeddings = np.load("../datasets/code_alpaca_tiny_nov_10x20_1-embeddings.npy")
    # # embeddings = np.load("../datasets/code_alpaca_100_nov_100xAll-embeddings.npy")
    # print(len(embeddings))

    # vG = gt.GraphView(G,
    #                   vfilt=lambda v: G.vp.iteration[v] < 4,
    #                   efilt=lambda e: G.vp.iteration[e.target()] < 4)
    # draw_lineage_graph(
    #     G,
    #     pos=genealogy_layout(G),
    #     outfile="lineage_graph_big"
    # )
    # draw_lineage_graph(vG, pos=gt.radial_tree_layout(vG, root=root))

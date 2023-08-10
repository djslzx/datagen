import copy
import os
import random
import sys
from datetime import datetime
from math import ceil
from typing import List, Generator, Union
import numpy as np
import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn import preprocessing, random_projection
from sklearn.manifold import MDS
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import palettable
import networkx as nx
import wandb
import langchain
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import get_openai_callback  # track token usage
from langchain.cache import SQLiteCache

import featurizers as feat
import util

# setup langchain cache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# fetch api key from env
API_KEY = os.getenv("OPENAI_API_KEY")

# Evol-Instruct mutation methods
EVOL_METHODS = [
    "Add new constraints and requirements to the original problem, adding approximately 10 additional words.",
    "Replace a commonly used requirement in the programming task with a less common and more specific one.",
    "If the original problem can be solved with only a few logical steps, please add more reasoning steps.",
    "Provide a piece of erroneous code as a reference to increase misdirection.",
    "Propose higher time or space complexity requirements, but please refrain from doing so frequently.",
]
EVOL_METHOD_NAMES = {
    "Add new constraints and requirements to the original problem, adding approximately 10 additional words.": "lengthen",
    "Replace a commonly used requirement in the programming task with a less common and more specific one.": "specific",
    "If the original problem can be solved with only a few logical steps, please add more reasoning steps.": "steps",
    "Provide a piece of erroneous code as a reference to increase misdirection.": "misdirection",
    "Propose higher time or space complexity requirements, but please refrain from doing so frequently.": "complexity",
}


def name_programming_problem(chat: ChatOpenAI, text: str) -> str:
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Come up with a name for the following programming problem.  "
        "The name should contain no more than 5 words.  "
        "Your response should contain no formatting.  "
        "Each word in the name should be separated by a space."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(input=text)


def evol_instruct_step(chat: ChatOpenAI, instruction: str, evol_method: str) -> dict:
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Please increase the difficulty of the given programming test question a bit. "
        "You can increase the difficulty using, but not limited to, the following methods: {evol_method}"
        "Please respond with a new programming test question that can be understood independently "
        "without any reference to the original question. "
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    output = chain.run(evol_method=evol_method, input=instruction)
    name = name_programming_problem(chat, output)
    return {
        "input": instruction,
        "evol_method": evol_method,
        "output": output,
        "output name": name,
    }


# generate a dataset using Evol-Instruct from the WizardLM paper
def evol_instruct(chat: ChatOpenAI,
                  iters: int,
                  seed_dataset: List[str],
                  evol_methods: List[str],
                  log_file: str):
    dataset = seed_dataset
    with get_openai_callback() as cb, open(log_file, "w") as f:
        # To take advantage of caching, we run each example `iters` times.
        # This doesn't affect the result b/c each example is mutated independently of the others.

        for i, x in enumerate(dataset, start=1):
            prompt = x
            completions = []
            for _ in tqdm(range(iters)):
                evol_method = random.choice(evol_methods)
                completion = evol_instruct_step(chat, prompt, evol_method)
                prompt = completion["output"]
                completions.append({"evol_method": completion["evol_method"],
                                    "output": prompt})

            # update log file
            json.dump({"input": x, "completions": completions}, f)
            f.write("\n")

            print(f"Completed evolution of instruction {i} of {len(dataset)}", file=sys.stderr)
            print(cb)


# generate a small dataset using Evol-Instruct + novelty pressure
def novel_instruct(chat: ChatOpenAI,
                   fe: feat.Featurizer,
                   iters: int,
                   seed_dataset: List[str],
                   evol_methods: List[str],
                   max_popn_size: int,
                   archive_per_iter: int) -> Generator[List[dict], None, None]:
    def take_samples(in_data: Union[List[str], np.ndarray]):
        # flat generator over samples from evol-instruct
        for x in tqdm(in_data, total=len(in_data)):
            for evol_method in evol_methods:
                sample = evol_instruct_step(chat, x, evol_method)
                yield sample

    def embed(v: List[str]) -> np.ndarray:
        em = fe.apply(v)
        # embeddings = SparseRandomProjection(n_components="auto").fit_transform(embeddings)
        # scale embeddings
        scaler = preprocessing.StandardScaler()
        em = scaler.fit_transform(em)
        return em

    knn = NearestNeighbors(metric="minkowski")
    popn = seed_dataset
    e_popn = embed(popn)
    archive = []
    e_archive = []

    log_table = wandb.Table(columns=["top-1", "top-2", "top-3", "bot-1", "bot-2", "bot-3"])
    with get_openai_callback() as cb:  # expose cost information
        for i in range(iters):
            # take samples using evol-instruct
            samples = np.array(list(take_samples(popn)), dtype=object)
            s_samples = np.array([d["output"] for d in samples], dtype=object)

            # use novelty pressure to select best samples
            knn.fit(np.concatenate([e_popn, e_archive], axis=0) if archive else e_popn)
            e_samples = embed(s_samples.tolist())
            dists, _ = knn.kneighbors(e_samples)
            dists = np.mean(dists, axis=1)
            ranks = np.argsort(-dists)
            i_selected = ranks[:max_popn_size]

            # update archive: take a random selection of samples and add them to the archive
            i_archived = np.random.choice(len(s_samples), size=archive_per_iter, replace=False)
            archive.extend(s_samples[i_archived])
            e_archive.extend(e_samples[i_archived])

            # update popn, embeddings
            popn = s_samples[i_selected]
            e_popn = e_samples[i_selected]

            # log everything
            print(cb, file=sys.stderr)
            inverted_ranks = util.invert_array(ranks)
            log_table.add_data(*(samples[ranks[:3]].tolist() +
                                 samples[ranks[-3:][::-1]].tolist()))
            wandb.log({
                "scores": wandb.Histogram(dists),
                "examples": copy.copy(log_table),
                "token usage": {
                    "total cost": cb.total_cost,
                    "total tokens": cb.total_tokens,
                    "prompt tokens": cb.prompt_tokens,
                    "completion tokens": cb.completion_tokens,
                },
            })
            yield [{"iteration": i,
                    "sample": sample,
                    "score": float(dists[j]),
                    "rank": int(inverted_ranks[j]),
                    "chosen?": bool(j in i_selected),
                    "archived?": bool(j in i_archived),
                    }
                   for j, sample in enumerate(samples)]


def build_lineage_graph(wizard_edges: List[dict], chat: ChatOpenAI = None, add_origin=False) -> nx.DiGraph:
    G = nx.DiGraph()
    # add first generation parents first
    if add_origin:
        G.add_node("ORIGIN", name="ORIGIN", text="ORIGIN", iteration=-1, rank=0, chosen=False, archived=False)
    for edge in wizard_edges:
        if edge["iteration"] == 0:
            parent_text = edge["sample"]["input"]
            parent_name = edge["sample"].get("input name",
                                             name_programming_problem(chat, parent_text) if chat else "")
            rank = edge["rank"]
            parent = (0, parent_text)
            if parent in G.nodes:
                rank = min(rank, G.nodes[parent]["rank"])
            G.add_node(
                parent,
                name=parent_name,
                text=parent_text,
                iteration=0,
                rank=rank,
                chosen=True,
                archived=False)
            if add_origin:
                G.add_edge("ORIGIN", parent, method="Code alpaca")

    # add the rest of the nodes
    for edge in wizard_edges:
        parent_text = edge["sample"]["input"]
        child_text = edge["sample"]["output"]
        child_name = edge["sample"].get("output name", "")
        i = edge["iteration"]
        rank = edge["rank"]
        chosen = edge["chosen?"]
        archived = edge["archived?"]
        method = EVOL_METHOD_NAMES[edge["sample"]["evol_method"]]

        # represent nodes as tuples so that nodes in different iterations can have the same text
        # (we assume that the text of a node is unique within an iteration)
        parent = (i, parent_text)
        child = (i + 1, child_text)
        G.add_edge(parent, child, method=method)
        G.add_node(
            child,
            name=child_name,
            text=child_text,
            iteration=i + 1,
            rank=rank,
            chosen=chosen,
            archived=archived
        )

    return G


def draw_lineage_graph(G: nx.DiGraph,
                       chosen_only=False,
                       ancestors_only=False,
                       scale_rank=False,
                       position=None,
                       embeddings=None,
                       by_iteration=False,
                       with_labels=True,
                       with_edges=True,
                       save_dir=None,
                       nodesize=100,
                       figsize=(10, 10)):
    timestamp = str(datetime.now().timestamp())

    if position is None:
        position = "iteration"

    if chosen_only:
        G = G.subgraph([n for n, data in G.nodes(data=True) if data["chosen"]])

    if ancestors_only:
        max_iter = max(i for _, i in G.nodes.data(data="iteration"))
        last_gen = [v for v, i in G.nodes.data(data="iteration") if i == max_iter]
        ancestors = []
        for node in last_gen:
            ancestors.append(node)  # include last gen nodes
            ancestors.extend(nx.ancestors(G, node))
        G = G.subgraph(ancestors)

    # colors
    colors = palettable.matplotlib.get_map("Viridis_5").mpl_colors
    base_color = colors[0]
    highlight_color = colors[-1]
    recent_color = colors[2]

    if position == "iteration":
        # extract node positions
        max_rank_by_iter = {}
        for _, d in G.nodes(data=True):
            max_rank_by_iter[d["iteration"]] = max(max_rank_by_iter.get(d["iteration"], 0),
                                                   d["rank"])
        max_rank = max(max_rank_by_iter.values())

        # arrange nodes by iteration/rank; scale rank if desired
        def scaling_factor(i):
            return max_rank / max_rank_by_iter[i] if scale_rank else 1

        pos = {n: (data["iteration"],
                   -data["rank"] * scaling_factor(data["iteration"]))
               for n, data in G.nodes(data=True)}
    elif position == "embedding" or position == "spring-embedding":
        assert embeddings is not None, "embeddings must be provided if position is 'embedding'"
        scaler = preprocessing.StandardScaler()  # scale embeddings
        embeddings = scaler.fit_transform(embeddings)
        mds = MDS(n_components=2, random_state=0)
        projection = mds.fit_transform(embeddings)
        pos = {n: (x, y) for n, (x, y) in zip(G.nodes, projection)}
        if position == "spring-embedding":
            pos = nx.spring_layout(G, pos=pos, iterations=1000)
    elif position == "spring":
        pos = nx.spring_layout(G, iterations=1000)
    elif isinstance(position, tuple) and position[0] == "graphviz":
        _, prog = position
        pos = nx.nx_agraph.graphviz_layout(G, prog=prog)
    else:
        raise ValueError(f"Invalid position option {position}")

    def draw_graph(graph, i):
        def color_node(node_data):
            if node_data["chosen"]:
                return base_color
            elif i and node_data["iteration"] == i:
                return recent_color
            else:
                return "white"

        p = {n: pos[n] for n in graph.nodes}
        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(
            graph,
            pos=p,
            # with_labels=False,
            # width=0.2,
            # edge_color="grey",
            # edge_color=[colors[method_index[method]] for _, _, method in graph.edges.data("method")],
            node_color=[color_node(data) for _, data in graph.nodes(data=True)],
            node_size=nodesize,
            alpha=0.5,
        )
        # draw only edges from most recent iteration
        if with_edges and i:
            nx.draw_networkx_edges(
                graph,
                pos=p,
                edgelist=[(u, v) for u, v, data in graph.edges(data=True)
                          if graph.nodes[v]["iteration"] == i],
                width=0.2,
                edge_color="grey",
            )
        if with_labels:
            nx.draw_networkx_labels(
                graph,
                pos={n: (x + 0.02, y) for n, (x, y) in p.items()},
                labels={n: data["name"] for n, data in graph.nodes(data=True)},
                font_size=6,
                horizontalalignment="left",
                verticalalignment="top",
                clip_on=False,
            )
        # make legend
        legend_handles = [
            Line2D([0], [0], marker='o', markerfacecolor=recent_color, color='w', label='recent'),
            Line2D([0], [0], marker='o', markerfacecolor=base_color, color='w', label='prior'),
        ]
        if with_edges:
            legend_handles += [
                Line2D([0], [0], color=color, lw=4, label=method)
                for method, color in zip(EVOL_METHOD_NAMES.values(), colors)
            ]
        plt.legend(handles=legend_handles, loc="lower left")
        plt.axis("off")
        plt.tight_layout()

        if save_dir is not None:
            # make a subdirectory for this lineage using iso timestamp
            lineage_dir = os.path.join(save_dir, timestamp)
            os.makedirs(lineage_dir, exist_ok=True)
            plt.savefig(os.path.join(lineage_dir, f"lineage_{i:03d}.png"))
        else:
            plt.show()

    if by_iteration:
        def gen_subgraph(graph, iteration):
            return graph.subgraph([n for n, data in graph.nodes(data=True)
                                   if data["iteration"] <= iteration])

        for i in range(max(i for _, i in G.nodes.data(data="iteration")) + 1):
            draw_graph(gen_subgraph(G, i), i)
    else:
        draw_graph(G, None)


# for a given prompt, what is the average rank of `mutate(prompt, method i)`?
def rank_distro(G: nx.DiGraph, rank_cap: int):
    # for each node, track the ranks of its out-neighbors by edge type
    rank_map = {}  # map edge type to list of observed ranks
    for _, v, data in G.edges(data=True):
        method = data["method"]
        rank = G.nodes[v]["rank"]
        if rank < rank_cap:
            rank_map[method] = rank_map.get(method, []) + [rank]

    # build a histogram of ranks for each edge type
    plt.hist(rank_map.values(), bins=20, label=list(rank_map.keys()))
    # for method, ranks in rank_map.items():
    #     plt.hist(ranks, label=method, alpha=0.5)
    plt.legend(loc='lower center')
    plt.show()


def avg_rank_by_iter(G: nx.DiGraph):
    rows = []
    for _, v, data in G.edges(data=True):
        vdata = G.nodes[v]
        method = data["method"]
        it = vdata["iteration"]
        rank = vdata["rank"]
        if it > 1:
            rows.append((method, it, -rank))

    df = pd.DataFrame(rows, columns=["method", "iter", "rank"])
    sns.relplot(
        data=df, x="iter", y="rank", col="method",
        kind="line", errorbar="sd",
    )
    plt.show()


def run_search(iters: int, seed_dataset: str, archive_per_iter: int, max_popn_size: int, output_file: str):
    instructions = [x["instruction"] for x in json.load(open(seed_dataset, "r"))]
    chat = ChatOpenAI(temperature=0.9, client=None)
    wandb.init(project="wizard")
    fe = feat.SentenceFeaturizer()
    with open(output_file, "w") as f:
        for log in novel_instruct(chat, fe,
                                  iters=iters,
                                  seed_dataset=instructions,
                                  evol_methods=EVOL_METHODS,
                                  max_popn_size=max_popn_size,
                                  archive_per_iter=archive_per_iter):
            for entry in log:
                s = json.dumps(entry, indent=None)
                f.write(s + "\n")


def mutate_prompt(chat: ChatOpenAI, text: str, k: int) -> List[str]:
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Produce {k} different ways of stating the following text. "
        "Preserve the fulling meaning of the original text, but use different words. "
        "Do not add or remove information from the text."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(k=k, input=text)


def embed_and_save(G: nx.Graph, filename: str, key="text"):
    # use sentence transformer to generate embeddings
    ft = feat.SentenceFeaturizer()
    nodes = list(G.nodes(data=True))
    embeddings = []
    for batch in tqdm(util.batched(nodes, batch_size=128), total=ceil(len(nodes) / 128)):
        texts = [data[key] for _, data in batch]
        embeddings.extend(ft.apply(texts))
    embeddings = np.array(embeddings)
    np.save(filename, embeddings)
    return embeddings


def name_problems_from_file(in_filename: str, out_filename: str):
    chat = ChatOpenAI(temperature=0.9, client=None)
    in_data = util.load_jsonl(in_filename)
    with get_openai_callback() as cb, open(out_filename, "w") as f:
        for d in in_data:
            named_d = d.copy()
            if d["iteration"] == 0:
                input_name = name_programming_problem(chat, d["sample"]["input"])
                print(f"Named {d['sample']['input']} as {input_name}", file=sys.stderr)
                named_d["sample"]["input name"] = input_name
            output_name = name_programming_problem(chat, d["sample"]["output"])
            print(f"Named {d['sample']['output']} as {output_name}", file=sys.stderr)
            named_d["sample"]["output name"] = output_name
            f.write(json.dumps(named_d) + "\n")
        print(cb, file=sys.stderr)


if __name__ == "__main__":
    # run_search(
    #     iters=20,
    #     archive_per_iter=5,
    #     max_popn_size=10,
    #     seed_dataset="../datasets/code_alpaca_tiny.json",
    #     output_file="../datasets/code_alpaca_tiny_nov_10x20_1.jsonl"
    # )
    dataset = util.load_jsonl("../datasets/code_alpaca_100_nov_100xAll.jsonl")
    # dataset = util.load_jsonl("../datasets/code_alpaca_tiny_nov_10x20_1.jsonl")
    print(len(dataset))
    G = build_lineage_graph(dataset, chat=ChatOpenAI(temperature=0.9, client=None))
    G = G.subgraph([n for n, data in G.nodes(data=True) if 1 < data["iteration"] < 4])
    print(G)

    # save embeddings to file
    # embeddings = embed_and_save(G, key="text", filename="../datasets/code_alpaca_tiny_nov_10x20_1-embeddings.npy")
    embeddings = np.load("../datasets/code_alpaca_100_nov_100xAll-embeddings.npy")

    draw_lineage_graph(
        G,
        by_iteration=True,
        position="embedding",  # ("graphviz", "sfdp"),
        embeddings=embeddings,
        with_labels=False,
        with_edges=False,
        nodesize=5,
        figsize=(12, 12),
        save_dir="../reports/images/lineage_graphs/big/",
    )

    # # ranking of edge types (evol methods)
    # rank_distro(G, rank_cap=10)
    # avg_rank_by_iter(G)

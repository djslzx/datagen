import copy
import os
import random
import sys
from typing import List, Generator, Union
import numpy as np
import json

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from einops import rearrange
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
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
    return {
        "input": instruction,
        "evol_method": evol_method,
        "output": output,
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
        embeddings = fe.apply(v)
        # embeddings = SparseRandomProjection(n_components="auto").fit_transform(embeddings)
        return embeddings

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


def build_lineage_graph(wizard_edges: List[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    # add first generation parents first
    for edge in wizard_edges:
        if edge["iteration"] == 0:
            parent_text = edge["sample"]["input"]
            rank = edge["rank"]
            parent = (0, parent_text)
            if parent in G.nodes:
                rank = min(rank, G.nodes[parent]["rank"])
            G.add_node(parent, text=parent_text, iteration=0, rank=rank, chosen=True, archived=False)

    # add the rest of the nodes
    for edge in wizard_edges:
        parent_text = edge["sample"]["input"]
        child_text = edge["sample"]["output"]
        i = edge["iteration"]
        rank = edge["rank"]
        chosen = edge["chosen?"]
        archived = edge["archived?"]
        method = EVOL_METHOD_NAMES[edge["sample"]["evol_method"]]

        # represent nodes as tuples so that different nodes can have the same text
        parent = (i, parent_text)
        child = (i+1, child_text)
        # G.add_node(parent, text=parent_text, iteration=i)
        G.add_node(child, text=child_text, iteration=i+1, rank=rank, chosen=chosen, archived=archived)
        G.add_edge(parent, child, method=method)
    return G


def draw_lineage_graph(G: nx.DiGraph, chosen_only=False, ancestors_only=False, scale_rank=False):
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
    archive_color = colors[-1]
    non_archive_color = colors[0]

    # color nodes by archive status
    node_colors = [archive_color if data["archived"] else non_archive_color
                   for _, data in G.nodes(data=True)]

    # color edges by evol method
    method_index = {n: i for i, n in enumerate(EVOL_METHOD_NAMES.values())}
    edge_colors = [colors[method_index[method]] for _, _, method in G.edges.data("method")]

    # extract node positions
    max_rank_by_iter = {}
    for _, d in G.nodes(data=True):
        max_rank_by_iter[d["iteration"]] = max(max_rank_by_iter.get(d["iteration"], 0),
                                               d["rank"])
    max_rank = max(max_rank_by_iter.values())

    # arrange nodes by iteration/rank; scale rank if desired
    def scaling_factor(i):
        return max_rank / max_rank_by_iter[i] if scale_rank else 1
    pos = {n : (data["iteration"],
                -data["rank"] * scaling_factor(data["iteration"]))
           for n, data in G.nodes(data=True)}

    fig = plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        edge_color=edge_colors,
        node_color=node_colors,
        node_size=100,
    )
    # fig.set_facecolor('#212121')
    plt.show()


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
    for _ ,v, data in G.edges(data=True):
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
                f.write(s+"\n")


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


if __name__ == "__main__":
    # run_search(
    #     iters=3,
    #     archive_per_iter=5,
    #     max_popn_size=10,
    #     seed_dataset="../datasets/code_alpaca_tiny.json",
    #     output_file="../datasets/code_alpaca_tiny_nov_10xAll.jsonl"
    # )

    # dataset = util.load_jsonl("../datasets/code_alpaca_tiny_nov_10xAll.jsonl")
    dataset = util.load_jsonl("../datasets/code_alpaca_100_nov_100xAll.jsonl")
    G = build_lineage_graph(dataset)
    # draw_lineage_graph(G, chosen_only=True)
    # rank_distro(G, rank_cap=500)
    avg_rank_by_iter(G)

    # count up number of nodes in each iteration
    counts = {}
    for _, i in G.nodes.data(data="iteration"):
        counts[i] = counts.get(i, 0) + 1
    print(counts)


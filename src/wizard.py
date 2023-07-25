import copy
import os
import random
import sys
from typing import List, Generator, Union
import numpy as np
import json

from matplotlib import pyplot as plt
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

def evol_instruct_step(chat: ChatOpenAI, input: str, evol_method: str) -> dict:
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Please increase the difficulty of the given programming test question a bit. "
        "You can increase the difficulty using, but not limited to, the following methods: {evol_method}"
        "Please respond with a new programming test question that can be understood independently "
        "without any reference to the original question. "
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    output = chain.run(evol_method=evol_method, input=input)
    return {
        "input": input,
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
                   samples_per_datum: int,
                   archive_per_iter: int) -> Generator[List[dict], None, None]:

    def take_samples(in_data: Union[List[str], np.ndarray]):
        # flat generator over samples from evol-instruct
        seen = set(in_data)
        for x in tqdm(in_data, total=len(in_data)):
            for _ in range(samples_per_datum):
                while True:
                    evol_method = random.choice(evol_methods)
                    sample = evol_instruct_step(chat, x, evol_method)
                    if sample["output"] not in seen:
                        break
                    else:
                        print("Repeated sample", file=sys.stderr)
                seen.add(sample["output"])
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
            i_selected = ranks[:len(popn)]

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
            })
            yield [{"iteration": i,
                    "sample": sample,
                    "score": float(dists[j]),
                    "rank": int(inverted_ranks[j]),
                    "chosen?": bool(j in i_selected),
                    "archived?": bool(j in i_archived),
                   }
                   for j, sample in enumerate(samples)]


def graph_lineage(filename: str, chosen_only=False):
    """
    Visualize the lineage of a dataset generated by novel_instruct as a graph, where
    nodes are instructions and edges are mutations.
    """
    data = util.load_jsonl(filename)
    G = nx.DiGraph()
    pos = {}
    archived_nodes = set()
    for d in data:
        i = d["iteration"]
        rank = -d["rank"]
        parent = d["sample"]["input"]
        child = d["sample"]["output"]
        chosen = d["chosen?"]
        archived = d["archived?"]

        if chosen_only and not chosen:
            continue

        # label edge with evol method
        method = EVOL_METHOD_NAMES[d["sample"]["evol_method"]]
        G.add_edge(parent, child, method=method)

        # update pos
        pos[child] = (i, rank)
        if parent not in pos:
            pos[parent] = (i-1, rank)

        # color if archived
        if archived:
            archived_nodes.add(child)

    i_name = {n : i for i, n in enumerate(EVOL_METHOD_NAMES.values())}
    colors = palettable.matplotlib.get_map("Viridis_5").mpl_colors
    edge_colors = [colors[i_name[method]] for _, _, method in G.edges.data("method")]
    node_colors = [colors[-1] if node in archived_nodes else colors[0]
                   for node in G.nodes]
    nx.draw(G, pos=pos, with_labels=False,
            node_color=node_colors, node_size=100,
            edge_color=edge_colors)
    plt.show()


def run_search(iters: int, seed_dataset: str, samples_per_datum: int, archive_per_iter: int, output_file: str):
    instructions = [x["instruction"] for x in json.load(open(seed_dataset, "r"))]
    chat = ChatOpenAI(temperature=0.9, client=None)
    wandb.init(project="wizard")
    fe = feat.SentenceFeaturizer()
    with open(output_file, "w") as f:
        for log in novel_instruct(chat, fe,
                                  iters=iters,
                                  seed_dataset=instructions,
                                  evol_methods=EVOL_METHODS,
                                  samples_per_datum=samples_per_datum,
                                  archive_per_iter=archive_per_iter):
            for entry in log:
                s = json.dumps(entry, indent=None)
                f.write(s+"\n")


if __name__ == "__main__":
    # run_search(
    #     iters=5,
    #     seed_dataset="../datasets/code_alpaca_tiny.json",
    #     samples_per_datum=4,
    #     archive_per_iter=5,
    #     output_file="../datasets/code_alpaca_tiny_nov.jsonl"
    # )
    graph_lineage("../datasets/code_alpaca_tiny_nov.jsonl")

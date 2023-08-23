import copy
import os
import random
import sys
from math import ceil
from typing import List, Generator, Union, Tuple
import numpy as np
import json
from sklearn import preprocessing, random_projection
from sklearn.manifold import MDS
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
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
MUTATORS = {
    "new constraints": "Add new constraints and requirements to the original problem, adding approximately 10 additional words.",
    "more specific": "Replace a commonly used requirement in the programming task with a less common and more specific one.",
    "more steps": "If the original problem can be solved with only a few logical steps, please add more reasoning steps.",
    "misdirection": "Provide a piece of erroneous code as a reference to increase misdirection.",
    "higher complexity": "Propose higher time or space complexity requirements, but please refrain from doing so frequently.",
}


def mutator_name(method_text: str) -> str:
    for k, v in MUTATORS.items():
        if v == method_text:
            return k
    return "unknown"


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


def mutate_problem(chat: ChatOpenAI, problem: str, mutator: str) -> Tuple[str, str]:
    """Use the LLM to mutate the given programming problem"""
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Please increase the difficulty of the given programming test question a bit. "
        "You can increase the difficulty using, but not limited to, the following methods: {mutator}"
        "Your response should consist of a new programming test question that is entirely self-contained: "
        "it should be solvable without reference to the original question. "
        "The new programming test question should be solvable without a network connection or any system libraries. "
        "Output only the new programming question, with no additional text."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    output = chain.run(mutator=mutator, input=problem)
    name = name_programming_problem(chat, output)
    return name, output


def propose_solution(chat: ChatOpenAI, problem: str) -> str:
    """Prompt the LLM to solve a programming problem"""
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Please write a solution in Python to the following programming test question."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(input=problem)


def propose_checker(chat: ChatOpenAI, problem: str) -> str:
    """
    Prompt the LLM to write a checker for a programming problem.

    A checker is a function f that takes in a proposed solution g
    and returns True iff g is a correct solution to the problem:
    """
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Given a programming problem p, a checker for p is a function f that takes in a proposed solution function g "
        "and returns True if and only if g is a correct solution to the problem. "
        "Please write a checker in Python for the following programming test question. "
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(input=problem)


# generate a dataset using Evol-Instruct from the WizardLM paper
def evol_instruct(chat: ChatOpenAI, iters: int, seed_dataset: List[str], mutators: List[str]) -> Generator[
    dict, None, None]:
    dataset = seed_dataset
    idgen = util.IdGen()
    with get_openai_callback() as cb:
        for i, text in tqdm(enumerate(dataset, start=1), total=len(dataset), desc="Chain", position=0):
            # Evolve each initial datapoint independently of the others
            parent_id = idgen.next()
            for gen in tqdm(range(iters), desc="Generation", position=1, leave=False):
                mutator = random.choice(mutators)
                name, text = mutate_problem(chat, text, mutator)
                soln = propose_solution(chat, text)
                checker = propose_checker(chat, text)
                child_id = idgen.next()
                yield {
                    "id": child_id,
                    "iter": gen,
                    "parent": parent_id,
                    "mutator": mutator_name(mutator),
                    "name": name,
                    "text": text,
                    "solution": soln,
                    "checker": checker,
                }
                parent_id = child_id

            wandb.log({
                "iter": i,
                "token usage": {
                    "total cost": cb.total_cost,
                    "total tokens": cb.total_tokens,
                    "prompt tokens": cb.prompt_tokens,
                    "completion tokens": cb.completion_tokens,
                },
            })


# generate a small dataset using Evol-Instruct + novelty pressure
def novel_instruct(chat: ChatOpenAI,
                   fe: feat.Featurizer,
                   iters: int,
                   seed_dataset: List[str],
                   mutators: List[str],
                   max_popn_size: int,
                   archive_per_iter: int) -> Generator[List[dict], None, None]:
    def take_samples(in_data: Union[List[str], np.ndarray]):
        # flat generator over samples from evol-instruct
        for x in tqdm(in_data, total=len(in_data)):
            for mutator in mutators:
                sample = mutate_problem(chat, x, mutator)
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


def run_search(iters: int, seed_dataset: str, archive_per_iter: int, max_popn_size: int, output_file: str):
    instructions = [x["instruction"] for x in json.load(open(seed_dataset, "r"))]
    chat = ChatOpenAI(temperature=0.9, client=None)
    wandb.init(project="wizard")
    fe = feat.SentenceFeaturizer()
    with open(output_file, "w") as f:
        for log in novel_instruct(chat, fe,
                                  iters=iters,
                                  seed_dataset=instructions,
                                  mutators=MUTATORS,
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
    wandb.init(project="wizard")
    chat = ChatOpenAI(temperature=0.9, client=None)
    with open("../datasets/evolinstruct-singlethread.jsonl", "w") as f:
        for entry in evol_instruct(
                chat=chat,
                iters=5,
                seed_dataset=[
                    "Create an array of length 5 which contains all even numbers between 1 and 10.",
                ],
                mutators=list(MUTATORS.values()),
        ):
            json.dump(entry, f)
            f.write("\n")

# run_search(
#     iters=20,
#     archive_per_iter=5,
#     max_popn_size=10,
#     seed_dataset="../datasets/code_alpaca_tiny.json",
#     output_file="../datasets/code_alpaca_tiny_nov_10x20_1.jsonl"
# )

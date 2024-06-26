import copy
import datetime
import os
import random
from tqdm import tqdm
from typing import List, Generator, Union, Tuple, Optional, Iterator
import numpy as np
import json

import pandas as pd
from sklearn import preprocessing, random_projection
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

import prompts
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


def simple_chat_prompt(system_prompt: str, user_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt),
    ])


def propose_name(chat: ChatOpenAI, problem: str) -> str:
    return prompts.run_saved_prompt(
        chat,
        key="name problem",
        problem=problem,
    )


def mutate_problem(chat: ChatOpenAI, problem: str, mutator: str) -> str:
    """Use the LLM to mutate the given programming problem"""
    return prompts.run_saved_prompt(
        chat,
        key="mutate problem",
        problem=problem,
        mutator=mutator,
    )


def wizard_solve(chat: ChatOpenAI, problem: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n"
            "\n"
            "### Instruction:\n"
            "{instruction}\n"
            "\n"
            "### Response:"
        ),
    ])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(instruction=problem)


def check_problem_novel(chat: ChatOpenAI, src_problem: str, dst_problem: str) -> str:
    """Check that a problem is sufficiently different from its parent"""
    return prompts.run_saved_prompt(
        chat,
        key="check novelty",
        src_problem=src_problem,
        dst_problem=dst_problem,
    )


def check_problem_solvable(chat: ChatOpenAI, problem: str) -> str:
    """Check that a problem can be solved by LLM"""
    return prompts.run_saved_prompt(
        chat,
        key="check solvable",
        problem=problem,
    )


# generate a dataset using Evol-Instruct from the WizardLM paper
def evol_instruct(chat: ChatOpenAI, iters: int, seed_dataset: List[str], mutators: List[str]) -> Iterator[dict]:
    dataset = seed_dataset
    ids = util.IdGen()
    for i, text in enumerate(dataset, start=1):
        # Evolve each initial datapoint independently of the others
        root_id = ids.next()
        root_name = propose_name(chat, text)
        yield {
            "id": root_id,
            "iter": 0,
            "root": root_id,
            "parent": root_id,
            "mutator": None,
            "name": root_name,
            "text": text,
        }
        parent_id = root_id
        parent_name = root_name
        for gen in range(1, iters + 1):
            mutator = random.choice(mutators)
            text = mutate_problem(chat, text, mutator)
            name = propose_name(chat, text)
            id = ids.next()
            yield {
                "id": id,
                "iter": gen,
                "root": root_id,
                "parent": parent_id,
                "parent name": parent_name,
                "mutator": mutator_name(mutator),
                "text": text,
                "name": name,
            }
            parent_id = id
            parent_name = name


def embed(ft: feat.Featurizer, texts: Union[List[str], np.ndarray], batch_size=128, saveto=None) -> np.ndarray:
    # use sentence transformer to generate embeddings
    embeddings = []
    for batch in tqdm(util.batched(texts, batch_size=batch_size), total=len(texts) // batch_size):
        embeddings.extend(ft.apply(batch))
    embeddings = np.array(embeddings)
    scaler = preprocessing.StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    if saveto:
        np.save(file=saveto, arr=embeddings)
    return embeddings


# generate a small dataset using Evol-Instruct + novelty pressure
def novel_instruct(chat: ChatOpenAI,
                   ft: feat.Featurizer,
                   iters: int,
                   seed_dataset: List[str],
                   mutators: List[str],
                   max_popn_size: int,
                   archive_per_iter: int) -> Iterator[dict]:
    idgen = util.IdGen()
    knn = NearestNeighbors(metric="minkowski")
    e_popn = embed(ft, seed_dataset)
    popn = []
    for text in seed_dataset:
        root_id = idgen.next()
        root = {
            "id": root_id,
            "iter": 0,
            "root": root_id,
            "parent": root_id,
            "mutator": None,
            "name": propose_name(chat, text),
            "text": text,
            "score": None,
            "rank": None,
            "chosen?": True,
            "archived?": False,
        }
        popn.append(root)
        yield {
            "kind": "data",
            "payload": root,
        }
    archive = []
    e_archive = []

    def take_sample(x: dict, m: str) -> dict:
        text = mutate_problem(chat, x["text"], m)
        return {
            "id": idgen.next(),
            "iter": i,
            "root": x["root"],
            "parent": x["id"],
            "mutator": mutator_name(m),
            "name": propose_name(chat, text),
            "text": text,
            "score": None,
            "rank": None,
            "chosen?": None,
            "archived?": None,
        }

    for i in range(1, iters + 1):
        # take samples using evol-instruct
        samples = np.array([take_sample(x, m)
                            for x in popn
                            for m in mutators], dtype=object)

        # use novelty pressure to select best samples
        knn.fit(np.concatenate([e_popn, e_archive], axis=0) if archive else e_popn)
        e_samples = embed(ft, [s["text"] for s in samples])
        dists, _ = knn.kneighbors(e_samples)
        dists = np.mean(dists, axis=1)
        ranks = np.argsort(-dists)
        i_selected = ranks[:max_popn_size]

        # update archive: take a random selection of samples and add them to the archive
        i_archived = np.random.choice(len(samples), size=archive_per_iter, replace=False)
        archive.extend(samples[i_archived])
        e_archive.extend(e_samples[i_archived])

        # update popn, embeddings
        popn = samples[i_selected]
        e_popn = e_samples[i_selected]

        # log everything
        yield {
            "kind": "log",
            "payload": {
                "scores": dists,
                "best": [s["name"] for s in samples[ranks[:3]]],
                "worst": [s["name"] for s in samples[ranks[-3:][::-1]]],
            },
        }

        # update samples with score, rank, chosen?, archived?, then output samples
        inverted_ranks = util.invert_array(ranks)
        for j, d in enumerate(samples):
            d.update({
                "score": float(dists[j]),
                "rank": int(inverted_ranks[j]),
                "chosen?": bool(j in i_selected),
                "archived?": bool(j in i_archived),
            })
            yield {
                "kind": "data",
                "payload": d,
            }


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


def run_evol_instruct(chat: ChatOpenAI, outfile: str, iters: int, seed_dataset_file: str):
    instructions = [x["instruction"] for x in json.load(open(seed_dataset_file, "r"))]
    wandb.init(project="wizard")
    log_table = wandb.Table(columns=["root name", "iter", "parent name", "name", "text"])
    with open(outfile, "w") as f, get_openai_callback() as cb:
        for d in evol_instruct(
                chat=chat,
                iters=iters,
                seed_dataset=instructions,
                mutators=list(MUTATORS.values())):
            s = json.dumps(d, indent=None)
            print(s)
            f.write(s + "\n")

            # log to wandb
            log_table.add_data(d["root"], d["iter"], d["parent name"], d["name"], d["text"])
            wandb.log({
                "iter": d["iter"],
                "root": d["root"],
                "names": copy.copy(log_table),
                "total cost": cb.total_cost,
            })


def run_novel_instruct(chat: ChatOpenAI, iters: int, seed_dataset: Union[str, List], archive_per_iter: int,
                       max_popn_size: int, output_file: str):
    if isinstance(seed_dataset, str):
        instructions = [x["instruction"] for x in json.load(open(seed_dataset, "r"))]
    else:
        assert isinstance(seed_dataset, list)
        instructions = seed_dataset

    fe = feat.SentenceFeaturizer()
    wandb.init(project="wizard")
    log_table = wandb.Table(columns=["top-1", "top-2", "top-3", "bot-1", "bot-2", "bot-3"])

    with open(output_file, "w") as f, get_openai_callback() as cb:
        for d in novel_instruct(chat, fe,
                                     iters=iters,
                                     seed_dataset=instructions,
                                     mutators=list(MUTATORS.values()),
                                     max_popn_size=max_popn_size,
                                     archive_per_iter=archive_per_iter):
            kind = d["kind"]
            payload = d["payload"]

            if kind == "data":
                s = json.dumps(payload, indent=None)
                print(s)
                f.write(s + "\n")
            elif kind == "log":
                scores = payload["scores"]
                best = payload["best"]
                worst = payload["worst"]

                # log to wandb
                log_table.add_data(*(best + worst))
                wandb.log({
                    "scores": wandb.Histogram(scores),
                    "examples": copy.copy(log_table),
                    "total cost": cb.total_cost,
                })
            else:
                raise ValueError(f"Unknown output kind: {kind} with payload {payload}")


if __name__ == "__main__":
    timestamp = util.timestamp()
    iters = 2
    chat = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")

    # run_evol_instruct(
    #     chat=chat,
    #     outfile=f"../datasets/evol-instruct-{iters}-{timestamp}.jsonl",
    #     iters=iters,
    #     seed_dataset="../datasets/code_alpaca_tiny.json",,
    # )
    tiny_seed = [  # first few project euler problems
        "If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. "
        "The sum of these multiples is 23. Find the sum of all the multiples of 3 or 5 below 1000.",
        "Find the sum of all even terms in the Fibonacci sequence whose values do not exceed four million.",
        "Find the largest prime factor of the number 600851475143.",
        "Find the largest palindrome made from the product of two 3-digit numbers.",
        "What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?",
        "What is the 10,001st prime number?",
        "A Pythagorean triplet is a set of three natural numbers, a < b < c, for which, a^2 + b^2 = c^2. "
        "For example, 3^2 + 4^2 = 9 + 16 = 25 = 5^2. There exists exactly one Pythagorean triplet for which a + b + c = 1000. "
        "Find the product abc.",
    ]
    problem_sample = [

    ]
    for problem in tiny_seed:
        out = wizard_solve(chat, problem)
        print(f"Problem:",
              f"{problem}",
              f"",
              f"Solution:",
              f"{out}",
              sep="\n")

    # run_novel_instruct(
    #     chat=chat,
    #     iters=iters,
    #     seed_dataset=tiny_seed,  # "../datasets/code_alpaca_tiny.json",
    #     archive_per_iter=5,
    #     max_popn_size=10,
    #     output_file=f"../datasets/novel-instruct-test-{iters}-{timestamp}.jsonl"
    # )

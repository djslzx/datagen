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


def propose_name(chat: ChatOpenAI, text: str) -> str:
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


def mutate_problem(chat: ChatOpenAI, problem: str, mutator: str) -> str:
    """Use the LLM to mutate the given programming problem"""
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Please increase the difficulty of the given programming test question a bit. "
        "You can increase the difficulty using, but not limited to, the following methods: {mutator}"
        "Your response should consist of a new programming test question that is entirely self-contained: "
        "it should be solvable without "
        "(a) knowing the original question, "
        "(b) having a network connection, or"
        "(c) using any system calls. "
        "Output only the new programming question, with no additional text."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    output = chain.run(mutator=mutator, input=problem)
    return output


def filter_problem(chat: ChatOpenAI, problem: str) -> Optional[bool]:
    """Use the LLM to filter out programming problems that are not self-contained"""
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Please determine whether the following programming test question is valid. "
        "A programming test question is valid if it is entirely self-contained, i.e., "
        "it is solvable without "
        "(a) referencing an 'original question', "
        "(b) having a network connection, or"
        "(c) using any system calls. "
        "Output only True or False, with no additional text."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    output = chain.run(input=problem)
    if output == "True":
        return True
    elif output == "False":
        return False
    else:
        return None


def propose_solution(chat: ChatOpenAI, problem: str) -> str:
    """Prompt the LLM to solve a programming problem"""
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Please write a solution in Python to the following programming test question. "
        "Your solution should only use the Python standard library. "
        "Output only the solution, with no additional text or comments. "
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
        "A simple example of a checker is a function that tests g against a set of input-output pairs. "
        "In cases where generating these input-output pairs is more difficult, a checker may instead sample outputs from g"
        "and verify that they satisfy the problem's constraints."
        "Please write a deterministic checker in Python for the following programming problem. "
        "Do not include a solution to the programming problem in your response. "
        "Output only the checker, with no additional text. "
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(input=problem)


# generate a dataset using Evol-Instruct from the WizardLM paper
def evol_instruct(chat: ChatOpenAI, iters: int, seed_dataset: List[str], mutators: List[str]) -> Iterator[dict]:
    dataset = seed_dataset
    idgen = util.IdGen()
    with get_openai_callback() as cb:
        log_table = wandb.Table(columns=["root name", "iter", "parent name", "name", "text"])
        for i, text in enumerate(dataset, start=1):
            # Evolve each initial datapoint independently of the others
            root_id = idgen.next()
            root_name = propose_name(chat, text)
            yield {
                "id": root_id,
                "iter": 0,
                "root": root_id,
                "parent": root_id,
                "mutator": None,
                "name": root_name,
                "text": text,
                "solution": propose_solution(chat, text),
                "checker": propose_checker(chat, text),
            }
            parent_id = root_id
            parent_name = root_name
            for gen in range(1, iters + 1):
                mutator = random.choice(mutators)
                text = mutate_problem(chat, text, mutator)
                name = propose_name(chat, text)
                id = idgen.next()
                yield {
                    "id": id,
                    "iter": gen,
                    "root": root_id,
                    "parent": parent_id,
                    "mutator": mutator_name(mutator),
                    "text": text,
                    "name": name,
                    "solution": propose_solution(chat, text),
                    "checker": propose_checker(chat, text),
                }
                log_table.add_data(root_name, gen, parent_name, name, text)
                wandb.log({
                    "iter": i,
                    "root": root_id,
                    "names": copy.copy(log_table),
                    "token usage": {
                        "total cost": cb.total_cost,
                        "total tokens": cb.total_tokens,
                        "prompt tokens": cb.prompt_tokens,
                        "completion tokens": cb.completion_tokens,
                    },
                })
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
                   archive_per_iter: int) -> Iterator[List[dict]]:
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
            # "solution": propose_solution(chat, text),
            # "checker": propose_checker(chat, text),
            "score": None,
            "rank": None,
            "chosen?": True,
            "archived?": False,
        }
        popn.append(root)
        yield root
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
            # "solution": propose_solution(chat, text),
            # "checker": propose_checker(chat, text),
            "score": None,
            "rank": None,
            "chosen?": None,
            "archived?": None,
        }

    log_table = wandb.Table(columns=["top-1", "top-2", "top-3", "bot-1", "bot-2", "bot-3"])
    with get_openai_callback() as cb:  # expose cost information
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
            inverted_ranks = util.invert_array(ranks)
            log_table.add_data(*([s["name"] for s in samples[ranks[:3]]] +
                                 [s["name"] for s in samples[ranks[-3:][::-1]]]))  # best and worst 3
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

            # update samples with score, rank, chosen?, archived?, then output samples
            for j, d in enumerate(samples):
                d.update({
                    "score": float(dists[j]),
                    "rank": int(inverted_ranks[j]),
                    "chosen?": bool(j in i_selected),
                    "archived?": bool(j in i_archived),
                })
                yield d


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


def run_evol_instruct(outfile: str, iters: int, seed_dataset_file: str):
    instructions = [x["instruction"] for x in json.load(open(seed_dataset_file, "r"))]
    wandb.init(project="wizard")
    chat = ChatOpenAI(temperature=0.9, client=None)
    with open(outfile, "w") as f:
        for entry in evol_instruct(
                chat=chat,
                iters=iters,
                seed_dataset=instructions,
                mutators=list(MUTATORS.values())):
            s = json.dumps(entry, indent=None)
            print(s)
            f.write(s + "\n")


def run_novel_instruct(iters: int, seed_dataset: Union[str, List], archive_per_iter: int, max_popn_size: int, output_file: str):
    if isinstance(seed_dataset, str):
        instructions = [x["instruction"] for x in json.load(open(seed_dataset_file, "r"))]
    else:
        assert isinstance(seed_dataset, list)
        instructions = seed_dataset
    chat = ChatOpenAI(temperature=0.9, client=None)
    wandb.init(project="wizard")
    fe = feat.SentenceFeaturizer()
    with open(output_file, "w") as f:
        for entry in novel_instruct(chat, fe,
                                    iters=iters,
                                    seed_dataset=instructions,
                                    mutators=list(MUTATORS.values()),
                                    max_popn_size=max_popn_size,
                                    archive_per_iter=archive_per_iter):
            s = json.dumps(entry, indent=None)
            print(s)
            f.write(s + "\n")


if __name__ == "__main__":
    timestamp = datetime.datetime.now().isoformat()
    iters = 2
    # run_evol_instruct(
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
    run_novel_instruct(
        iters=iters,
        seed_dataset=tiny_seed,  #"../datasets/code_alpaca_tiny.json",
        archive_per_iter=5,
        max_popn_size=10,
        output_file=f"../datasets/novel-instruct-test-{iters}-{timestamp}.jsonl"
    )

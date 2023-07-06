import os
import random

import openai
import numpy as np
import json
from typing import List, Generator, Union

from einops import rearrange
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection

import featurizers as feat

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

CODE_DATASET = json.load(open("../datasets/code_alpaca_tiny.json", "r"))


def to_str(json_item: dict) -> str:
    return f"instruction: {json_item['instruction']}\n" \
           f"input: {json_item['input']}\n" \
           f"output: {json_item['output']}" \


def make_system_prompt(method: str) -> str:
    return f"Please increase the difficulty of the given programming test question a bit. " \
           f"You can increase the difficulty using, but not limited to, the following methods: " \
           f"{method}"


def prompt_chatgpt(mutation_method: str, datum: str) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": make_system_prompt(mutation_method)},
            {"role": "user", "content": datum},
        ],
    )
    return completion.choices[0].message["content"]


# generate a small dataset using Evol-Instruct from the WizardLM paper
def evol_instruct_step(input: str, evol_methods: List[str]) -> dict:
    mutation_method = random.choice(evol_methods)
    return {
        "input": input,
        "mutation_method": mutation_method,
        "output": prompt_chatgpt(mutation_method, input),
    }


# generate a small dataset using Evol-Instruct + novelty pressure
def novel_instruct(iters: int,
                   seed_dataset: List[str],
                   evol_methods: List[str],
                   samples_per_datum: int,
                   archive_per_iter: int) -> Generator[List[dict], None, None]:

    def take_samples(in_data: List[str]):
        # flat generator over samples from evol-instruct
        for x in in_data:
            for _ in range(samples_per_datum):
                yield evol_instruct_step(x, evol_methods)

    def update_archive(A, E_A, S, E_S):
        # take a random selection of samples and add them to the archive
        I_A = np.random.choice(len(S), size=archive_per_iter, replace=False)
        A.extend(S[I_A])
        E_A.extend(E_S[I_A])

    cg = feat.CodeGen(size="350M")

    def embed(v: List[str]) -> np.ndarray:
        # embed a vector using the CodeGen featurizer
        embeddings = rearrange(cg.apply(v), "b n d -> b (n d)")
        embeddings = SparseRandomProjection(n_components=2000).fit_transform(embeddings)
        return embeddings

    knn = NearestNeighbors(metric="minkowski")
    popn = seed_dataset
    e_popn = embed(popn)
    archive = []
    e_archive = []

    for i in range(iters):
        # take samples using evol-instruct
        samples = np.array(list(take_samples(popn)), dtype=object)
        s_samples = np.array([d["output"] for d in samples], dtype=object)

        # use novelty pressure to select best samples
        knn.fit(np.concatenate([e_popn, e_archive], axis=0) if archive else e_popn)
        e_samples = embed(s_samples.tolist())
        dists, _ = knn.kneighbors(e_samples)
        dists = np.mean(dists, axis=1)
        i_selected = np.argsort(-dists)[:len(popn)]

        update_archive(archive, e_archive, s_samples, e_samples)
        popn = s_samples[i_selected]
        e_popn = e_samples[i_selected]

        # log generation
        yield samples[i_selected].tolist()



if __name__ == "__main__":
    str_data = [to_str(datum) for datum in CODE_DATASET]
    # with open("../datasets/code_alpaca_tiny_evol.jsonl", "w") as f:
    #     for msg in evol_instruct_step(str_data, EVOL_METHODS):
    #         print(msg)
    #         f.write(json.dumps(msg, indent=4) + "\n")
    with open("../datasets/code_alpaca_tiny_nov.jsonl", "w") as f:
        for msg in novel_instruct(iters=2,
                                  seed_dataset=str_data,
                                  evol_methods=EVOL_METHODS,
                                  samples_per_datum=2,
                                  archive_per_iter=5):
            print(msg)
            f.write(json.dumps(msg, indent=4))

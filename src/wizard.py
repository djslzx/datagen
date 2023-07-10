import os
import random
import sys

from typing import List, Generator
import numpy as np
import json
from tqdm import tqdm
from einops import rearrange
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection

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


def evol_instruct_step(chat: ChatOpenAI, input: str, evol_methods: List[str]) -> dict:
    evol_method = random.choice(evol_methods)
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Please increase the difficulty of the given programming test question a bit. "
        "You can increase the difficulty using, but not limited to, the following methods: {evol_method}"
        "Respond with the modified programming test question. You should not reference the original question. "
        "Keep the original format of instruction, input, and output."
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
                completion = evol_instruct_step(chat, prompt, evol_methods)
                prompt = completion["output"]
                completions.append({"evol_method": completion["evol_method"],
                                    "output": prompt})

            # update log file
            json.dump({"input": x, "completions": completions}, f)
            f.write("\n")

            print(f"Completed evolution of instruction {i} of {len(dataset)}", file=sys.stderr)
            print(cb)


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
        ranks = np.argsort(-dists)
        i_selected = ranks[:len(popn)]

        update_archive(archive, e_archive, s_samples, e_samples)
        popn = s_samples[i_selected]
        e_popn = e_samples[i_selected]

        # log generation
        inverted_ranks = util.invert_array(ranks)
        yield [sample.update({"iteration": i,
                              "score": dists[j],
                              "rank": inverted_ranks[j]})
               for j, sample in enumerate(samples)]



if __name__ == "__main__":
    data = [util.dict_to_text(x) for x in json.load(open("../datasets/code_alpaca_tiny.json", "r"))]
    chat = ChatOpenAI(temperature=0.9)
    evol_instruct(
        chat,
        iters=3,
        seed_dataset=data,
        evol_methods=EVOL_METHODS,
        log_file="../datasets/evol_instruct_5x3.jsonl"
    )
    util.pp_jsonl("../datasets/evol_instruct_5x3.jsonl")

    # with open("../datasets/code_alpaca_tiny_nov.jsonl", "w") as f:
    #     for msg in novel_instruct(iters=2,
    #                               seed_dataset=str_data,
    #                               evol_methods=EVOL_METHODS,
    #                               samples_per_datum=2,
    #                               archive_per_iter=5):
    #         s = json.dumps(msg, indent=4)
    #         print(s, file=sys.stderr)
    #         f.write(s + "\n")

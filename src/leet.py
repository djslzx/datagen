"""
Experiments with leetcode dataset
"""

import json
from pprint import pp
from tqdm import tqdm
import itertools as it
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Dict

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import featurizers as feat
import util
import wizard


# try mapping a bunch of names to problems
# 1. zero-shot
# 2. few-shot with leetcode problems

def zero_shot_name_to_problem(chat: ChatOpenAI, name: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful AI assistant."),
        HumanMessagePromptTemplate.from_template(
            "Come up with a leetcode-style programming problem that suits the given problem name.\n"
            "\n"
            "### Name\n"
            "{name}\n"
            "\n"
            "### Problem"
        )
    ])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(name=name)


def few_shot_name_to_problem(chat: ChatOpenAI, df: pd.DataFrame, name: str) -> str:
    samples = df[["title", "description"]][df["title"] != name].sample(3)
    samples_prompt = "\n".join([
        f"###Name\n"
        f"{s['title']}\n\n"
        f"###Problem\n"
        f"{s['description']}\n"
        for _, s in samples.iterrows()
    ]).replace("{", "{{").replace("}", "}}")
    print(f"Few-shot prompt: {samples_prompt}")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful AI assistant."),
        HumanMessagePromptTemplate.from_template(
            samples_prompt +
            "###Name\n"
            "{name}\n\n"
            "###Problem"
        )
    ])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(name=name)


if __name__ == "__main__":
    names = [
        "Determine Color of a Chessboard Square",
        "Sentence Similarity III",
        "Count Nice Pairs in an Array",
        "Maximum Number of Groups Getting Fresh Donuts",
        "Truncate Sentence",
        "Finding the Users Active Minutes",
        "Minimum Absolute Sum Difference",
        "Number of Different Subsequences GCDs",
        "Maximum Number of Accepted Invitations",
        "Find Customers With Positive Revenue this Year",
        "Sign of the Product of an Array",
        "Find the Winner of the Circular Game",
        "Minimum Sideway Jumps",
        "Finding MK Average",
    ]

    chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")

    # actual leetcode problem
    df = pd.read_csv("../datasets/leetcode.csv")
    subset = df[df["title"].apply(lambda x: x in names)].copy()
    print(subset["description"])

    # zshots = []
    # for name in names:
    #     zshot = zero_shot_name_to_problem(chat, name)
    #     print(f"{name} => {zshot}")
    #     zshots.append(zshot)
    # subset["zeroshot"] = zshots

    generated = []
    for name in names:
        fshot = few_shot_name_to_problem(chat, df, name)
        print(f"{name} => {fshot}")
        generated.append(fshot)
    subset["fewshot"] = generated

    subset[["title", "description", "fewshot"]].to_csv("../datasets/leetcode-fewshot.csv")

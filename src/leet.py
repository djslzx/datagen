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
    timestamp = util.timestamp()
    names = [
        ('NS', 'nth Fibonacci Number Finder'),
        ('NS', "Minimum Spanning Tree (MST) using Prim's Algorithm."),
        ('NS', 'Longest Increasing Subsequence'),
        ('NS-euler', 'Longest Subarray with K Distinct Elements'),
        ('NS-euler', 'Palindrome Checker'),
        ('NS-euler', 'Message Analytics Manager'),
        ('Wiz-wide', 'HTML Webpage Generator'),
        ('Wiz-wide', 'Even Integer Sum'),
        ('Wiz-wide', 'Even Number List Sum'),
        ('Wiz-deep', 'Digit Square Sum Calculation'),
        ('Wiz-deep', 'Subset Sum'),
        ('Wiz-deep', 'Longest Subarray Target Sum'),
        # ('NS', 'Common Letters in Two Strings'),
        # ('NS', 'Average Acceleration Calculation'),
        # ('NS', 'Election Winner'),
        # ('NS-euler', 'Find All Divisors'),
        # ('NS-euler', 'Fibonacci Number Calculation Time Complexity: D) O(2^n)'),
        # ('NS-euler', 'User Manager System'),
        # ('Wiz-wide', 'Unique Alphabet Collecting'),
        # ('Wiz-wide', 'Pattern Printing Program'),
        # ('Wiz-wide', 'Minimum Serving Time with Queue'),
        # ('Wiz-deep', 'Count and Sort Frequent Elements'),
        # ('Wiz-deep', 'Find Duplicate Substrings'),
        # ('Wiz-deep', 'Employee Value Combinations'),
        ('Leetcode', "Determine Color of a Chessboard Square"),
        ('Leetcode', "Sentence Similarity III"),
        ('Leetcode', "Count Nice Pairs in an Array"),
        ('Leetcode', "Maximum Number of Groups Getting Fresh Donuts"),
        # ('Leetcode', "Truncate Sentence"),
        # ('Leetcode', "Finding the Users Active Minutes"),
        # ('Leetcode', "Minimum Absolute Sum Difference"),
        # ('Leetcode', "Number of Different Subsequences GCDs"),
        # ('Leetcode', "Maximum Number of Accepted Invitations"),
        # ('Leetcode', "Find Customers With Positive Revenue this Year"),
        # ('Leetcode', "Sign of the Product of an Array"),
        # ('Leetcode', "Find the Winner of the Circular Game"),
        # ('Leetcode', "Minimum Sideway Jumps"),
        # ('Leetcode', "Finding MK Average"),
    ]
    chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")

    # actual leetcode problems
    lc = pd.read_csv("../datasets/leetcode.csv")
    lc_names = [n for s, n in names if s == "Leetcode"]
    data = pd.read_json("../datasets/annotated-sep20.jsonl", lines=True)

    rows = []
    for source, name in tqdm(names):
        zshot = zero_shot_name_to_problem(chat, name)
        fshot = few_shot_name_to_problem(chat, lc, name)
        # zshot = "-".join(["zeroshot", source, name])
        # fshot = "-".join(["fewshot", source, name])
        if name in lc_names:
            orig = lc.loc[lc["title"] == name]["description"].values[0]
        else:
            orig = data.loc[data["name"] == name]["text"].values[0]
        rows.append({
            "name": name,
            "source": source,
            "zero shot": zshot,
            "few shot": fshot,
            "original": orig
        })
    df = pd.DataFrame(rows)
    print(df)
    df[["name", "source", "original", "zero shot", "few shot"]].to_csv(
        f"../datasets/leetcode-extended-{timestamp}.csv")

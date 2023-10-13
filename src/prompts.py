"""
Collect prompts here
"""
import os
from pprint import pp
from typing import List, Generator, Union, Tuple, Optional, Iterator
import random
import yaml
import numpy as np
import pandas as pd
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

# fetch prompts from prompt file
PROMPTS = yaml.load(open("../datasets/prompts/prompts.yaml", "r"), Loader=yaml.FullLoader)
print(f"Loaded prompts: {list(PROMPTS.keys())}")


def run_chain(chat: ChatOpenAI, sys_prompt: str, user_prompt: str, **kwargs) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt),
    ])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(**kwargs)


def sample_str(xs: List[str], n: int) -> str:
    return ", ".join([f"'{x}'" for x in random.choices(population=xs, k=n)])


def make_problem(chat: ChatOpenAI, concepts: List[str]) -> str:
    prompt = PROMPTS["generate coverless"]
    return run_chain(
        chat,
        sys_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        concepts=sample_str(concepts, 3)
    )


def restyle_problem(chat: ChatOpenAI, problem: str, concepts: List[str]) -> str:
    prompt = PROMPTS["restyle into coverless"]
    return run_chain(
        chat,
        sys_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        problem=problem,
        concepts=sample_str(concepts, 3)
    )


if __name__ == "__main__":
    PROBLEMS = [
        """
        Design a class called "Triangle" that represents a triangle. The "Triangle" class should have the following attributes:
        - "side1" (an integer) - The length of side 1 of the triangle.
        - "side2" (an integer) - The length of side 2 of the triangle.
        - "side3" (an integer) - The length of side 3 of the triangle.

        The "Triangle" class should also have the following methods:

        - "__init__" - Initializes a new instance of the "Triangle" class with the given lengths of all sides.
        - "get_perimeter" - Returns the perimeter of the triangle, which is calculated by adding all three side lengths together.
        - "get_area" - Returns the area of the triangle, which is calculated using Heron's formula: 
        area = sqrt(s * (s - a) * (s - b) * (s - c)), where s is the semiperimeter of the triangle and a, b, and c are the lengths of the sides.

        Implement the "Triangle" class in Python. Make sure to include a sample usage of the "Triangle" class to demonstrate its functionality.

        For example:

        t1 = Triangle(3, 4, 5)
        print(t1.get_perimeter())  # Output: 12
        print(t1.get_area())  # Output: 6.0
        """,
        """
        Write a function `multiplyArrays(arr1, arr2)` that takes in two arrays `arr1` and `arr2` of equal length, and returns an array `result` where each element `result[i]` is the product of `arr1[i]` and `arr2[i]`. 
        For example, given `arr1 = [2, 3, 4]` and `arr2 = [5, 6, 7]`, the function should return `[10, 18, 28]` since `10 = 2 * 5`, `18 = 3 * 6`, and `28 = 4 * 7`. 
        Your implementation should have a time complexity of O(n), where n is the length of the input arrays.
        """,
        """
        Given a binary tree, validate if it is a binary search tree (BST).

        A BST is defined as follows:
        - The left subtree of a node contains only nodes with keys less than the node's key.
        - The right subtree of a node contains only nodes with keys greater than the node's key.
        - Both the left and right subtrees must also be binary search trees.
        
        For example, given the following tree:
        
                5
               / \
              3   7
             / \
            1   4
        
        The output should be `true` since this tree satisfies the BST property.
        
        Note:
        - Assume that each node in the tree has a unique key value.
        - The tree may be unbalanced.
        """,
    ]
    CHAT = ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo-16k-0613")
    STORIES = [
        "You are a bank manager...",
        "You are a bank robber...",
        "Fibonacci numbers are...",
        "A perfect number is...",
        "You wake up in a strange room...",
        "It is the year 1963...",
    ]
    CONCEPTS = [
        "Recursion",
        "Trie data structure",
        "Stack programs",
        "Lisp interpreter",
        "Parallel computation",
        "Graph theory",
        "Knot theory",
        "Greedy algorithm",
    ]
    for problem in PROBLEMS:
        print("new problem:")
        print(make_problem(CHAT, concepts=CONCEPTS))
        print()

        print("original problem:")
        print(problem)
        print()

        print("restyled problem:")
        print(restyle_problem(CHAT, problem=problem, concepts=CONCEPTS))
        print()

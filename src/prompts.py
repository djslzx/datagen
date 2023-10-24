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


def run_prompt(chat: ChatOpenAI, system_prompt: str, user_prompt: str, **kwargs) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt),
    ])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(**kwargs)


def run_saved_prompt(chat: ChatOpenAI, key: str, **kwargs) -> str:
    prompt = PROMPTS[key]
    inputs = prompt["inputs"]
    assert len(inputs) == len(kwargs) and all([k in kwargs for k in inputs]), \
        f"Mismatched inputs: {inputs} vs {kwargs.keys()}"

    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]

    return run_prompt(
        chat,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        **kwargs,
    )


def sample_str(xs: List[str], n: int) -> str:
    return ", ".join([f"'{x}'" for x in random.choices(population=xs, k=n)])


def make_problem(chat: ChatOpenAI, concepts: List[str]) -> str:
    return run_saved_prompt(
        chat,
        key="new coverless",
        concepts=sample_str(concepts, 3),
    )


def restyle_problem(chat: ChatOpenAI, problem: str, concepts: List[str]) -> str:
    return run_saved_prompt(
        chat,
        key="restyle as coverless",
        problem=problem,
        concepts=sample_str(concepts, 3),
    )


def n_solns(chat: ChatOpenAI, problem: str, n: int) -> str:
    return run_saved_prompt(
        chat,
        key="n solutions",
        problem=problem,
        n=n,
    )


def n_tests(chat: ChatOpenAI, problem: str, n: int) -> str:
    return run_saved_prompt(
        chat,
        key="n tests",
        problem=problem,
        n=n,
    )


def split_solns(solns: str) -> List[str]:
    return util.split_py_markdown(solns)


def split_tests(text: str):
    text = util.strip_markdown(text)

    # split by decls
    blocks = ["def test_" + x
              for x in text.split("def test_")[1:]
              if x.strip()]
    out = []
    # only keep indented lines; stop at first non-indented line
    for block in blocks:
        lines = block.split("\n")
        block_text = lines[0] + "\n"
        for line in lines[1:]:
            if line.startswith("    "):
                block_text += line + "\n"
            else:
                break
        out.append(block_text)
    return out


def generate_solns_and_tests(problems: List[str]) -> Generator[Tuple[int, str, str], None, None]:
    for id, problem in enumerate(problems):
        yield id, "original", problem
        orig_solns = n_solns(CHAT, problem=problem, n=3)
        yield id, "original solutions", orig_solns
        orig_tests = n_tests(CHAT, problem=problem, n=3)
        yield id, "original tests", orig_tests
        orig_tests_split = split_tests(orig_tests)
        for i, test in enumerate(orig_tests_split):
            yield id, f"original tests split {i}", test

        restyled = restyle_problem(CHAT, problem=problem, concepts=CONCEPTS)
        yield id, "restyled", restyled
        restyled_solns = n_solns(CHAT, problem=restyled, n=3)
        yield id, "restyled solutions", restyled_solns
        restyled_tests = n_tests(CHAT, problem=restyled, n=3)
        yield id, "restyled tests", restyled_tests
        restyled_tests_split = split_tests(restyled_tests)
        for i, test in enumerate(restyled_tests_split):
            yield id, f"restyled tests split {i}", test


def log_and_save(g: Generator, filename: str) -> pd.DataFrame:
    assert filename.endswith(".csv"), f"Filename must end with .csv, got {filename}"

    rows = []
    for id, name, value in g:
        row = {
            "id": id,
            "name": name,
            "value": value,
        }
        pp(row)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename)
    return df


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
    CHAT = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
    CONCEPTS = [

    ]
    ts = util.timestamp()
    gen = generate_solns_and_tests(PROBLEMS)
    log_and_save(gen, filename="../datasets/prompts/prompts-{ts}.csv")

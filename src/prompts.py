"""
Collect prompts here
"""
import os
import sys
from pprint import pp
from typing import List, Generator, Union, Tuple, Optional, Iterator, Iterable
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
from langchain.callbacks import get_openai_callback
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
    inputs = set(kwargs.keys())
    expected_inputs = set(prompt["inputs"])
    assert inputs == expected_inputs, \
        f"Mismatched inputs: {inputs} vs {expected_inputs}"

    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]

    return run_prompt(
        chat,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        **kwargs,
    )


def make_problem(chat: ChatOpenAI) -> str:
    return run_saved_prompt(
        chat,
        key="new coverless",
    )


def restyle_problem(chat: ChatOpenAI, problem: str) -> str:
    return run_saved_prompt(
        chat,
        key="restyle as coverless",
        problem=problem,
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


def check_if_python(chat: ChatOpenAI, problem: str) -> str:
    return run_saved_prompt(
        chat,
        key="check if python",
        problem=problem,
    )


def rate_difficulty(chat: ChatOpenAI, problem: str) -> str:
    return run_saved_prompt(
        chat,
        key="rate difficulty",
        problem=problem,
    )


def gen_solns_and_tests(chat: ChatOpenAI, problems: Iterable[Tuple[int, str]]) -> Generator[
    Tuple[int, str, str], None, None]:
    for id, problem in problems:
        yield id, "original problem", problem

        restyled = restyle_problem(chat, problem=problem)
        yield id, "restyled problem", restyled

        restyled_solns = n_solns(chat, problem=restyled, n=3)
        yield id, "solutions", restyled_solns

        for i, soln in enumerate(util.split_py_markdown(restyled_solns)):
            yield id, f"solution {i}", soln

        restyled_tests = n_tests(chat, problem=restyled, n=3)
        yield id, "tests", restyled_tests

        for i, test in enumerate(util.split_tests(restyled_tests)):
            yield id, f"test {i}", test


def gen_solns_and_tests_dict(chat: ChatOpenAI, problems: List[dict]) -> Generator[dict, None, None]:
    problems = [(d["id"], d["text"]) for d in problems]
    with get_openai_callback() as cb:
        for id, key, val in gen_solns_and_tests(chat, problems):
            yield {
                "id": id,
                "key": key,
                "value": val,
                "cost": cb.total_cost,
            }


if __name__ == "__main__":
    PROBLEMS = [
        """
        Design a class called "Triangle" that represents a triangle.  The "Triangle" class should also have the following methods:
        - "__init__" - Initializes a new instance of the "Triangle" class with the given lengths of all sides.
        - "get_perimeter" - Returns the perimeter of the triangle, which is calculated by adding all three side lengths together.
        - "get_area" - Returns the area of the triangle, which is calculated using Heron's formula: 
        area = sqrt(s * (s - a) * (s - b) * (s - c)), where s is the semiperimeter of the triangle and a, b, and c are the lengths of the sides.
        """,
        """
        Write a function `multiplyArrays(arr1, arr2)` that takes in two arrays `arr1` and `arr2` of equal length, and returns an array `result` where each element `result[i]` is the product of `arr1[i]` and `arr2[i]`. 
        For example, given `arr1 = [2, 3, 4]` and `arr2 = [5, 6, 7]`, the function should return `[10, 18, 28]` since `10 = 2 * 5`, `18 = 3 * 6`, and `28 = 4 * 7`. 
        """,
        """
        Given a binary tree, validate if it is a binary search tree (BST).
        """,
        """
        Write a SQL query to find all the users in a database with age greater than 25.
        """,
        """
        Write a function in Java that takes an integer N as input and generates the modified Fibonacci sequence up to N numbers.
        """
    ]
    CHAT = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
    for problem in PROBLEMS:
        restyled = restyle_problem(CHAT, problem)
        print(
            problem,
            restyled,
            # restyled,
            # check_if_python(CHAT, problem=problem),
            # check_if_python(CHAT, problem=restyled),
            rate_difficulty(CHAT, problem=problem),
            rate_difficulty(CHAT, problem=restyled),
        )

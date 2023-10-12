"""
Collect prompts here
"""
import os
from typing import List, Generator, Union, Tuple, Optional, Iterator
import random
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


def run_chain(chat: ChatOpenAI, sys_prompt: str, user_prompt: str, **kwargs) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt),
    ])
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(**kwargs)


def cover_story_problem(chat: ChatOpenAI, stories: List[str], concepts: List[str]) -> str:
    s_stories = ", ".join([f"'{story}'" for story in random.choices(population=stories, k=3)])
    s_concepts = ", ".join([f"'{concept}'" for concept in random.choices(population=concepts, k=3)])
    prompt = """
You are an AI teaching assistant for a computer science department, where your job is to construct programming problems to teach students at varying levels of competency.  A programming problem consists of a "cover story", a "key concept", and a "specification".  The cover story motivates the problem; the key concept is the idea from computer science that the problem seeks to teach or test; and the specification gives guidelines about how solutions to the problem should be structured.  Propose a novel problem consisting of a cover story, a concept, and a specification.  

You MUST the following format; it is critical that the headings are indicated with three hashes (###), as your responses will be automatically parsed.

### Cover story
e.g. {stories}

### Concept
e.g. {concepts}

### Problem description
Specify the problem.  The problem should be motivated by the cover story.

### Specification
If the problem can be solved with a function, state the function's signature in Python, with type annotations.

If the problem uses classes and methods, state the class name and the methods that will be tested in Python, with type annotations.

### Example
An input-output example.
"""
    return run_chain(chat, sys_prompt=prompt, user_prompt="", stories=s_stories, concepts=s_concepts)


def evolve_with_hole(chat: ChatOpenAI, problem: str, hole_fill: str) -> str:
    prompt = """
You are an AI teaching assistant.  Produce a new, more {hole} programming problem, using the following problem as inspiration.  Make sure to keep the format the same!
    """
    return run_chain(chat, sys_prompt=prompt, user_prompt="{problem}", hole=hole_fill, problem=problem)


if __name__ == "__main__":
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
    FUN_ADJS = ['bubbly', 'zesty', 'giddy', 'wacky', 'cheeky', 'spunky',
                'frolicsome', 'sprightly', 'pizzazzy', 'snazzy']

    problems = []
    for i in range(1):
        problem = cover_story_problem(CHAT, stories=STORIES, concepts=CONCEPTS)
        print(problem)
        problems.append(problem)

    print("fun!")
    for adj in FUN_ADJS:
        print(f"##A {adj} extension:")
        print(evolve_with_hole(CHAT, problems[0], adj))

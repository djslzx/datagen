import os
import random

import openai
import numpy as np
import json
from typing import List
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


def fetch_dataset_item(i: int) -> str:
    item = CODE_DATASET[i]
    return f"instruction: {item['instruction']}\n" \
           f"input: {item['input']}\n" \
           f"output: {item['output']}" \


def make_system_prompt(method: str) -> str:
    return f"Please increase the difficulty of the given programming test question a bit. " \
           f"You can increase the difficulty using, but not limited to, the following methods: " \
           f"{method}"


# generate a small dataset using Evol-Instruct from the WizardLM paper
def evol_instruct(iters: int, seed_dataset: List[str], evol_methods: List[str]):
    for i in range(iters):
        for datum in seed_dataset:
            mutation_method = random.choice(evol_methods)
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": make_system_prompt(mutation_method)},
                    {"role": "user", "content": datum},
                ],
            )
            yield {
                "input": datum,
                "mutation_method": mutation_method,
                "output": completion.choices[0].message["content"],
            }


# generate a small dataset using Evol-Instruct + novelty pressure
def nov_instruct():
    raise NotImplementedError


if __name__ == "__main__":
    str_data = [fetch_dataset_item(i) for i in range(len(CODE_DATASET))]
    with open("../datasets/code_alpaca_tiny_evol.jsonl", "w") as f:
        for msg in evol_instruct(1, str_data, EVOL_METHODS):
            print(msg)
            f.write(json.dumps(msg, indent=4) + "\n")

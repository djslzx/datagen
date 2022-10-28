"""
Programmatically prompt Codex to generate data from a small starting dataset.
Examine the generated data's statistics to understand how far we can go with this idea.
"""

import os
import openai
from time import sleep

from cfg import PCFG
from lindenmayer import S0LSystem

openai.api_key = os.getenv("OPENAI_API_KEY")
CODE_MODEL = "code-davinci-002"
TEXT_MODEL = "text-davinci-002"
PROMPT = "codex/prompt.txt"


def prompt_text():
    with open(PROMPT, "r") as f:
        prompt = "".join(f.readlines())
    assert prompt, "Found empty prompt"
    prompt = prompt.replace('f', 'F')  # temporary measure
    return prompt


def prompt_axiom(length=20):
    g_newline = PCFG(start="axiom",
                     rules={
                         "axiom": [["op?", "f_expr"],
                                   ["X"]],
                         "f_expr": [["F"],
                                    ["f_expr", "op?", "F"]],
                         "op?": [["op"], PCFG.Empty],
                         "op": [["+"], ["-"]]
                     },
                     weights={
                         "axiom": [4, 1],
                         "f_expr": [3, 1],
                         "op?": [1, 2],
                         "op": [1, 1],
                     })
    newline = g_newline.iterate_until(length)
    print("(", "".join(newline), ")", sep="")
    newline = "".join([w for w in newline
                       if w not in g_newline.nonterminals
                       and w != PCFG.Eps]) + ";"
    return newline


def prompt_codex(n: int, max_tokens: int, temperature: float, model: str):
    assert model in {CODE_MODEL, TEXT_MODEL}
    if model == CODE_MODEL:
        outfile = "codex/code/completions.txt"
        logfile = "codex/code/completions.log"
    else:
        outfile = "codex/text/completions.txt"
        logfile = "codex/text/completions.log"

    with open(outfile, "a") as out, open(logfile, "a") as log:
        for i in range(n):
            text = prompt_text()
            axiom = ""  # prompt_axiom()
            response = openai.Completion.create(
                model=model,
                prompt=text + axiom,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=")",
            )
            if model == CODE_MODEL:
                sleep(3)
            else:
                sleep(1)

            text = axiom + response["choices"][0]["text"]
            print(model, text)
            out.write(f"{text}\n")
            log.write(f"{response}\n")


def render_codex_outputs(samples: int, filename: str):
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            try:
                s = S0LSystem.from_sentence(line.strip())
                print(s)
            except ValueError:
                continue

            for j in range(samples):
                print(f"Expanding {i}-th system, {j}-th sample")
                depth, word = s.expand_until(length=500)
                S0LSystem.to_svg(word, d=5, theta=43,
                                 filename=f"codex/renders/text/codex{i:02d}-{j}")


if __name__ == '__main__':
    # prompt = get_prompt()
    # print(prompt)
    # for i in range(100):
    #     print(prompt_axiom())

    temp = 0.9
    prompt_codex(n=100, max_tokens=100, temperature=temp, model=TEXT_MODEL)
    prompt_codex(n=100, max_tokens=100, temperature=temp, model=CODE_MODEL)
    # render_codex_outputs(samples=3)

"""
Programmatically prompt Codex to generate data from a small starting dataset.
Examine the generated data's statistics to understand how far we can go with this idea.
"""

import os
import openai
from time import sleep

from cfg import PCFG
from lindenmayer import S0LSystem
from book_zoo import zoo

openai.api_key = os.getenv("OPENAI_API_KEY")
CODE_MODEL = "code-davinci-002"
TEXT_MODEL = "text-davinci-002"


def make_prompt() -> str:
    systems = [specimen for specimen, angle in zoo]
    sentences = ["".join(system.to_sentence()) for system in systems]
    prompt = "\n".join(sentences) + "\n"
    prompt = prompt.replace('f', 'F')  # temporary measure
    return prompt


def prompt_codex(n: int, max_tokens: int, temperature: float, model: str):
    assert model in {CODE_MODEL, TEXT_MODEL}
    if model == CODE_MODEL:
        outfile = "../out/codex-samples/code/completions.txt"
        logfile = "../out/codex-samples/code/completions.log"
    else:
        outfile = "../out/codex-samples/text/completions.txt"
        logfile = "../out/codex-samples/text/completions.log"

    with open(outfile, "a") as out, open(logfile, "a") as log:
        for i in range(n):
            text = make_prompt()
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
            print(f"{model}: {text}")
            out.write(f"{text}\n")
            log.write(f"{response}\n")


def render_codex_outputs(samples: int, filename: str, out_dir: str):
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            try:
                s = S0LSystem.from_sentence([*line.strip()])
                print(s)
            except ValueError:
                continue

            for j in range(samples):
                print(f"Expanding {i}-th system, {j}-th sample")
                depth, word = s.expand_until(length=500)
                try:
                    S0LSystem.to_svg(word, d=5, theta=43,
                                     filename=f"{out_dir}/{i:02d}-{j}")
                except IndexError:
                    pass


if __name__ == '__main__':
    # prompt = get_prompt()
    # print(prompt)
    # for i in range(100):
    #     print(prompt_axiom())

    # temp = 0.9
    # prompt_codex(n=100, max_tokens=100, temperature=temp, model=TEXT_MODEL)
    # prompt_codex(n=100, max_tokens=100, temperature=temp, model=CODE_MODEL)
    render_codex_outputs(filename="../out/codex-samples/code/completions.txt",
                         out_dir="../out/codex-samples/code/renders/",
                         samples=3)
    render_codex_outputs(filename="../out/codex-samples/text/completions.txt",
                         out_dir="../out/codex-samples/text/renders/",
                         samples=3)

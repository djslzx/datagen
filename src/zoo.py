"""
Breed together interesting L-system specimens
"""
import sys
import random
from typing import List, Tuple
from lindenmayer import S0LSystem
import book_zoo
import codex_zoo


def mutate_succ(succ: str, rate=0.2) -> str:
    transform = {
        '+': lambda: '-',
        '-': lambda: '+',
        'F': lambda: random.choice(['FF', '+F', '-F']),
    }
    out = ""
    for c in succ:
        if c in transform and random.random() < rate:
            out += transform[c]()
        else:
            out += c
    return out


def mate(a: S0LSystem, b: S0LSystem) -> S0LSystem:
    """Mix two S0L-systems together randomly"""

    # merge random rules together to get longer rules
    # chop random rules to get shorter rules
    # mutate + <-> -
    # ensure that all non-drawing tokens are used as nonterminals?

    # take all rules from both parents
    keys = a.productions.keys() | b.productions.keys()
    rules = {key: [] for key in keys}
    weights = {key: [] for key in keys}
    for s in [a, b]:
        for key in s.productions.keys():
            rules[key] += s.productions[key]
            weights[key] += s.distribution[key]

    # randomly drop rules
    n_keep = random.randint(1, len(rules))
    rules = random.choices(rules, k=n_keep)

    # randomly mutate rules (non-brackets)
    n_rule_mutations = random.randint(0, len(rules))
    for succs in random.choices(rules.values(),
                                k=n_rule_mutations):
        n_succ_mutations = random.randint(0, len(succs))
        for succ in random.choices(succs, k=n_succ_mutations):
            succ = mutate_succ(succ)
    
    # randomly merge/split rules
    
    print("rules:", rules)
    pass


def view_zoo(zoo: List[Tuple[S0LSystem, float]],
             n_specimens: int, expand_length: int, outdir: str):
    for i, (system, theta) in enumerate(zoo):
        # print(f"Handling system {system}...")
        words = [system.expand_until(expand_length)
                 for _ in range(n_specimens)]
        for j, (depth, word) in enumerate(words):
            print(f"Rendering {i}-th word of length {len(word)}...")
            S0LSystem.render(
                word, d=5, theta=theta,
                filename=f"{outdir}/system{i}-render{j}@depth{depth}",
            )


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: zoo.py DIR")
        sys.exit(1)
    OUTDIR = sys.argv[1]
    # view_zoo(zoo=ZOO, n_specimens=1, expand_length=1000, outdir=OUTDIR)

    # check if codex is copying the dataset
    # for i, (g, _) in enumerate(codex_zoo.zoo):
    #     for j, (h, _) in enumerate(book_zoo.zoo):
    #         if g.productions == h.productions:
    #             print(f"Found a copy at position {i}, {j}: {g}")

    view_zoo(
        zoo=book_zoo.zoo,
        n_specimens=1,
        expand_length=1000,
        outdir=f"{OUTDIR}/book/"
    )

    view_zoo(
        zoo=codex_zoo.zoo,
        n_specimens=1,
        expand_length=1000,
        outdir=f"{OUTDIR}/codex/"
    )

    view_zoo(
        zoo=codex_zoo._prompt_zoo,
        n_specimens=1,
        expand_length=1000,
        outdir=f"{OUTDIR}/prompt/"
    )

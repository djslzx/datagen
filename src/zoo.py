"""
Breed together interesting L-system specimens
"""
import sys
import random
from typing import List, Tuple
from lindenmayer import S0LSystem

# List of specimens and their angles
ZOO: List[Tuple[S0LSystem, float]] = [
    # Deterministic L-systems
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+FF-FF-F-F+F+F"
                  "F-F-F+F+FF+FF-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="-F",
        productions={
            "F": ["F+F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F+F+F+F",
        productions={
            "F": ["F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF"],
            "f": ["fffff"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["FF-F-F-F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["FF-F-F-F-FF"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["FF-F+F-F-FF"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["FF-F--F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-FF--F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F+F-F-F"]
        },
        distribution="uniform",
    ), 90),
    # Left- right- rules
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F+F+",
                  "-F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F+F+F",
                  "F-F-F"]
        },
        distribution="uniform",
    ), 60),
    # Rewriting techniques
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F+F++F-F--FF-F+",
                  "-F+FF++F+F--F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="-F",
        productions={
            "F": ["FF-F-F+F+F-F-FF""+F+FFF-F+F+FF+F-FF-F-F+F+FF",
                  "+FF-F-F+F+FF+FFF-F-F+FFF-F-FF+F+F-F-F+F+FF"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="-L",
        productions={
            "L": ["LF+RFR+FL-F-LFLFL-FRFR+"],
            "R": ["-LFLF+RFRFR+F+RF-LFL-FR"],
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="-L",
        productions={
            "L": ["LFLF+RFR+FLFL-FRF-LFL-FR+F+RF-LFL-FRFRFR+"],
            "R": ["-LFLFLF+RFR+FL-F-LF+RFR+FLF+RFRF-LFL-FRFR"],
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="L",
        productions={
            "L": ["LFRFL-F-RFLFR+F+LFRFL"],
            "R": ["RFLFR+F+LFRFL-F-RFLFR"]
        },
        distribution="uniform",
    ), 90),
    # Branching w/ edge rewriting
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F[+F]F[-F]F"]
        },
        distribution="uniform",
    ), 25.7),
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F[+F]F[-F][F]"]
        },
        distribution="uniform",
    ), 20),
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["FF-[-F+F+F]+[+F-F-F]"]
        },
        distribution="uniform",
    ), 22.5),
    # Branching with node rewriting
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["F[+X]F[-X]+X"],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 20),
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["F[+X][-X]FX"],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 25.7),
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["F-[[X]+X]+F[+FX]-X"],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 22.5),
    # Stochastic branching
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["F[+X]FX",
                  "F[-X]FX",
                  "F[+X][-X]FX",
                  "FFX",
                  ],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 20),
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F[+F]",
                  "F[-F]",
                  "FF"],
        },
        distribution="uniform",
    ), 20),
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["[+FX][-X]FX",
                  "[+X][-FX]FX",
                  "[+FX][-FX]FX"],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 20),
    # (S0LSystem(
    #     axiom="",
    #     productions={
    #         "F": [""]
    #     },
    #     distribution="uniform",
    # ), 90),
]


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
        print(f"Handling system {system}...")
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
    
    # add to zoo
    

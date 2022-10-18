import random
import pickle
import argparse
import multiprocess as mp
from pprint import pp
from typing import List
import pdb

from lindenmayer import LSystem, S0LSystem
from cfg import CFG, PCFG
from inout import io, autograd_io
import util
import book_zoo

GENERAL_MG = PCFG(
    start="L-SYSTEM",
    weights="uniform",
    rules={
        "L-SYSTEM": [
            ["AXIOM", ";", "RULES"],
        ],
        "AXIOM": [
            ["NT", "AXIOM"],
            ["T", "AXIOM"],
            ["NT"],
            ["T"],
        ],
        "RULES": [
            ["RULE", ",", "RULES"],
            ["RULE"],
        ],
        "RULE": [
            ["LHS", "~", "RHS"],
        ],
        "LHS": [
            ["NT"],
        ],
        "RHS": [
            ["[", "RHS", "]", "RHS"],
            ["NT", "RHS"],
            ["T", "RHS"],
            ["[", "RHS", "]"],
            ["NT"],
            ["T"],
        ],
        "NT": [
            ["F"],
            ["f"],
            ["X"],
        ],
        "T": [
            ["+"],
            ["-"],
        ],
    },
)

HANDCODED_MG = PCFG(
    start="L-SYSTEM",
    rules={
        "L-SYSTEM": [
            ["AXIOM", ";", "RULES"],
        ],
        "AXIOM": [
            ["PLUSES", "F", "AXIOM"],
            ["MINUSES", "F", "AXIOM"],
            ["F", "AXIOM"],
            ["PLUSES", "F"],
            ["MINUSES", "F"],
            ["F"],
        ],
        "RULES": [
            ["RULE", ",", "RULES"],
            ["RULE"],
        ],
        "RULE": [
            ["LHS", "~", "RHS"],
        ],
        "LHS": [
            ["F"],
        ],
        "RHS": [
            ["F", "[", "PLUSES", "F", "INNER", "]", "RHS", "F"],
            ["F", "[", "MINUSES", "F", "INNER", "]", "RHS", "F"],
            ["F", "INNER"],
        ],
        "INNER": [
            ["INNER", "PLUSES", "FS"],
            ["INNER", "MINUSES", "FS"],
            ["FS"],
        ],
        "PLUSES": [
            ["+", "PLUSES"],
            ["+"],
        ],
        "MINUSES": [
            ["-", "MINUSES"],
            ["-"],
        ],
        "FS": [
            ["FS", "F"],
            ["F"],
        ],
    },
    weights={
        "L-SYSTEM": [1],
        "AXIOM": [1],
        "M": [0.25, 0.25, 0.5],
        "RULES": [0.5, 0.5],
        "RULE": [1],
        "LHS": [1],
        "RHS": [0.25, 0.25, 0.5],
        "INNER": [0.25, 0.25, 0.5],
        "PLUSES": [0.5, 0.5],
        "MINUSES": [0.5, 0.5],
        "FS": [0.5, 0.5],
    }
)

METAGRAMMARS = {
    "general": GENERAL_MG,
    "handcoded": HANDCODED_MG,
}


def S0LSystem_from_CFG(metagrammar: PCFG,
                       n_rules: int,
                       max_axiom_length: int,
                       max_rule_length: int,
                       levels=1) -> S0LSystem:
    """
    Generates a random stochastic context-free L-system, using a PCFG.
    """
    mg = metagrammar
    for i in range(levels):
        mg = mg.to_bigram()
    gs = mg.iterate_fully()
    return S0LSystem.from_str(gs)


def make_grammar(args, index: int,
                 bigram_level=1, fit=True, log=True) -> S0LSystem:
    # fit metagrammar to zoo
    mg = METAGRAMMARS[args.generator]
    mg_new = autograd_io(mg, [sys.to_sentence() for sys, _ in book_zoo.zoo])
    g = S0LSystem_from_CFG(
        mg_new,
        n_rules=random.randint(*args.n_rules),
        max_axiom_length=random.randint(*args.axiom_length),
        max_rule_length=random.randint(*args.rule_length),
    )
    print(f"[{index}] {g}")

    if log:
        # save grammar text representation to text file for debug
        with open(f'{args.out_dir}/grammars.log', 'a') as f:
            f.write(f'Grammar {index}:\n{g}\n')

    # save grammar object for model training
    with open(f'{args.out_dir}/grammars.dat', 'ab') as f:
        pickle.dump((index, g), f)

    return g


def make_word(args, grammar: S0LSystem, grammar_i: int, specimen_i: int,
              angle: float, verbose=False, log=True) -> str:
    word = grammar.nth_expansion(args.devel_depth)

    if verbose:
        word_preview = word[:20] + ("..." if len(word) > 20 else "")
        print(f"[{grammar_i}, {specimen_i}] "
              f"Generated word {word_preview} of length {len(word)}")

    if log:
        # write debug text
        with open(f'{args.out_dir}/words.log', 'a') as f:
            f.write(f'[{grammar_i}, {specimen_i}, {angle}]: {word}\n')

    # write object
    with open(f'{args.out_dir}/words.dat', 'ab') as f:
        pickle.dump((grammar_i, specimen_i, angle, word), f)

    return grammar_i, specimen_i, word, angle


def make_sticks(args, grammar: int, specimen: int, word: str, angle: float,
                verbose=False, log=True, render_to_svg=False):
    if verbose:
        word_preview = word[:20] + ("..." if len(word) > 20 else "")
        print(f"[{grammar}, {specimen}] "
              f"Rendering {word_preview} of length {len(word)}")

    sticks = S0LSystem.to_sticks(
        word,
        d=10,
        theta=angle,
    )

    if log:
        with open(f'{args.out_dir}/sticks.log', 'a') as f:
            s = "\n  ".join(str(stick) for stick in sticks)
            f.write(f'{grammar, specimen, angle}: [\n  {s}\n]\n')

    with open(f'{args.out_dir}/sticks.dat', 'ab') as f:
        pickle.dump((grammar, specimen, angle, sticks), f)

    if render_to_svg:
        LSystem.to_svg(
            sticks,
            f'{args.out_dir}/render[{grammar},{specimen}]@{angle}deg.svg'
        )


def make_turtle_render(args, id: str, word: str, angle: float, verbose=False):
    if verbose:
        word_preview = word[:20] + ("..." if len(word) > 20 else "")
        print(f"{id} Rendering {word_preview} of length {len(word)}")

    LSystem.render_with_turtle(
        word,
        d=10,
        theta=angle,
        filename=f'{args.out_dir}/render{id}{angle}deg{args.devel_depth}depth'
    )


def make():
    p = argparse.ArgumentParser(description="Generate L-systems.")
    p.add_argument('generator', type=str, choices=list(METAGRAMMARS.keys()),
                   help="Which L-system metagrammar should be used")
    p.add_argument('n_grammars', type=int,
                   help='The number of grammars to make')
    p.add_argument('axiom_length', type=int, nargs=2,
                   help="The min and max length of a grammar's axiom")
    p.add_argument('n_rules', type=int, nargs=2,
                   help='The min and max number of rules per grammar')
    p.add_argument('rule_length', type=int, nargs=2,
                   help="The min and max length of a rule's successor")
    p.add_argument('n_specimens', type=int,
                   help='The number of samples to take from each grammar')
    p.add_argument('devel_depth', type=int,
                   help="The max size of a specimen's output string")
    p.add_argument('out_dir', type=str,
                   help="Where output files should be stored")
    args = p.parse_args()
    angles = [random.randint(10, 40),
              random.randint(40, 60),
              random.randint(60, 90)]

    with mp.Pool() as pool:
        print("Making grammars...")
        grammars = pool.imap(lambda n: make_grammar(args, n),
                             range(args.n_grammars))

        print("Making endpoint words...")
        words = pool.starmap(
            lambda grammar, i, j, theta:
            make_word(args, grammar, i, j, theta, verbose=True),
            ((grammar, i, j, theta)
             for i, grammar in enumerate(grammars)
             for j in range(args.n_specimens)
             for theta in angles)
        )

        print("Rendering endpoint words to sticks...")
        pool.starmap(
            lambda grammar, specimen, word, angle:
            make_sticks(args, grammar, specimen, word, angle,
                        verbose=True,
                        render_to_svg=True),
            words
        )

        # print("Rendering endpoint words to images...")
        # pool.starmap(
        #     lambda grammar, specimen, word, angle:
        #     make_render(args, f'[{grammar},{specimen}]', word, angle,
        #                 verbose=True),
        #     words
        # )


def check_io(n_samples: int, zoo_limit=None):
    specimens = [sys for sys, angle in book_zoo.zoo[:zoo_limit]]

    for i, specimen in enumerate(specimens):
        d, s = specimen.expand_until(1000)
        S0LSystem.render(s, d=5, theta=43,
                         filename=f"../out/samples/reference_{i}")

    corpus = [s.to_sentence() for s in specimens]
    mg = GENERAL_MG.to_CNF(debug=False)
    tuned_mg = io(mg, corpus, debug=False, log=False)
    print("Finished tuning")

    untuned, tuned = [], []
    for i in range(n_samples):
        print(f"Sampling {i+1}/{n_samples}...")
        try:
            untuned.append(S0LSystem.from_sentence(mg.iterate_until(100)))
        except ValueError:
            pass
        try:
            tuned.append(S0LSystem.from_sentence(tuned_mg.iterate_fully()))
        except ValueError:
            pass

    for name, sys in zip(["untuned", "tuned"], [untuned, tuned]):
        print(f"Rendering from {name}")
        for i, sol in enumerate(sys):
            print(sol)
            d, s = sol.expand_until(1000)
            S0LSystem.render(s, d=5, theta=43,
                             filename=f"../out/samples/{name}_{i}")


def check_io_autograd(io_iters: int, n_samples: int, zoo_limit=None):
    def sample(i: int, G: PCFG):
        print(G)
        for j in range(n_samples):
            try:
                lsys = S0LSystem.from_sentence(G.iterate_until(100))
                print(lsys)
                d, s = lsys.expand_until(1000)
                S0LSystem.render(s, d=5, theta=43,
                                 filename=f"../out/samples/{i},{j}")
            except ValueError:
                pass

    corpus = [sys.to_sentence() for sys, angle in book_zoo.zoo[:zoo_limit]]
    mg = GENERAL_MG.to_CNF(debug=False)
    autograd_io(mg, corpus, iters=io_iters, log=True, callback=sample)


if __name__ == '__main__':
    # make()
    check_io(n_samples=10, zoo_limit=1)
    # check_io_autograd(io_iters=100, n_samples=10, zoo_limit=1)

import sys
import random
from typing import List, Tuple
import itertools as it
from lindenmayer import SOLSystem, CFG
import util


def random_grammar(alphabet: List[str],
                   n_rules_cap: int,
                   n_successors_cap: int,
                   letter_range: Tuple[int, int]):
    """Generate a bracketed stochastic L-system with axiom F."""

    # generate production rules
    n_productions = random.randint(1, n_rules_cap)
    productions = {}
    for _ in range(n_productions):
        # choose predecessor
        # TODO: don't hard-code, allow intermediate vars
        predecessor = "F"

        # choose successors
        n_successors = random.randint(1, n_successors_cap)
        successors = [random_successor(alphabet, letter_range)
                      for j in range(n_successors)]

        # add to productions
        productions[predecessor] = successors

    # generate production rule probabilities
    distribution = {}
    for predecessor, successors in productions.items():
        distribution[predecessor] = util.uniform_vec(len(successors))

    return SOLSystem(
        axiom="F",
        productions=productions,
        distribution=distribution
    )


def random_successor(alphabet: List[str],
                     letter_range: Tuple[int, int]) -> str:
    """
    Generate a random successor by making a random balanced bracket
    permutation, then inserting letters from the alphabet
    """
    n_letters = int(random.randint(*letter_range))  # not including brackets
    max_bracket_pairs = int(n_letters * 0.2)
    n_bracket_pairs = \
        0 if max_bracket_pairs == 0 \
        else random.randint(0, max_bracket_pairs)

    # randomly insert brackets
    succ = util.random_balanced_brackets(n_bracket_pairs)

    # randomly insert other letters
    for _ in range(n_letters):
        pos = random.randint(0, len(succ) - 1)
        letter = random.choice(alphabet)
        succ.insert(pos, letter)

    # remove empty braces and convert to string
    return "".join(succ).replace('[]', '')


def SOLSystem_from_SOLSystem(n_rules: int, n_expands: int) -> SOLSystem:
    """
    Generates the rules of a stochastic OL-System using a stochastic OL-system.
    """
    metagrammar = SOLSystem(
        axiom="F",
        productions={
            "F": ['+F', '-F', '[F]', 'FF']
        },
        distribution='uniform',
    )
    rules = [metagrammar.nth_expansion(n_expands) for i in range(n_rules)]
    return SOLSystem(
        axiom="F",
        productions={
            "F": rules,
        },
        distribution={
            "F": util.uniform_vec(n_rules)
        }
    )


def SOLSystem_from_CFG(n_rules: int, max_rule_length: int) -> SOLSystem:
    """
    Generates a random stochastic context-free L-system, using a CFG.
    """
    # TODO: rollout to multiple levels and add probabilities
    # by learning from a preliminary dataset (or just hard-code for now)
    metagrammar = CFG(rules={
        "T": [
            ["+"],
            ["-"],
        ],
        "NT": [
            ["F"]
        ],
        "RHS": [
            ["[", "B", "]"],
            ["RHS", "NT"],
            ["RHS", "T"],
        ],
        "B": [
            ["B", "NT"],
            ["B", "T"],
        ],
    })

    rules = []
    for _ in range(n_rules):
        start = ["RHS"]
        fixpt = metagrammar.iterate_until(start, length=max_rule_length)
        rules.append(metagrammar.to_str(fixpt))

    return SOLSystem(
        axiom="F",
        productions={
            "F": rules
        },
        distribution={
            "F": util.uniform_vec(n_rules)
        }
    )


def save_random_sol(
        grammar: SOLSystem,
        name: str,
        n_specimens: int,
        devel_length: int,
        render_development: bool,
        save_path: str,
):
    angles = [
        15, 30, 45, 60, 75, 90,
        # 13, 17, 19, 23, 29,
        # 31, 37, 41, 43, 47,
        # 53, 59, 61, 67, 71,
    ]

    # save random grammar
    print(grammar)
    with open(f'{save_path}/{name}-grammar.txt', 'w') as f:
        f.write(f'Grammar: {grammar}')

    # render specimens
    for specimen, angle in it.product(range(n_specimens), angles):

        if render_development:
            # render entire development sequence
            for level, word in enumerate(grammar.expansions(devel_length)):
                print(word)
                grammar.render(
                    word,
                    length=10,
                    angle=angle,
                    filename=f'{save_path}/{name}-{angle}deg' +
                    f'-{specimen}-lvl{level:02d}'
                )
        else:
            # take a snapshot of the last development stage
            word = grammar.nth_expansion(devel_length)
            print(word)
            grammar.render(
                word,
                length=10,
                angle=angle,
                filename=f'{save_path}/{name}-{angle}deg' +
                f'-{specimen}-lvl{devel_length:02d}'
            )


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print("Usage: datagen.py PROJECT_DIR")
        sys.exit(1)

    out_dir = sys.argv[1]
    N_GRAMMARS = 10
    for i in range(N_GRAMMARS):
        g = SOLSystem_from_CFG(
            n_rules=random.randint(2, 5),
            max_rule_length=10,
        )
        print(g)
        save_random_sol(
            grammar=g,
            name=f'grammar{i}',
            n_specimens=3,
            devel_length=5,
            render_development=False,
            save_path=out_dir,
        )

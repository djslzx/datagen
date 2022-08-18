import random
from typing import List, Tuple
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
    for i in range(n_productions):
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
    n_letters = int(random.randint(*letter_range))  # not including brackets
    max_bracket_pairs = int(n_letters * 0.2)
    n_bracket_pairs = \
        0 if max_bracket_pairs == 0 \
        else random.randint(0, max_bracket_pairs)

    # randomly insert brackets
    succ = util.random_balanced_brackets(n_bracket_pairs)

    # randomly insert other letters
    for i in range(n_letters):
        pos = random.randint(0, len(succ) - 1)
        letter = random.choice(alphabet)
        succ.insert(pos, letter)

    # remove empty braces and convert to string
    return "".join(succ).replace('[]', '')


def meta_SOL_SOLSystem(n_rules: int, n_expands: int) -> SOLSystem:
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


def meta_CFG_SOLSystem() -> SOLSystem:
    """
    Yields a random stochastic context-free L-system.
    """
    g = CFG(rules={
        "rules": [["F -> ", "rule", "\n", "rules"],
                  ["rule"]],
        "rule": [["NT", "rhs"]],
        "NT": [["F"]],
        "rhs": [["[", "rhs", "]"],
                ["rhs", "NT"],
                ["rhs", "T"]],
        "T": [["+"], ["-"], ["f"]]
    })
    start = ["rules"]
    fp = g.fixpoint(start, max_iters=100)
    s = g.to_str(fp)
    assert util.parens_are_balanced(s), "Expression has unbalanced parentheses"
    print(s)


def save_random_sol(
        name: str,
        min_letters: int,
        max_letters: int,
        n_expands: int,
        n_samples: int,
        n_levels: int
):
    with open(f'../imgs/{name}-grammar.txt', 'w') as f:
        # make and log random grammar
        g = meta_SOL_SOLSystem(
            n_rules=random.randint(1, 5),
            n_expands=n_expands,
        )
        # g = random_system.random_grammar(
        #     alphabet=['F', '+', '-'],
        #     n_rules_cap=3,
        #     n_successors_cap=4,
        #     letter_range=(min_letters, max_letters),
        # )
        # angle = 90 // random.randint(1, 10)
        angle = random.choice([13, 17, 19, 23, 29,
                               31, 37, 41, 43, 47,
                               53, 59, 61, 67, 71,
                               73, 79, 83, 89])
        print(angle, g)
        f.write(f'Angle: {angle}\n')
        f.write(f'Grammar: {g}')

        # render popn and save grammar
        for i in range(n_samples):
            for level, word in enumerate(g.expansions(n_levels)):
                word = g.nth_expansion(level)
                print(word)
                g.render(
                    word,
                    length=10,
                    angle=angle,
                    filename=f'../imgs/{name}-{angle}deg-{i}-lvl{level:02d}'
                )


if __name__ == '__main__':
    n_grammars = 10
    for i in range(n_grammars):
        save_random_sol(
            name=f'grammar{i}',
            min_letters=1,
            max_letters=6,
            n_samples=10,
            n_levels=4,
            n_expands=7,
        )

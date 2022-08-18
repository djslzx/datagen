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


def meta_CFG_SOLSystem(n_rules: int, max_rule_length: int) -> SOLSystem:
    """
    Generates a random stochastic context-free L-system, using a CFG.
    """
    metagrammar = CFG(rules={
        "T": [["+"], ["-"]],
        "NT": [["F"]],
        "RHS": [["[", "RHS", "]"],
                ["RHS", "NT"],
                ["RHS", "T"]],
    })
    rules = []
    for i in range(n_rules):
        start = ["RHS"]
        fp = metagrammar.iterate_until(start, length=max_rule_length)
        s = metagrammar.to_str(fp)
        rules.append(s)
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
        name: str,
        max_rule_length: int,
        n_specimens: int,
        development_depth: int,
        render_development: bool,
):
    # make a new grammar
    g = meta_CFG_SOLSystem(
        n_rules=random.randint(2, 5),
        max_rule_length=max_rule_length,
    )

    # choose a prime angle
    angle = random.choice([
        13, 17, 19, 23, 29,
        31, 37, 41, 43, 47,
        # 53, 59, 61, 67, 71,
    ])

    print(angle, g)
    with open(f'../imgs/{name}-{angle}deg-grammar.txt', 'w') as f:
        # save random grammar
        f.write(f'Angle: {angle}\n')
        f.write(f'Grammar: {g}')

    # render specimens
    for i in range(n_specimens):
        # render entire development sequence
        if render_development:
            for level, word in enumerate(g.expansions(development_depth)):
                print(word)
                g.render(
                    word,
                    length=10,
                    angle=angle,
                    filename=f'../imgs/{name}-{angle}deg' +
                    f'-{i}-lvl{level:02d}'
                )
        else:
            # take a snapshot of the last development stage
            word = g.nth_expansion(development_depth)
            print(word)
            g.render(
                word,
                length=10,
                angle=angle,
                filename=f'../imgs/{name}-{angle}deg' +
                f'-{i}-lvl{development_depth:02d}'
            )


if __name__ == '__main__':
    n_grammars = 20
    for i in range(n_grammars):
        save_random_sol(
            name=f'grammar{i}',
            n_specimens=3,
            development_depth=4,
            max_rule_length=20,
            render_development=False,
        )

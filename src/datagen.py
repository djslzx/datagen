import random
import pickle
import argparse
import multiprocess as mp
from lindenmayer import S0LSystem, CFG, LSystem
import util


def S0LSystem_from_S0LSystem(n_rules: int, n_expands: int) -> S0LSystem:
    """
    Generates the rules of a stochastic OL-System using a stochastic OL-system.
    """
    metagrammar = S0LSystem(
        axiom="F",
        productions={
            "F": ['+F', '-F', '[F]', 'FF']
        },
        distribution='uniform',
    )
    rules = [metagrammar.nth_expansion(n_expands) for i in range(n_rules)]
    return S0LSystem(
        axiom="F",
        productions={
            "F": rules,
        },
        distribution={
            "F": util.uniform_vec(n_rules)
        }
    )


def S0LSystem_from_CFG(metagrammar: CFG,
                       n_rules: int,
                       max_axiom_length: int,
                       max_rule_length: int) -> S0LSystem:
    """
    Generates a random stochastic context-free L-system, using a CFG.
    """
    # TODO: rollout to multiple levels and add probabilities
    # by learning from a preliminary dataset (or just hard-code for now)
    rules = []
    for _ in range(n_rules):
        fixpt = metagrammar.iterate_until(["RHS"], length=max_rule_length)
        rules.append(fixpt)

    axiom = metagrammar.iterate_until(["AXIOM"], length=max_axiom_length)
    if "F" not in axiom:
        axiom = "F"

    return S0LSystem(
        axiom=axiom,
        productions={
            "F": rules,
        },
        distribution="uniform",
    )


def constrained_random_S0LSystem(n_rules: int,
                                 max_axiom_length: int,
                                 max_rule_length: int) -> S0LSystem:
    metagrammar = CFG(rules={
        "AXIOM": [
            ["AXIOM", "NT"],
            ["AXIOM", "T"],
        ],
        "T": [
            ["+F"],
            ["-F"],
        ],
        "NT": [
            ["F"],
        ],
        "RHS": [
            ["[", "B", "NT", "]"],
            ["[", "B", "T", "]"],
            ["RHS", "NT"],
            ["RHS", "T"],
        ],
        "B": [
            ["B", "NT"],
            ["B", "T"],
        ],
    })
    return S0LSystem_from_CFG(metagrammar,
                              n_rules,
                              max_axiom_length,
                              max_rule_length)


def general_random_S0LSystem(n_rules: int,
                             max_axiom_length: int,
                             max_rule_length: int) -> S0LSystem:
    metagrammar = CFG(rules={
        "AXIOM": [
            ["AXIOM", "NT"],
            ["AXIOM", "T"],
        ],
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
    return S0LSystem_from_CFG(metagrammar,
                              n_rules,
                              max_axiom_length,
                              max_rule_length)


def make_grammar(args, index: int, log=True) -> S0LSystem:
    # g = general_random_S0LSystem(
    #     n_rules=random.randint(2, 5),
    #     max_axiom_length=6,
    #     max_rule_length=10,
    # )

    g = constrained_random_S0LSystem(
        n_rules=random.randint(*args.n_rules),
        max_axiom_length=random.randint(*args.axiom_length),
        max_rule_length=random.randint(*args.rule_length),
    )

    print(f"[{index}] {g}")

    if log:
        # save grammar text representation to text file for debug
        with open(f'{args.out_dir}/grammars.log', 'a') as f:
            f.write(f'Grammar {index}: {g}\n')

    # save grammar object for model training
    with open(f'{args.out_dir}/grammars.dat', 'ab') as f:
        pickle.dump((index, g), f)

    return g


def make_word(args, grammar: S0LSystem, grammar_i: int, specimen_i: int,
              angle: float, verbose=False, log=True) -> str:
    word = grammar.nth_expansion(args.devel_depth)
    # depth, word = grammar.expand_until(1000)

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
                verbose=False, log=True):
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

    # LSystem.to_svg(
    #     sticks,
    #     f'{args.out_dir}/render[{grammar},{specimen}]@{angle}deg.svg'
    # )


def make_render(args, id: str, word: str, angle: float, verbose=False):
    if verbose:
        word_preview = word[:20] + ("..." if len(word) > 20 else "")
        print(f"{id} Rendering {word_preview} of length {len(word)}")

    LSystem.render_with_turtle(
        word,
        d=10,
        theta=angle,
        filename=f'{args.out_dir}/render{id}{angle}deg'
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Generate L-systems.")
    p.add_argument('n_grammars', type=int,
                   help='The number of grammars to make')
    p.add_argument('axiom_length', type=int, nargs=2,
                   help="The min and max length of a grammar's axiom")
    p.add_argument('n_rules', type=int, nargs=2,
                   help='The min and max number of rules per grammars')
    p.add_argument('rule_length', type=int, nargs=2,
                   help="The min and max length of a rule's successor")
    p.add_argument('n_specimens', type=int,
                   help='The number of samples to take from each grammar')
    p.add_argument('devel_depth', type=int,
                   help="The length of each specimen's development sequence")
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
            make_sticks(args, grammar, specimen, word, angle, verbose=True),
            words
        )

        print("Rendering endpoint words to images...")
        pool.starmap(
            lambda grammar, specimen, word, angle:
            make_render(args, f'[{grammar},{specimen}]', word, angle,
                        verbose=True),
            words
        )

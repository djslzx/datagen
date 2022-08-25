import random
import pickle
import argparse
import multiprocess as mp
from lindenmayer import S0LSystem, CFG
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
        distribution={
            "F": util.uniform_vec(n_rules),
        }
    )


def constrained_random_S0LSystem(n_rules: int,
                                 max_axiom_length: int,
                                 max_rule_length: int) -> S0LSystem:
    metagrammar = CFG(rules={
        "AXIOM": [
            ["F"],
            ["AXIOM", "+F"],
            ["AXIOM", "-F"],
        ],
        "T": [
            ["+F"],
            ["-F"],
        ],
        "NT": [
            ["F"],
            # ["f"],
        ],
        "RHS": [
            ["[", "T", "B", "]"],
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
    angles = [random.randint(10, 90) for _ in range(5)]

    def make_grammar(index: int, log=True) -> S0LSystem:
        g = constrained_random_S0LSystem(
            n_rules=random.randint(*args.n_rules),
            max_axiom_length=random.randint(*args.axiom_length),
            max_rule_length=random.randint(*args.rule_length),
        )
        print(f"[{index}] {g}")

        if log:
            # save grammar text representation to text file for debug
            with open(f'{args.out_dir}/grammars.log', 'a') as f:
                f.write(f'Grammar {index}: {g}')

        # save grammar object for model training
        with open(f'{args.out_dir}/grammars.dat', 'ab') as f:
            pickle.dump((index, g), f)

        return g

    def make_word(grammar: S0LSystem, grammar_i: int, specimen_i: int,
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

    def make_render(grammar: int, specimen: int, word: str, angle: float,
                    verbose=False):
        if verbose:
            word_preview = word[:20] + ("..." if len(word) > 20 else "")
            print(f"[{grammar}, {specimen}] "
                  f"Rendering {word_preview} of length {len(word)}")

        S0LSystem.render(
            word,
            length=10,
            angle=angle,
            filename=(f'{args.out_dir}/'
                      f'endpoint[{grammar},{specimen}]'
                      f'{angle}deg'),
        )

    with mp.Pool() as pool:
        print("Making grammars...")
        grammars = pool.imap(make_grammar, range(args.n_grammars))

        # render endpoint
        print("Making endpoint words...")
        words = pool.starmap(lambda grammar, i, j, angle:
                             make_word(grammar, i, j, angle, verbose=True),
                             ((grammar, i, j, a)
                              for i, grammar in enumerate(grammars)
                              for j in range(args.n_specimens)
                              for a in angles))

        print("Rendering endpoint words...")
        pool.starmap(
            lambda grammar, specimen, word, angle:
            make_render(grammar, specimen, word, angle, verbose=True),
            words
        )

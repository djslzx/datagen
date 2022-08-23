import sys
import random
import pickle
import multiprocess as mp
from lindenmayer import SOLSystem, CFG
import util


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


if __name__ == '__main__':
    ARGS = [
        "N_GRAMMARS",           # number of grammars to gen
        "MIN_N_RULES",          # min #rules per grammar
        "MAX_N_RULES",          # max #rules per grammar
        "MIN_RULE_LENGTH",      # min successor length
        "MAX_RULE_LENGTH",      # max successor length
        "N_SPECIMENS",          # number of renders of each grammar
        "DEVEL_DEPTH",         # rendering iteration depth
        "OUT_DIR",              # directory to store outputs
    ]
    if len(sys.argv) - 1 != len(ARGS):
        print("Usage: datagen.py " + " ".join(ARGS))
        sys.exit(1)

    # parse args
    (n_grammars,
     min_n_rules,
     max_n_rules,
     min_rule_length,
     max_rule_length,
     n_specimens,
     devel_depth) = [int(x) for x in sys.argv[1:-1]]
    out_dir = sys.argv[-1]
    angles = [15, 30, 45, 60, 90]

    def make_grammar(index: int, log=True) -> SOLSystem:
        g = SOLSystem_from_CFG(
            n_rules=random.randint(min_n_rules,
                                   max_n_rules),
            max_rule_length=random.randint(min_rule_length,
                                           max_rule_length),
        )
        print(f"[{index}] {g}")

        if log:
            # save grammar text representation to text file for debug
            with open(f'{out_dir}/grammars.log', 'a') as f:
                f.write(f'Grammar {index}: {g}')

        # save grammar object for model training
        with open(f'{out_dir}/grammars.dat', 'ab') as f:
            pickle.dump((index, g), f)

        return g

    def make_word(grammar: SOLSystem, grammar_i: int, specimen_i: int,
                  verbose=False, log=True) -> str:
        word = grammar.nth_expansion(devel_depth)

        if verbose:
            word_preview = word[:20] + ("..." if len(word) > 20 else "")
            print(f"[{grammar_i}, {specimen_i}] "
                  f"Generated word {word_preview} of length {len(word)}")

        if log:
            # write debug text
            with open(f'{out_dir}/words.log', 'a') as f:
                f.write(f'[{grammar_i}, {specimen_i}]: {word}\n')

        # write object
        with open(f'{out_dir}/words.dat', 'ab') as f:
            pickle.dump((grammar_i, specimen_i, word), f)

        return grammar_i, specimen_i, word

    def make_render(grammar: int, specimen: int, word: str, angle: float,
                    verbose=False):
        if verbose:
            word_preview = word[:20] + ("..." if len(word) > 20 else "")
            print(f"[{grammar}, {specimen}] "
                  f"Rendering {word_preview} of length {len(word)}")

        SOLSystem.render(
            word,
            length=10,
            angle=angle,
            filename=(f'{out_dir}/'
                      f'endpoint[{grammar},{specimen}]'
                      f'{angle}deg'),
        )

    with mp.Pool() as pool:
        print("Making grammars...")
        grammars = pool.imap(make_grammar, range(n_grammars))

        # render endpoint
        print("Making endpoint words...")
        words = pool.starmap(lambda grammar, i, j:
                             make_word(grammar, i, j, verbose=True),
                             ((grammar, i, j)
                              for i, grammar in enumerate(grammars)
                              for j in range(n_specimens)))

        print("Rendering endpoint words...")
        pool.starmap(
            lambda grammar, specimen, word, angle:
            make_render(grammar, specimen, word, angle, verbose=True),
            ((grammar, specimen, word, angle)
             for grammar, specimen, word in words
             for angle in angles)
        )

import sys
import random
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
        "RENDER_DEVEL",         # render whole seq or only endpoint
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
     devel_depth) = [int(x) for x in sys.argv[1:-2]]
    render_devel = bool(sys.argv[-2])
    out_dir = sys.argv[-1]
    angles = [15, 30, 45, 60, 90]

    def make_grammar(index: int) -> SOLSystem:
        g = SOLSystem_from_CFG(
            n_rules=random.randint(min_n_rules,
                                   max_n_rules),
            max_rule_length=random.randint(min_rule_length,
                                           max_rule_length),
        )
        # log and save grammar
        print(f"[{index}] Grammar: {g}")
        with open(f'{out_dir}/{index}.grammar', 'w') as f:
            f.write(f'Grammar: {g}')
        return g

    with mp.Pool() as pool:
        print("Making grammars...")
        grammars = pool.imap(make_grammar, range(n_grammars))

        if render_devel:
            # render full developmental sequence
            print("Making word sequences...")
            word_seqs = pool.imap(lambda g: list(g.expansions(devel_depth)),
                                  grammars)

            print("Rendering word sequences...")
            pool.starmap(
                lambda grammar, specimen, word, angle: SOLSystem.render(
                    word,
                    length=10,
                    angle=angle,
                    filename=(f'{out_dir}/'
                              f'specimen[{grammar},{specimen}]'
                              f'{angle}deg')
                ),
                [(grammar, specimen, word, angle)
                 for word_seq in word_seqs
                 for grammar, word in enumerate(word_seq)
                 for specimen in range(n_specimens)
                 for angle in angles]
            )
        else:
            # render endpoint
            print("Making endpoint words...")
            words = pool.imap(lambda g: g.nth_expansion(devel_depth),
                              grammars)

            print("Rendering endpoint words...")
            pool.starmap(
                lambda grammar, specimen, word, angle: SOLSystem.render(
                    word,
                    length=10,
                    angle=angle,
                    filename=(f'{out_dir}/'
                              f'specimen[{grammar},{specimen}]'
                              f'{angle}deg')
                ),
                ((grammar, specimen, word, angle)
                 for grammar, word in enumerate(words)
                 for specimen in range(n_specimens)
                 for angle in angles)
            )

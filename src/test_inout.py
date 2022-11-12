from inout import *
from lindenmayer import LSYSTEM_MG
import book_zoo as zoo


def test_inward_diag():
    cases = [
        (1, 0,
         [(0, 0)]),
        (2, 0,
         [(0, 0), (1, 1), (0, 1)]),
        (3, 0,
         [(0, 0), (1, 1), (2, 2),
          (0, 1), (1, 2),
          (0, 2)]),
        (5, 1,
         [(0, 1), (1, 2), (2, 3), (3, 4),
          (0, 2), (1, 3), (2, 4),
          (0, 3), (1, 4),
          (0, 4)]),
    ]
    for n, start, y in cases:
        out = list(inward_diag(n, start))
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_inward_diag")


def test_outward_diag():
    cases = [
        (1, None,
         [(0, 0)]),
        (2, None,
         [(0, 1), (1, 1), (0, 0)]),
        (3, None,
         [(0, 2),
          (1, 2), (0, 1),
          (2, 2), (1, 1), (0, 0)]),
        (5, 4,
         [
             (1, 4), (0, 3),
             (2, 4), (1, 3), (0, 2),
             (3, 4), (2, 3), (1, 2), (0, 1),
             (4, 4), (3, 3), (2, 2), (1, 1), (0, 0),
         ]),
    ]
    for n, start, y in cases:
        out = list(outward_diag(n, start))
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_outward_diag")


def demo_io():
    cases = [
        (PCFG.from_CFG(CFG.from_rules("S", [
            ("S", ["N", "V"]),
            ("V", ["V", "N"]),
            ("N", ["N", "P"]),
            ("P", ["PP", "N"]),
            ("N", ["She"]),
            ("V", ["eats"]),
            ("N", ["pizza"]),
            ("PP", ["without"]),
            ("N", ["anchovies"]),
            ("V", ["V", "N", "P"]),
            ("N", ["hesitation"]),
        ]).to_CNF()).normalized(),
         [["She", "eats", "pizza", "without", "anchovies"],
          ["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_CFG(CFG.from_rules("S", [
            ("S", ["N", "V"]),
            ("V", ["V", "N"]),
            ("N", ["N", "P"]),
            ("P", ["PP", "N"]),
            ("N", ["She"]),
            ("V", ["eats"]),
            ("N", ["pizza"]),
            ("PP", ["without"]),
            ("N", ["anchovies"]),
            ("V", ["V", "N", "P"]),
            ("N", ["hesitation"]),
        ]).to_CNF()).normalized(),
         [["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_CFG(CFG("S", {
            "S": [["A", "A"], ["B", "B"]],
            "A": [["a"]],
            "B": [["b"]],
        }).to_CNF()).normalized(),
         [["a", "a"]]),
        (PCFG.from_CFG(LSYSTEM_MG.to_CNF()).normalized(),
         [sys.to_sentence() for sys in zoo.simple_zoo_systems]),
    ]
    for g, corpus in cases:
        print(corpus, '\n', g, '\n')
        # g_io = inside_outside(g, corpus, verbose=False)
        # print("io", g_io)
        g_log = g.apply_to_weights(T.log)
        g_log.log_mode = True
        g_logio = log_io(g_log, corpus, verbose=True)
        print("logio:", g_logio.apply_to_weights(T.exp))
        # autograd_io(g, corpus)


def demo_inside():
    g = PCFG.from_CFG(CFG.from_rules("S", [
        ("S", ["N", "V"]),
        ("V", ["V", "N"]),
        ("N", ["N", "P"]),
        ("P", ["PP", "N"]),
        ("N", ["She"]),
        ("V", ["eats"]),
        ("N", ["pizza"]),
        ("PP", ["without"]),
        ("N", ["anchovies"]),
        ("V", ["V", "N", "P"]),
        ("N", ["hesitation"]),
    ]).to_CNF()).normalized()
    corpus = [["She", "eats", "pizza", "without", "anchovies"]]
    a = inside(g, corpus[0])
    print('a')
    print_map(a)
    print()

    a, b = outside(g, corpus[0])
    print('a, b')
    print_map(a)
    print_map(b)


def test_log_vs_standard():
    """
    Check that the behavior of log versions of functions
    is consistent with non-log versions
    """
    grammar_corpus_pairs = [
        (PCFG.from_CFG(CFG.from_rules("S", [
            ("S", ["N", "V"]),
            ("V", ["V", "N"]),
            ("N", ["N", "P"]),
            ("P", ["PP", "N"]),
            ("N", ["She"]),
            ("V", ["eats"]),
            ("N", ["pizza"]),
            ("PP", ["without"]),
            ("N", ["anchovies"]),
            ("V", ["V", "N", "P"]),
            ("N", ["hesitation"]),
        ]).to_CNF()).normalized(),
         [["She", "eats", "pizza", "without", "anchovies"],
          ["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_CFG(CFG.from_rules("S", [
            ("S", ["N", "V"]),
            ("V", ["V", "N"]),
            ("N", ["N", "P"]),
            ("P", ["PP", "N"]),
            ("N", ["She"]),
            ("V", ["eats"]),
            ("N", ["pizza"]),
            ("PP", ["without"]),
            ("N", ["anchovies"]),
            ("V", ["V", "N", "P"]),
            ("N", ["hesitation"]),
        ]).to_CNF()).normalized(),
         [["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_CFG(CFG("S", {
            "S": [["A", "A"], ["B", "B"]],
            "A": [["a"]],
            "B": [["b"]],
        }).to_CNF()).normalized(),
         [["a", "a"]]),
    ]
    # inside check
    for g, corpus in grammar_corpus_pairs:
        g_log = g.apply_to_weights(T.log)
        g_log.log_mode = True

        a = inside(g, corpus)
        log_a = log_alpha(g_log, corpus)

        for i, j, A in a.keys():
            a_val = a[i, j, A]
            log_a_val = log_a[i, j, A]

            if (a_val - log_a_val.exp()).abs() > 0.01:
                print(a_val, log_a_val.exp())
                pdb.set_trace()

    print(" [+] passed test_log")


if __name__ == '__main__':
    demo_inside()
    demo_io()
from inout import *
import util
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


pizza_cfg = CFG.from_rules("S", [
    ("S", ["N", "V"]),
    ("V", ["V", "N"]),
    ("N", ["N", "P"]),
    ("P", ["PP", "N"]),
    ("V", ["V", "N", "P"]),
    ("N", ["She"]),
    ("V", ["eats"]),
    ("N", ["pizza"]),
    ("PP", ["without"]),
    ("N", ["anchovies"]),
    ("N", ["hesitation"]),
]).to_CNF()


def test_io_pizza_1():
    g = PCFG.from_CFG(pizza_cfg).normalized()
    corpus = [["She", "eats", "pizza", "without", "anchovies"],
              ["She", "eats", "pizza", "without", "hesitation"]]
    g_fit = inside_outside(g, corpus)
    # make sure that g_fit assigns the same weight to 'anchovies' and 'hesitation'
    assert util.vec_approx_eq(g_fit.weight("N", ["anchovies"]),
                              g_fit.weight("N", ["hesitation"]))


def test_io_pizza_2():
    # fit PCFG to dataset w/ anchovies XOR hesitation, then compare
    g = PCFG.from_CFG(pizza_cfg).normalized()
    g_anchovies = inside_outside(g, [["She", "eats", "pizza", "without", "anchovies"]])
    g_hesitation = inside_outside(g, [["She", "eats", "pizza", "without", "hesitation"]])

    print(g_anchovies)
    print(g_hesitation)

    assert g_anchovies.weight("N", ["anchovies"]) > g_anchovies.weight("N", ["hesitation"])
    assert g_hesitation.weight("N", ["hesitation"]) > g_hesitation.weight("N", ["anchovies"])
    assert util.vec_approx_eq(g_anchovies.weight("N", ["anchovies"]),
                              g_hesitation.weight("N", ["hesitation"]))
    assert util.vec_approx_eq(g_anchovies.weight("N", ["hesitation"]),
                              g_hesitation.weight("N", ["anchovies"]))

    for nt in g.nonterminals:
        if nt == "N":
            for succ in g.successors(nt):
                if succ not in [["anchovies"], ["hesitation"]]:
                    w1, w2 = g_anchovies.weight(nt, succ), g_hesitation.weight(nt, succ)
                    assert util.vec_approx_eq(w1, w2), \
                        f"Unexpected weight variation for rules {nt} -> {succ}: {w1} != {w2}"
        else:
            w1, w2 = g_anchovies.weights[nt], g_hesitation.weights[nt]
            assert util.vec_approx_eq(w1, w2), \
                f"Unexpected weight variation for nonterminal {nt}: {w1} != {w2}"


def demo_io():  # pragma: no cover
    cases = [
        (PCFG.from_CFG(pizza_cfg).normalized(),
         [["She", "eats", "pizza", "without", "anchovies"],
          ["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_CFG(pizza_cfg).normalized(),
         [["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_CFG(CFG("S", {
            "S": [["A", "A"], ["B", "B"]],
            "A": [["a"]],
            "B": [["b"]],
        }).to_CNF()).normalized(),
         [["a", "a"]]),
        # (PCFG.from_CFG(LSYSTEM_MG.to_CNF()).normalized(),
        #  [sys.to_sentence() for sys in zoo.simple_zoo_systems]),
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


def test_log_vs_standard_io():
    """
    Check that the behavior of log versions of functions
    is consistent with non-log versions
    """
    grammar_corpus_pairs = [
        (PCFG.from_CFG(pizza_cfg).normalized(),
         [["She", "eats", "pizza", "without", "anchovies"],
          ["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_CFG(pizza_cfg).normalized(),
         [["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_CFG(CFG("S", {
            "S": [["A", "A"], ["B", "B"]],
            "A": [["a"]],
            "B": [["b"]],
        }).to_CNF()).normalized(),
         [["a", "a"]]),
    ]
    for g, corpus in grammar_corpus_pairs:
        g_log = g.log()
        a = inside(g, corpus)
        log_a = log_alpha(g_log, corpus)
        for i, j, A in a.keys():
            a_val = a[i, j, A]
            log_a_val = log_a[i, j, A]
            assert (a_val - log_a_val.exp()).abs() <= 0.01


if __name__ == '__main__':  # pragma: no cover
    demo_io()
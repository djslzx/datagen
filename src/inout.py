"""
Implementation of the inside-outside algorithm for CFGs

alpha(i, j, A, w, G) = P(phi, A -> w_i...w_j)
beta(i, j, A, w, G) = P(phi, S -> w1 ... w_i-1 . A . w_j+1 ... w_n)
"""
from typing import Dict, Tuple, List, Iterable, Callable
from cfg import PCFG
import math
import pdb

# FIXME: numerical precision issues?
# -- yup, definitely this: the values approach zero b/c of multiplies and this makes us lose info
# >> pp(lookup_map(alpha, lambda a,b,c: a==0))


def print_map(alpha: Dict, precision=4):
    for k, v in alpha.items():
        if v > 10 ** -precision:
            print(k, f'{v:.{precision}f}')
    print()


def lookup_map(d: Dict, p: Callable) -> List:
    return [(k, v) for k, v in d.items() if p(*k) and v > 0]


def inward_diag(n: int, start=0) -> Iterable[Tuple[int, int]]:
    """
    Returns the coordinates of an inward-bound traversal of a triangular grid.

    `n`: The width/height of the triangular grid
    `start`: Which diagonal to start on.
    """
    for i in range(start, n):
        for j in range(n - i):
            yield j, i + j


def outward_diag(n: int, start=None) -> Iterable[Tuple[int, int]]:
    """
    Returns the coordinates of an outward-bound traversal of a triangular grid.

    `n`: The width/height of the triangular grid
    `end`: Which diagonal to end on.
    """
    if not start:
        start = n
    for i in reversed(range(start)):
        for j in reversed(range(n - i)):
            yield j, i + j


def inside(G: PCFG, s: PCFG.Sentence, debug=False) -> Dict:
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    alpha = {}
    n = len(s)

    # initialize outermost diagonal
    for i in range(n):
        for A in G.nonterminals:
            w = [s[i]]
            a = G.weight(A, w)
            alpha[i, i, A] = a

    # recurse on other diagonals, proceeding inwards
    for i, j in inward_diag(n, start=1):
        # init each cell to 0
        for A in G.rules:
            alpha[i, j, A] = 0

        for A, succ, w in G.as_rule_list():
            if len(succ) != 2:
                continue
            B, C = succ
            if debug:
                print(f"{A} -> {B} {C}")
            for k in range(i, j):
                a = w * alpha[i, k, B] * alpha[k+1, j, C]
                alpha[i, j, A] += a
                if debug:
                    print(f"alpha({i},{j},{A}) = {w} * "
                          f"{alpha[i, k, B]} * "
                          f"{alpha[k+1, j, C]} = {a}")
    return alpha


def outside(G: PCFG, s: PCFG.Sentence, debug=False) -> Dict:
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    alpha = inside(G, s, debug)
    if debug:
        print("outside alpha:")
        print_map(alpha)

    beta = {}
    n = len(s)

    # start with inner diagonal (singleton)
    for A in G.rules:
        beta[0, n-1, A] = int(A == G.start)

    # recurse on other diagonals, proceeding outwards
    for i, j in outward_diag(n, start=n-1):

        # initialize all betas to 0
        for A in G.rules:
            beta[i, j, A] = 0

        for B, succ, w in G.as_rule_list():
            if len(succ) != 2:
                continue

            # A is right child
            C, A = succ
            for k in range(i):
                b = w * alpha[k, i-1, C] * beta[k, j, B]
                beta[i, j, A] += b
                if debug:
                    print(f"beta({i},{j},{A}) = w[{B} -> {C} {A}] * "
                          f"alpha[{k},{i-1},{C}] * beta[{k},{j},{B}] = "
                          f"{w} * {alpha[k, i-1, C]} * {beta[k, j, B]} = {b}")

            # A is left child
            A, C = succ
            for k in range(j+1, n):
                b = w * alpha[j+1, k, C] * beta[i, k, B]
                beta[i, j, A] += b
                if debug:
                    print(f"beta({i},{j},{A}) = w[{B} -> {A} {C}] * "
                          f"alpha[{j+1},{k},{C}] * beta[{i},{k},{B}] = "
                          f"{w} * {alpha[j+1, k, C]} * {beta[i, k, B]} = {b}")

    if debug:
        print_map(beta)
    return alpha, beta


def compute_counts(G: PCFG, corpus: List[PCFG.Sentence], debug=False):
    """
    Count the number of times any rule A -> x is used in the corpus.
    """
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    counts = {}
    S = G.start

    # initialize
    for A, succ, phi in G.as_rule_list():
        counts[A, tuple(succ)] = 0

    # compute
    for W in corpus:
        alpha, beta = outside(G, W, debug=False)
        n = len(W)

        for A, succ, phi in G.as_rule_list():
            if debug:
                print(f"{A} -> {succ} for word {W}")

            # alpha_0,n-1(S) = P_phi(S -> W) = P_phi(W)
            if alpha[0, n-1, S] == 0:
                pdb.set_trace()

            weight = phi / alpha[0, n-1, S]

            if len(succ) == 1:
                w = succ[0]
                counts[A, (w,)] += weight * \
                    math.fsum(beta[i, i, A] for i in range(n))

            elif len(succ) == 2:
                B, C = succ
                tot = 0
                for i in range(n):
                    for j in range(i, n):
                        for k in range(j+1, n):
                            if debug:
                                print(f"sum[{A} -> {succ}] += "
                                      f"beta({i},{k},{A}) * "
                                      f"alpha[{i},{j},{B}] * "
                                      f"alpha[{j+1},{k},{C}] ",
                                      end='')
                                print(f"= {beta[i,k,A]} * {alpha[i,j,B]} "
                                      f"* {alpha[j+1,k,C]}")
                            tot += (beta[i, k, A] *
                                    alpha[i, j, B] *
                                    alpha[j+1, k, C])
                counts[A, (B, C)] += weight * tot
            else:
                raise ValueError("Expected PCFG to be in CNF, "
                                 f"but got the rule {A} -> {succ}.")
    return counts


def inside_outside(G: PCFG, corpus: List[PCFG.Sentence], debug=False) -> PCFG:
    """
    Perform one step of inside-outside.
    """
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    counts = compute_counts(G, corpus, debug)
    sum_counts = {A: sum(counts[A, tuple(succ)]
                         for succ in succs)
                  for A, succs in G.rules.items()}
    rules = []
    for A, succ, _ in G.as_rule_list():
        weight = counts[A, tuple(succ)] / sum_counts[A] \
            if sum_counts[A] > 0 else 0
        rules.append((A, succ, weight))
    return PCFG.from_rule_list(G.start, rules)


def demo_io():
    g = PCFG.from_rule_list(
        start="S",
        rules=[
            ("S", ["N", "V"], 1),
            ("V", ["V", "N"], 1),
            ("N", ["N", "P"], 1),
            ("P", ["PP", "N"], 1),
            ("N", ["She"], 1),
            ("V", ["eats"], 1),
            ("N", ["pizza"], 1),
            ("PP", ["without"], 1),
            ("N", ["anchovies"], 1),
            ("V", ["V", "N", "P"], 1),
            ("N", ["hesitation"], 1),
        ],
    ).to_CNF()
    g.set_uniform_weights()
    print(g, '\n')
    s1 = ["She", "eats", "pizza", "without", "anchovies"]
    s2 = ["She", "eats", "pizza", "without", "hesitation"]

    for i in range(3):
        g = inside_outside(g, [s1, s2])
        print(g)
        print()

    # a1 = inside(g, s1, debug=True)
    # a2 = inside(g, s2, debug=True)
    # print_map(a1)
    # print_map(a2)

    # alpha, beta = outside(g, s1, debug=False)
    # print_map(alpha)
    # print_map(beta)

    # c = counts(g, [s1, s2], debug=True)
    # print_map(c)


def test_inside():
    cases = [
        (PCFG("AXIOM", {
            "AXIOM": [["_F", "M"]],
            "M": [["F"], ["-S", "FM"]],
            "FM": [["_F", "M"]],
            "-S": [["-"], ["_-", "-S"]],
            "_F": [["F"]],
            "_-": [["-"]],
        }, "uniform"),
         "F - F - F F".split(),
         # FIXME: incomplete
         # FIXME: F-F-F is not attainable with this grammar
         (0, 4, "AXIOM")),
    ]
    for pcfg, W, key in cases:
        alpha = inside(pcfg, W)
        if alpha[key] == 0:
            print(f"Failed test_inside on grammar={pcfg}, word={W}, key={key}")
            inside(pcfg, W, debug=True)
            pdb.set_trace()
            inside(pcfg, W)
            exit(1)
    print(" [+] passed test_inside")


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


if __name__ == '__main__':
    test_inward_diag()
    test_outward_diag()
    test_inside()
    # demo_io()

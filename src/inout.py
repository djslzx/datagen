"""
Implementation of the inside-outside algorithm for CFGs

alpha(i, j, A, w, G) = P(phi, A -> w_i...w_j)
beta(i, j, A, w, G) = P(phi, S -> w1 ... w_i-1 . A . w_j+1 ... w_n)
"""
from typing import Dict, Iterable, Tuple
from cfg import PCFG


def print_map(alpha: Dict):
    for k, v in alpha.items():
        if v > 0:
            print(k, f'{v:.5f}')
    print()


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


def inside(G: PCFG, s: PCFG.Sentence, debug=False) -> Dict:
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
    alpha = inside(G, s, debug=False)
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
    return beta


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
    # g.set_uniform_weights()
    print(g)
    s1 = ["She", "eats", "pizza", "without", "anchovies"]
    s2 = ["She", "eats", "pizza", "without", "hesitation"]

    # a1 = inside(g, s1, debug=True)
    # a2 = inside(g, s2, debug=True)
    # print_map(a1)
    # print_map(a2)

    outside(g, s1, debug=True)


if __name__ == '__main__':
    test_inward_diag()
    test_outward_diag()
    demo_io()

"""
Implementation of the inside-outside algorithm for CFGs

alpha(i, j, A, w, G) = P(phi, A -> w_i...w_j)
beta(i, j, A, w, G) = P(phi, S -> w1 ... w_i-1 . A . w_j+1 ... w_n)
"""
from typing import Dict, Iterable, Tuple
from cfg import PCFG


def print_alpha(alpha: Dict):
    for k, v in alpha.items():
        if v > 0:
            print(k, f'{v:.5f}')


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
    alpha = inside(G, s, debug=debug)
    beta = {}
    n = len(s)

    # start with inner diagonal (singleton)
    beta[n-1, n-1, G.start] = 1
    for A in G.rules:
        beta[n-1, n-1, A] = 0

    # recurse on other diagonals, proceeding outwards
    for i, j in inward_diag(n, end=n-2, rev=True):
        print(i, j)

        # initialize all betas to 0
        for A in G.rules:
            beta[i, j, A] = 0

        for A, succ, w in G.as_rule_list():
            if len(succ) != 2:
                continue
            B, C = succ
            for k in range(i):
                pass

            for k in range(j+1, n):
                pass


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
    w1 = ["She", "eats", "pizza", "without", "anchovies"]
    w2 = ["She", "eats", "pizza", "without", "hesitation"]
    a1 = inside(g, w1, debug=True)
    a2 = inside(g, w2, debug=True)
    print_alpha(a1)
    print_alpha(a2)


if __name__ == '__main__':
    test_inward_diag()
    test_outward_diag()
    demo_io()

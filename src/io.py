"""
Implementation of the inside-outside algorithm for CFGs

alpha(i, j, A, w, G) = P(phi, A -> w_i...w_j)
beta(i, j, A, w, G) = P(phi, S -> w1 ... w_i-1 . A . w_j+1 ... w_n)
"""
from typing import Dict
from cfg import PCFG

# TODO: use CYK


def inside(G: PCFG, word: PCFG.Word):
    alpha = {}
    back = {}
    n = len(word)

    for i in range(n):
        for nt in G.nonterminals:
            alpha[i, i, nt] = G.weight(nt, [word[i-1]])

    print(alpha)


def alpha(cache: Dict, G: PCFG, word: PCFG.Word,
          i: int, j: int, nt: PCFG.Letter):
    """
    alpha(i, j, nt) = P(phi, nt -> w_i...w_j)
     = sum_B,C sum_k in i..j P_phi(nt -> B . C) alpha(i, k, B) alpha(k+1, j, C)

    Assumes that G is in Chomsky normal form.
    """
    assert nt in G.rules, f"{nt} is not a nonterminal in grammar {G}"
    if (i, j, nt) in cache:
        return cache[i, j, nt]
    if i == j:
        # print(i, nt, word[i-1], G.weight(nt, word[i-1]),
        #       [G.weight(nt, [l]) for l in word])
        cache[i, i, nt] = G.weight(nt, [word[i-1]])
    else:
        total = 0
        for s, w in zip(G.rules[nt], G.weights[nt]):
            if len(s) == 2:
                b, c = s
                print(b, c)
                for k in range(i, j+1):
                    total += (w *
                              alpha(cache, G, word, i, k, b) *
                              alpha(cache, G, word, k+1, j, c))
        cache[i, j, nt] = total
    print(i, j, nt, '=>', cache[i, j, nt])
    return cache[i, j, nt]


def beta(a_cache: Dict, b_cache: Dict, G: PCFG, word: PCFG.Word,
         i: int, j: int, nt: PCFG.Letter):
    """
    beta(i, j, A) = Pr(S -> w1..wi-1 . A . wj+1..wn)
    beta(1, n, S) = Pr(S -> S) = 1
    beta(1, n, A) = Pr(S -> A) 0, where A /= S
    """
    assert nt in G.rules, f"{nt} is not a nonterminal in grammar {G}"
    if (i, j, nt) in b_cache:
        return b_cache[i, j, nt]
    if i == 1:
        if nt == G.start:
            # beta(1, n, S) = Pr(S -> S) = 1
            b_cache[i, j, nt] = 1
        else:
            # beta(1, n, A) = Pr(S -> A) 0, where A /= S
            b_cache[i, j, nt] = 0
        return b_cache[i, j, nt]

    # beta(i, j, A) = sum_B,C sum_k=1..i-1 phi(B -> CA) *
    #                 alpha(k, i-1, C) * beta(k, j, B) +
    #                 sum_B,C sum_k=j+1..n phi(B -> AC) *
    #                 alpha(j+1, k, C) * beta(i, k, B)
    n = len(word)
    total = 0
    for b, succ, w in G.as_rule_list():
        if b == nt or len(succ) < 2:
            continue
        if nt == succ[1]:
            c, _ = succ
            for k in range(1, i):
                total += (w * alpha(a_cache, G, word, k, i-1, c) *
                          beta(a_cache, b_cache, G, word, k, j, b))
        elif nt == succ[0]:
            _, c = succ
            for k in range(j+1, n+1):
                total += (w * alpha(a_cache, G, word, j+1, k, c) *
                          beta(a_cache, b_cache, G, word, i, k, b))
    b_cache[i, j, nt] = total
    return b_cache[i, j, nt]


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
    print(g)
    s = g.start
    cache = {}
    w = ["She", "eats", "pizza", "without", "anchovies"]
    # a = alpha(cache, g, w, 1, 5, s)
    inside(g, w)


if __name__ == '__main__':
    demo_io()

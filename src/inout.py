"""
Implementation of the inside-outside algorithm for CFGs

alpha(i, j, A, w, G) = P(phi, A -> w_i...w_j)
beta(i, j, A, w, G) = P(phi, S -> w1 ... w_i-1 . A . w_j+1 ... w_n)
"""
from __future__ import annotations
import time
from typing import Dict, Tuple, List, Iterable
import numpy as np
import scipy.stats as stats
import torch as T
import math
import pdb

from cfg import CFG, PCFG


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


def inside(G: PCFG, s: CFG.Sentence) -> Dict[Tuple[int, int, CFG.Word], T.Tensor]:
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    alpha = {}
    n = len(s)

    # initialize outermost diagonal
    for i in range(n):
        for A in G.nonterminals:
            alpha[i, i, A] = G.weight(A, [s[i]])

    # recurse on other diagonals, proceeding inwards
    for i, j in inward_diag(n, start=1):
        for A in G.nonterminals:
            alpha[i, j, A] = T.tensor(0, dtype=T.float64)
            for succ in G.successors(A):
                if len(succ) != 2:
                    continue
                B, C = succ
                for k in range(i, j):
                    alpha[i, j, A] += G.weight(A, succ) * alpha[i, k, B] * alpha[k+1, j, C]
    return alpha


def outside(G: PCFG, s: CFG.Sentence) -> Tuple[Dict[Tuple[int, int, CFG.Word], T.Tensor],
                                               Dict[Tuple[int, int, CFG.Word], T.Tensor]]:
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    alpha = inside(G, s)
    beta = {}
    n = len(s)

    # start with inner diagonal (singleton)
    for A in G.nonterminals:
        beta[0, n-1, A] = T.tensor(int(A == G.start))

    # recurse on other diagonals, proceeding outwards
    for i, j in outward_diag(n, start=n-1):
        for A in G.nonterminals:
            beta[i, j, A] = T.tensor(0, dtype=T.float64)
        for A, succ, w in G.as_weighted_rules():
            if len(succ) != 2:
                continue
            B, C = succ
            for k in range(j+1, n):  # i < j < k
                beta[i, j, B] += w * beta[i, k, A] * alpha[j+1, k, C]
            for k in range(i):  # k < i < j
                beta[i, j, C] += w * beta[k, j, A] * alpha[k, i-1, B]
    return alpha, beta


def autograd_inside(G: PCFG, s: CFG.Sentence) -> T.Tensor:
    assert G.is_in_CNF()
    n = len(s)
    alpha = np.empty((n, n), dtype=object)

    # initialize outermost diagonal
    for i in range(n):
        alpha[i, i] = {}
        for A in G.nonterminals:
            alpha[i, i][A] = G.weight(A, [s[i]])

    # recurse on other diagonals, proceeding inwards
    for i, j in inward_diag(n, start=1):
        alpha[i, j] = {}
        for A in G.nonterminals:
            alpha[i, j][A] = T.tensor(0, dtype=T.float64)
            for succ in G.successors(A):
                if len(succ) != 2:
                    continue
                B, C = succ
                for k in range(i, j):
                    alpha[i, j][A] += G.weight(A, succ) * alpha[i, k][B] * alpha[k + 1, j][C]
    return alpha[0, len(s) - 1][G.start]


def autograd_outside(G: PCFG, corpus: List[CFG.Sentence], iters, verbose=False) -> PCFG:
    """
    Uses automatic differentiation to implement Inside-Outside.
    """
    assert G.is_in_CNF()
    assert G.is_normalized()

    g = G.copy()
    optimizer = T.optim.Adam(g.parameters())
    for i in range(iters):
        print(f"[Inside-outside: iter {i}]", end="")
        for word in corpus:
            if verbose:
                print(f"  Fitting to word {''.join(word)} of length {len(word)}...")
                t_start = time.time()
            else:
                print(".", end="")

            loss = -T.log(autograd_inside(g, word))
            loss.backward()
            optimizer.step()

            if verbose:
                print(f"    took {time.time() - t_start}s")
        print()
    return g.normalized()


def compute_counts(G: PCFG, corpus: List[CFG.Sentence], verbose=False) -> Dict[Tuple[CFG.Word, Tuple[str]], float]:
    """
    Count the number of times any rule A -> x is used in the corpus.
    """
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    counts = {}
    for A, succ, _ in G.as_weighted_rules():
        counts[A, tuple(succ)] = 0
    for i, W in enumerate(corpus, 1):
        if verbose:  # pragma: no cover
            print(f"Processing word {i}/{len(corpus)}: {W}...")

        alpha, beta = outside(G, W)
        n = len(W)
        pr_W = alpha[0, n-1, G.start]
        assert pr_W > 0, f"Found {W}, which cannot be generated by {G}"
        for A, succ, phi in G.as_weighted_rules():
            if len(succ) == 1:
                counts[A, tuple(succ)] += phi / pr_W * \
                    math.fsum(beta[i, i, A] for i in range(n) if succ[0] == W[i])
            elif len(succ) == 2:
                B, C = succ
                counts[A, (B, C)] += phi / pr_W * sum(
                    beta[i, k, A] * alpha[i, j, B] * alpha[j + 1, k, C]
                    for i in range(n)
                    for j in range(i, n)
                    for k in range(j+1, n)
                )
    return counts


def inside_outside_step(G: PCFG, corpus: List[CFG.Sentence],
                        smoothing: float = 0.1, verbose=False) -> PCFG:
    """
    Perform one step of inside-outside.
    """
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    assert 0.1 <= smoothing <= 10, "Smoothing should be in [0.1, 10]"

    counts = compute_counts(G, corpus, verbose=verbose)
    counts = {k: v + smoothing for k, v in counts.items()}
    sums = {A: sum(counts[A, tuple(succ)] for succ in succs)
            for A, succs in G.rules()}
    rules = []
    for A, succ, _ in G.as_weighted_rules():
        weight = counts[A, tuple(succ)] / sums[A]
        rules.append((A, succ, weight))
    return PCFG.from_weighted_rules(G.start, rules)


def inside_outside(G: PCFG, corpus: List[CFG.Sentence],
                   smoothing=0.1, verbose=False) -> PCFG:
    """
    Perform inside-outside until the grammar converges.
    """
    assert G.is_in_CNF(), "Inside-outside requires the grammar to be in CNF"
    assert G.is_normalized()

    def step(g: PCFG, corpus: List[CFG.Sentence]) -> PCFG:
        return inside_outside_step(g, corpus, smoothing, verbose=verbose)

    g = G.normalized()
    i = 1
    while True:
        g_prev, g = g, step(g, corpus)
        if verbose:  # pragma: no cover
            print(f"IO step {i}:\n"
                  f"prev: {g_prev}\n"
                  f"current: {g}")
            i += 1
        if g.approx_eq(g_prev, threshold=1e-7):
            break
    return g


def log_alpha(G: PCFG, s: CFG.Sentence) -> Dict[Tuple[int, int, CFG.Word], T.Tensor]:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    log_a = {}
    n = len(s)

    # initialize outermost diagonal
    for i in range(n):
        for A in G.nonterminals:
            log_a[i, i, A] = G.weight(A, [s[i]])

    # recurse on other diagonals, proceeding inwards
    for i, j in inward_diag(n, start=1):
        # init each cell to 0
        for A in G.nonterminals:
            log_a[i, j, A] = T.tensor(-T.inf)

        # add up weights of rules
        for A, succ, w in G.as_weighted_rules():
            if len(succ) != 2:
                continue
            B, C = succ
            for k in range(i, j):
                log_a[i, j, A] = T.logaddexp(log_a[i, j, A],
                                             w + log_a[i, k, B] + log_a[k+1, j, C])
    return log_a


def log_alpha_beta(G: PCFG, s: CFG.Sentence) -> Tuple[Dict[Tuple[int, int, CFG.Word], T.Tensor],
                                                      Dict[Tuple[int, int, CFG.Word], T.Tensor]]:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    log_a = log_alpha(G, s)
    log_b = {}
    n = len(s)

    # start with inner diagonal (singleton)
    for A in G.nonterminals:
        log_b[0, n-1, A] = T.log(T.tensor(A == G.start))

    # recurse on other diagonals, proceeding outwards
    for i, j in outward_diag(n, start=n-1):
        for A in G.nonterminals:
            log_b[i, j, A] = -T.tensor(T.inf)
        for A, succ, w in G.as_weighted_rules():
            if len(succ) != 2:
                continue
            B, C = succ
            for k in range(j+1, n):  # i < j < k
                log_b[i, j, B] = T.logaddexp(log_b[i, j, B],
                                             w + log_b[i, k, A] + log_a[j+1, k, C])
            for k in range(i):  # k < i < j
                log_b[i, j, C] = T.logaddexp(log_b[i, j, C],
                                             w + log_b[k, j, A] + log_a[k, i-1, B])

    return log_a, log_b


def log_counts(G: PCFG, corpus: List[CFG.Sentence], verbose=False) -> Dict[Tuple[CFG.Word, Tuple[str]], T.Tensor]:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    log_c = {}
    for A, succ, _ in G.as_weighted_rules():
        log_c[A, tuple(succ)] = T.tensor(-T.inf)
    for i, W in enumerate(corpus, 1):
        if verbose: print(f"Processing word {W} ({i}/{len(corpus)})")

        log_a, log_b = log_alpha_beta(G, W)
        n = len(W)
        log_pr_W = log_a[0, n-1, G.start]  # alpha_0,n-1(S) = P_phi(S -> W) = P_phi(W)
        assert not T.isneginf(log_pr_W), f"Found {W}, which cannot be generated by {G}"

        for A, succ, log_w in G.as_weighted_rules():
            if verbose: print(f"{A} -> {succ} [{log_w}]")

            if len(succ) == 1:
                term = log_w - log_pr_W + T.logsumexp(T.tensor([
                    log_b[i, i, A] if W[i] == succ[0] else T.tensor(-T.inf)
                    # log_b[i, i, A] + T.log(T.tensor(W[i] == succ[0]))
                    # = log_b[i,i,A] + (0 if W[i] == succ[0] else -inf)
                    for i in range(n)
                ]), dim=0)
                log_c[A, tuple(succ)] = T.logaddexp(log_c[A, tuple(succ)], term)
            elif len(succ) == 2:
                B, C = succ
                term = log_w - log_pr_W + T.logsumexp(T.tensor([
                    log_b[i, k, A] + log_a[i, j, B] + log_a[j + 1, k, C]
                    # Pr(S uses A to make W_i..k) * Pr(B -> W_i..j) * Pr(C -> W_j+1..k)
                    # = beta[i, k, A] * alpha[i, j, B] * alpha[j+1, k, C]
                    for i in range(n)
                    for j in range(i, n)
                    for k in range(j + 1, n)  # k starts at j+1 b/c of a_j+1,k
                ]), dim=0)
                log_c[A, (B, C)] = T.logaddexp(log_c[A, (B, C)], term)
    return log_c


def log_io_step(G: PCFG, corpus: List[CFG.Sentence], alpha=0.1, verbose=False) -> PCFG:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    alpha = T.tensor(alpha)
    log_c = log_counts(G, corpus, verbose=verbose)
    rules = []
    for A, succs in G.rules():
        n = T.tensor(len(succs))
        denom = T.logaddexp(T.logsumexp(T.tensor([log_c[A, tuple(succ)] for succ in succs]), dim=0),
                            T.log(alpha * n))
        for succ in succs:
            weight = T.logaddexp(log_c[A, tuple(succ)], T.log(alpha))
            rules.append((A, succ, weight - denom))
    return PCFG.from_weighted_rules(G.start, rules)


def log_dirio_step(G: PCFG, corpus: List[CFG.Sentence], alpha, verbose=False) -> PCFG:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    if verbose: print(f"Processing corpus of length {len(corpus)}")
    log_c = log_counts(G, corpus, verbose=verbose)
    rules = []
    for A, succs in G.rules():
        n = T.tensor(len(succs))
        a = np.repeat(alpha, n)
        c = np.array([log_c[A, tuple(succ)].exp().detach() for succ in succs])
        new_weights = stats.dirichlet.rvs(alpha=a + c)[0]
        for succ, w in zip(succs, new_weights):
            rules.append((A, succ, T.tensor(w).log()))
    return PCFG.from_weighted_rules(G.start, rules)


def log_io(G: PCFG, corpus: List[CFG.Sentence], smoothing=0.1, verbose=False) -> PCFG:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    g = G.apply_to_weights(lambda x: x)
    i = 1
    t = time.time()
    while True:
        g_prev, g = g, log_io_step(G, corpus, smoothing)
        if verbose:  # pragma: no cover
            duration = time.time() - t
            print(f"IO step {i} took {duration}s")
            i += 1
            t = time.time()
        if g.approx_eq(g_prev, threshold=1e-8):
            break
    return g

"""
Implementation of the inside-outside algorithm for CFGs

alpha(i, j, A, w, G) = P(phi, A -> w_i...w_j)
beta(i, j, A, w, G) = P(phi, S -> w1 ... w_i-1 . A . w_j+1 ... w_n)
"""
import time
from typing import Dict, Tuple, List, Iterable, Callable, Optional
from pprint import pp
import torch as T
import math
import pdb

from cfg import PCFG


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


def inside(G: PCFG, s: PCFG.Sentence) -> Dict:
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    alpha = {}
    n = len(s)

    # initialize outermost diagonal
    for i in range(n):
        for A in G.nonterminals:
            alpha[i, i, A] = G.weight(A, [s[i]])

    # recurse on other diagonals, proceeding inwards
    for i, j in inward_diag(n, start=1):
        # init each cell to 0
        for A in G.rules:
            alpha[i, j, A] = 0
        # add up weights of rules
        for A, succ, w in G.as_rule_list():
            if len(succ) != 2:
                continue
            B, C = succ
            for k in range(i, j):
                a = w * alpha[i, k, B] * alpha[k+1, j, C]
                alpha[i, j, A] += a
    return alpha


def outside(G: PCFG, s: PCFG.Sentence) -> Tuple[Dict, Dict]:
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    alpha = inside(G, s)
    beta = {}
    n = len(s)

    # start with inner diagonal (singleton)
    for A in G.rules:
        beta[0, n-1, A] = int(A == G.start)

    # recurse on other diagonals, proceeding outwards
    for i, j in outward_diag(n, start=n-1):
        for A in G.rules:
            beta[i, j, A] = 0
        for A, succ, w in G.as_rule_list():
            if len(succ) != 2:
                continue
            B, C = succ
            for k in range(j+1, n):  # i < j < k
                beta[i, j, B] += w * beta[i, k, A] * alpha[j+1, k, C]
            for k in range(i):  # k < i < j
                beta[i, j, C] += w * beta[k, j, A] * alpha[k, i-1, B]
    return alpha, beta


def autograd_io(G: PCFG, corpus: List[PCFG.Sentence], iters=1000,
                callback=None, log=False):
    """
    Uses automatic differentiation to implement Inside-Outside.
    """
    def ins(g: PCFG, w: PCFG.Sentence) -> float:
        alpha = inside(g, w)
        z = alpha[0, len(w) - 1, g.start]
        return z

    def diff(plist1: List[T.Tensor], plist2: List[T.Tensor]) -> float:
        return T.stack([(x - y).abs().sum()
                        for x, y in zip(plist1, plist2)]).sum()

    def eq(plist1, plist2) -> bool:
        return all(T.eq(x, y)
                   for x, y in zip(plist1, plist2))

    optimizer = T.optim.Adam(G.parameters())
    # prev = None
    # while prev is None or not eq(prev, G.parameters()):  # diff(prev, G.parameters()) > 1e-10:
    for i in range(iters):
        # prev = list([p.clone() for p in G.parameters()])
        if log:
            print(f"[{i}/{iters}]")
        for word in corpus:
            z = ins(G, word)    # TODO: log pr
            loss = -z
            loss.backward()
            optimizer.step()
        if callback:
            callback(i, G)

    G.normalize_weights_()
    # d = diff(prev, G.parameters())
    # print(prev, list(G.parameters()), d)


def compute_counts(G: PCFG, corpus: List[PCFG.Sentence], verbose=False):
    """
    Count the number of times any rule A -> x is used in the corpus.
    """
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    counts = {}
    for A, succ, _ in G.as_rule_list():
        counts[A, tuple(succ)] = 0
    for i, W in enumerate(corpus, 1):
        if verbose:
            print(f"Processing word {i}/{len(corpus)}: {W}...")

        alpha, beta = outside(G, W)
        n = len(W)
        pr_W = alpha[0, n-1, G.start]
        assert pr_W > 0, f"Found {W}, which cannot be generated by {G}"
        for A, succ, phi in G.as_rule_list():
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


def inside_outside_step(G: PCFG, corpus: List[PCFG.Sentence],
                        smoothing: float = 0.1, verbose=False) -> PCFG:
    """
    Perform one step of inside-outside.
    """
    assert G.is_in_CNF(), "Inside-outside requires G to be in CNF"
    assert 0.1 <= smoothing <= 10, "Smoothing should be in [0.1, 10]"

    counts = compute_counts(G, corpus, verbose=verbose)
    counts = {k: v + smoothing for k, v in counts.items()}
    sums = {A: sum(counts[A, tuple(succ)] for succ in succs)
            for A, succs in G.rules.items()}
    rules = []
    for A, succ, _ in G.as_rule_list():
        weight = counts[A, tuple(succ)] / sums[A]
        rules.append((A, succ, weight))
    return PCFG.from_rule_list(G.start, rules)


def log_alpha(G: PCFG, s: PCFG.Sentence) -> Dict:
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
        for A in G.rules:
            log_a[i, j, A] = T.tensor(-T.inf)

        # add up weights of rules
        for A, succ, w in G.as_rule_list():
            if len(succ) != 2:
                continue
            B, C = succ
            for k in range(i, j):
                log_a[i, j, A] = T.logaddexp(log_a[i, j, A],
                                             w + log_a[i, k, B] + log_a[k+1, j, C])
    return log_a


def log_alpha_beta(G: PCFG, s: PCFG.Sentence) -> Tuple[Dict, Dict]:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    log_a = log_alpha(G, s)
    log_b = {}
    n = len(s)

    # start with inner diagonal (singleton)
    for A in G.rules:
        log_b[0, n-1, A] = T.log(T.tensor(A == G.start))

    # recurse on other diagonals, proceeding outwards
    for i, j in outward_diag(n, start=n-1):
        for A in G.rules:
            log_b[i, j, A] = -T.tensor(T.inf)
        for A, succ, w in G.as_rule_list():
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


def log_counts(G: PCFG, corpus: List[PCFG.Sentence]) -> Dict:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    log_c = {}
    for A, succ, _ in G.as_rule_list():
        log_c[A, tuple(succ)] = T.tensor(-T.inf)
    for i, W in enumerate(corpus, 1):
        log_a, log_b = log_alpha_beta(G, W)
        n = len(W)
        log_pr_W = log_a[0, n-1, G.start]  # alpha_0,n-1(S) = P_phi(S -> W) = P_phi(W)
        assert not T.isneginf(log_pr_W), f"Found {W}, which cannot be generated by {G}"

        for A, succ, log_w in G.as_rule_list():
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


def log_io_step(G: PCFG, corpus: List[PCFG.Sentence], smoothing=0.1) -> PCFG:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    alpha = T.tensor(smoothing)
    log_c = log_counts(G, corpus)
    rules = []
    for A, succs in G.rules.items():
        n = T.tensor(len(succs))
        denom = T.logaddexp(T.logsumexp(T.tensor([log_c[A, tuple(succ)] for succ in succs]), dim=0),
                            T.log(alpha * n))
        for succ in succs:
            weight = T.logaddexp(log_c[A, tuple(succ)], T.log(alpha))
            rules.append((A, succ, weight - denom))
    return PCFG.from_rule_list(G.start, rules)


def log_io(G: PCFG, corpus: List[PCFG.Sentence], smoothing=0.1, verbose=False) -> PCFG:
    assert G.is_in_CNF()
    assert G.log_mode
    assert G.is_normalized()

    g = G.apply_to_weights(lambda x: x)
    i = 1
    t = time.time()
    while True:
        g_prev, g = g, log_io_step(G, corpus, smoothing)
        if verbose:
            duration = time.time() - t
            print(f"IO step {i} took {duration}s")
            i += 1
            t = time.time()
        if g.approx_eq(g_prev, threshold=1e-8):
            break
    return g


def inside_outside(G: PCFG, corpus: List[PCFG.Sentence],
                   smoothing=0.1, verbose=False) -> PCFG:
    """
    Perform inside-outside until the grammar converges.
    """
    assert G.is_in_CNF(), "Inside-outside requires the grammar to be in CNF"
    assert G.is_normalized()

    def step(g: PCFG, corpus: List[PCFG.Sentence]) -> PCFG:
        return inside_outside_step(g, corpus, smoothing, verbose=verbose)

    g = G.normalized()
    i = 1
    while True:
        g_prev, g = g, step(g, corpus)
        if verbose:
            print(f"IO step {i}:\n"
                  f"prev: {g_prev}\n"
                  f"current: {g}")
            i += 1
        if g.approx_eq(g_prev, threshold=1e-7):
            break
    return g


def demo_io():
    cases = [
        (PCFG.from_rule_list(
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
        ).to_CNF().normalized(),
            [["She", "eats", "pizza", "without", "anchovies"],
             ["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_rule_list(
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
        ).to_CNF().normalized(),
            [["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG(start="S",
              rules={
                  "S": [["A", "A"], ["B", "B"]],
                  "A": [["a"]],
                  "B": [["b"]],
              }).to_CNF(),
         [["a", "a"]]),
    ]
    for g, corpus in cases:
        print(corpus, '\n', g, '\n')
        g_io = inside_outside(g, corpus, verbose=False)
        print("io", g_io)

        g_log = g.apply_to_weights(T.log)
        g_log.log_mode = True
        g_logio = log_io(g_log, corpus)
        print("logio:", g_logio.apply_to_weights(T.exp))
        # autograd_io(g, corpus)


def demo_inside():
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
    corpus = [["She", "eats", "pizza", "without", "anchovies"]]
    a = inside(g, corpus[0])
    print('a')
    print_map(a)
    print()

    a, b = outside(g, corpus[0])
    print('a, b')
    print_map(a)
    print_map(b)


def test_log():
    """
    Check that the behavior of log versions of functions
    is consistent with non-log versions
    """
    grammar_corpus_pairs = [
        (PCFG.from_rule_list(
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
        ).to_CNF().normalized(),
         [["She", "eats", "pizza", "without", "anchovies"],
          ["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG.from_rule_list(
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
        ).to_CNF().normalized(),
         [["She", "eats", "pizza", "without", "hesitation"]]),
        (PCFG(start="S",
              rules={
                  "S": [["A", "A"], ["B", "B"]],
                  "A": [["a"]],
                  "B": [["b"]],
              }).to_CNF(),
         [["a", "a"]]),
    ]
    # inside check
    for g, corpus in grammar_corpus_pairs:
        g_log = g.apply_to_weights(T.log)
        g_log.log_mode = True

        a = inside(g, corpus)
        log_a = log_alpha(g_log, corpus)

        for i, j, A in a.keys():
            a_val = T.tensor(a[i, j, A])
            log_a_val = T.tensor(log_a[i, j, A])

            if (a_val - log_a_val.exp()).abs() > 0.01:
                print(a_val, log_a_val.exp())
                pdb.set_trace()

    print(" [+] passed test_log")


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
    # test_inward_diag()
    # test_outward_diag()
    # demo_inside()
    test_log()
    demo_io()
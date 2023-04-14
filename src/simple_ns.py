# ns without the evo
"""
def ns(d: distance function,
       m: sample count,
       L: DSL):
    S = empty_set
    G = uniform_grammar(L)
    for t in 1..T
        samples = get_samples(G, m)
        x = best(samples, d)
        S = S + x
        G = fit(G, x)  # can be done incrementally with EM/counts
    return S
"""
from typing import Set, Tuple, Dict, List, Callable, Collection
import numpy as np
from sklearn.neighbors import NearestNeighbors
from einops import rearrange, reduce
from pprint import pp

from lang import Language, Tree
import regexpr
import util

def evaluate(L: Language, x: Tree, n: int) -> Set[str]:
    if x is not None:
        return {L.eval(x, env={}) for _ in range(n)}
    else:
        return set()

def features(L: Language, s: Collection[Tree], n: int) -> np.ndarray:
    # output shape: (|s|, n)
    xs = []
    for x in s:
        outputs = evaluate(L, x, n)
        xs.append(L.featurizer.apply(outputs))
    return np.concatenate(xs)

def best(L: Language, xs: Collection[Tree], knn: NearestNeighbors, n: int) -> Tuple[Tree, float]:
    fs = features(L, xs, n)  # (|xs|, n)
    best_score = 0
    best_tree = None
    for x, f in zip(xs, fs):
        dists, _ = knn.kneighbors(rearrange(f, "n -> 1 n"))
        score = reduce(dists, "1 n_neighbors -> 1", reduction='sum')
        if score > best_score or best_tree is None:
            best_score = score
            best_tree = x
    return best_tree, best_score

def ns(L: Language, s0: Set[Tree], m: int, t: int):
    s = list(s0)
    for i in range(t):
        f_s = features(L, s)
        knn = NearestNeighbors().fit(f_s)
        L.fit(s, alpha=1e-10)
        samples = [L.sample() for j in range(m)]
        x, score = best(L, samples, knn)
        print(f"Added {x} w/ score {score}")
        s.append(x)
    return s

if __name__ == "__main__":
    l = regexpr.Regex()
    s0 = {l.parse(".")}
    p = ns(l, s0=s0, m=64, t=100)
    pp(p)

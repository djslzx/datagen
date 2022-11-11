import torch as T
import itertools as it
from typing import List, Tuple, Iterable, Optional, Set


def split_list(s: List[str], t: str) -> List[List[str]]:
    out = []
    while True:
        try:
            i = s.index(t)
            r, s = s[:i], s[i+1:]
            if r:
                out.append(r)
        except ValueError:
            if s:
                out.append(s)
            return out


def combinations(alphabet: Set) -> Iterable:
    """
    Returns all any-length combinations of letters in the alphabet.

    Constructs all words that can be formed as a combination of
    0 or 1 uses of each letter in the alphabet.
    """
    return it.chain.from_iterable(it.combinations(alphabet, r=i + 1)
                                  for i in range(len(alphabet)))


def language_plus(alphabet: Set) -> Iterable[Iterable]:
    """
    Return all words consisting of at least one letter that can be constructed
    from any combination of 0 or 1 uses of each letter in the alphabet.
    """
    return [word for word in combinations(alphabet) if word]


def remove_at_pos(s: Iterable, indices: Iterable[int]) -> str:
    """Remove the letters at positions in `indices`"""
    assert all(i >= 0 for i in indices), "Expected nonnegative indices"
    out = s
    for i in sorted(indices, reverse=True):
        out = out[:i] + out[i + 1:]
    return out


def unique(vec: List) -> Tuple[bool, Optional[Tuple]]:
    for i in range(len(vec)):
        for j in range(i+1, len(vec)):
            a, b = vec[i], vec[j]
            if a == b:
                return False, (a, b)
    return True, None


def approx_eq(a: T.Tensor, b: T.Tensor, threshold=10 ** -4) -> bool:
    inf_mask = T.logical_not(T.isinf(a))
    return T.equal(T.isposinf(a), T.isposinf(b)) and \
        T.equal(T.isneginf(a), T.isneginf(b)) and \
        T.all((a - b)[inf_mask].abs() <= threshold)

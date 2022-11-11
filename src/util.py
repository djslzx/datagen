import random
import numpy as np
import torch as T
from os import mkdir
import itertools as it
from typing import List, Tuple, Iterable, Optional, Set
import subprocess
from hashlib import md5


def md5_hash(s: str) -> str:
    return md5(s.encode()).hexdigest()


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
    if T.isposinf(a) and T.isposinf(b):
        return True
    if T.isneginf(a) and T.isneginf(b):
        return True
    return abs(a - b) <= threshold


def coinflip(p: float):
    assert 0 <= p <= 1, f"p is not a probability, p={p}"
    return random.random() < p


def try_mkdir(path: str):
    try:
        mkdir(path)
    except FileExistsError:
        pass


def random_balanced_brackets(n_bracket_pairs: int) -> List[str]:
    out = ['[', ']']
    for _ in range(n_bracket_pairs - 1):
        start_pos = random.randint(0, len(out) - 1)
        end_pos = random.randint(start_pos + 1, len(out))
        out.insert(start_pos, "[")
        out.insert(end_pos, "]")
    return out


def uniform_vec(n_elements: int) -> np.array:
    vec = np.random.rand(n_elements)
    return vec / vec.sum()


def gaussian_vec(n_elements: int) -> np.array:
    vec = np.abs(np.random.normal(
        loc=1/n_elements,
        scale=1/4,
        size=n_elements
    ))
    return vec / vec.sum()


def parens_are_balanced(s: str) -> bool:
    n_open = 0
    for c in s:
        if c == '[':
            n_open += 1
        elif c == ']':
            if n_open == 0:
                return False
            n_open -= 1
    return n_open == 0


def parse_braces(s: str) -> List[Tuple[int, int]]:
    pairs = []
    stack = []
    for i, c in enumerate(s):
        if c == '[':
            stack.append(i)
        elif c == ']':
            assert stack, "Unbalanced parentheses"
            # add the pair to the stack
            pairs.append((stack.pop(), i))
    assert not stack, "Found nonempty stack"
    return sorted(pairs)

import random
import numpy as np
from os import mkdir
import itertools as it
from typing import List, Tuple, Dict, Any, Iterable, Optional


def softplus(a: float, b: float) -> float:
    return a + np.log1p(np.exp(b - a))


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


def test_split_list():
    cases = [
        (["aa", "bb", "cc"], "x", [["aa", "bb", "cc"]]),
        (["aa", "bb", "cc"], "a", [["aa", "bb", "cc"]]),
        (["aa", "bb", "cc"], "aa", [["bb", "cc"]]),
        (["aa", "bb", "cc"], "bb", [["aa"], ["cc"]]),
        (["aa", "bb", "cc"], "cc", [["aa", "bb"]]),
    ]
    for s, t, y in cases:
        out = split_list(s, t)
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_split_list")


def language(iterable: Iterable[Any]) -> Iterable[Iterable[Any]]:
    """
    Return all the words in the language induced by the alphabet `iterable`
    """
    return it.chain.from_iterable(it.combinations(iterable, r=i+1)
                                  for i in range(len(iterable)))


def language_plus(iterable: Iterable[Any]) -> Iterable[Iterable[Any]]:
    return [word for word in language(iterable) if word]


def remove_from_string(s: str, indices: List[int]) -> str:
    """Remove the first occurrence of each letter in `letters` from `s`"""
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


def test_unique():
    cases = [
        ([1, 2, 3], True),
        ([1, 2, 3, 1], False),
        ([[1], [2], [3]], True),
        ([[1], [2], [1]], False),
    ]
    for x, y in cases:
        out, _ = unique(x)
        assert out == y, f"Expected {y}, got {out}"
    print(" [+] test_unique() passed")


def normalize(vec: List[float]) -> List[float]:
    assert all(x >= 0 for x in vec), "All entries should be nonnegative"
    norm = sum(vec)
    if norm == 0:
        m = len(vec)
        return [1 / m for x in vec]
    return [x / norm for x in vec]


def normalize_weights(distro: Dict[Any, List[float]]) -> Dict[Any, List[float]]:
    return {
        pred: normalize(weights)
        for pred, weights in distro.items()
    }


def approx_eq(a: float, b: float, threshold=10 ** -4) -> bool:
    return abs(a - b) < threshold


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


if __name__ == '__main__':
    # for i in range(10):
    #     print(gaussian_vec(5))
    # s = '[asdf[ddd]s[df]sdf][dsdf]'
    # for a, b in parse_braces(s):
    #     print(s[a:b+1])

    test_unique()
    test_split_list()

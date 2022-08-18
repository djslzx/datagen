import random
import numpy as np
from os import mkdir
from typing import List, Tuple


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
    for i in range(10):
        print(gaussian_vec(5))

    s = '[asdf[ddd]s[df]sdf][dsdf]'
    for a, b in parse_braces(s):
        print(s[a:b+1])

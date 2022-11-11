import pytest
from util import *


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


def test_language():
    cases = [
        ({"a", "b", "c"}, set(tuple(sorted(s))
                              for s in ["a", "b", "c", "ab", "bc", "ac", "abc"])),
    ]
    for alphabet, lang in cases:
        out = set(tuple(sorted(x)) for x in combinations(alphabet))
        assert lang == out, \
            f"Expected alphabet {alphabet} to generate\n{lang},\nbut got\n{out}"


def test_remove_at_pos():
    cases = [
        ("0123456789", [0], "123456789"),
        ("0123456789", [0, 1, 2, 3], "456789"),
        ("0123456789", [0, 9, 3], "1245678"),
    ]
    for x, indices, y in cases:
        out = remove_at_pos(x, indices)
        assert out == y, f"Expected {y} but got {out}"


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


def test_approx_eq():
    cases = [
        (1.0, 1.01, 0.1, True),
        (1.0, 1.01, 0.01, True),
        (1.0, 1.01, 0.001, False),
    ]









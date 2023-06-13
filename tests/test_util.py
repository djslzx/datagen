import numpy as np
import pytest
from util import *

def test_flatten():
    cases = [
        {}, {},
        {"a": 1}, {"a": 1},
        {"a": {"x": 1, "y": 2}}, {"a.x": 1, "a.y": 2},
        {"a": {"x": 1, "y": 2}, "b": 3}, {"a.x": 1, "a.y": 2, "b": 3},
        {"a": {"x": {"1": 1,
                     "2": 2},
               "y": 2},
         "b": 3},
        {"a.x.1": 1, "a.x.2": 2, "a.y": 2, "b": 3},
    ]
    for x, y in zip(cases[::2], cases[1::2]):
        out = flatten(x)
        assert y == out, f"Expected {y} but got {out}"


def test_split_endpoints():
    cases = [
        [0], [(0, 0)],
        [1], [(0, 1)],
        [1, 1], [(0, 1), (1, 2)],
        [1, 2, 1], [(0, 1), (1, 3), (3, 4)],
        [1, 2, 1, 5, 10], [(0, 1), (1, 3), (3, 4), (4, 9), (9, 19)],
    ]
    for x, y in zip(cases[::2], cases[1::2]):
        out = split_endpoints(x)
        assert y == out, f"Expected {y} but got {out}"


def test_batch():
    cases = [
        ([0, 1, 2, 3, 4, 5], 1), [[0], [1], [2], [3], [4], [5]],
        ([0, 1, 2, 3, 4, 5], 2), [[0, 1], [2, 3], [4, 5]],
        ([0, 1, 2, 3, 4, 5], 3), [[0, 1, 2], [3, 4, 5]],
        ([0, 1, 2, 3, 4, 5], 4), [[0, 1, 2, 3], [4, 5]],
        ([0, 1, 2, 3, 4, 5], 5), [[0, 1, 2, 3, 4], [5]],
        ([0, 1, 2, 3, 4, 5], 6), [[0, 1, 2, 3, 4, 5]],
        ([0, 1, 2, 3, 4, 5], 7), [[0, 1, 2, 3, 4, 5]],
    ]
    for args, ans in zip(cases[::2], cases[1::2]):
        out = list(batched(*args))
        assert out == ans, f"Expected {ans} but got {out}"


def test_param_tester():
    p = ParamTester({"a": [1, 2],
                     "b": [0, 1, 2],
                     "c": 0})
    configs = [
        {"a": 1, "b": 0, "c": 0},
        {"a": 2, "b": 0, "c": 0},
        {"a": 1, "b": 1, "c": 0},
        {"a": 2, "b": 1, "c": 0},
        {"a": 1, "b": 2, "c": 0},
        {"a": 2, "b": 2, "c": 0},
    ]
    for i, config in enumerate(p):
        assert configs[i] == config, f"Expected {configs[i]} on iter {i} but got {config}"

    # single config
    p = ParamTester({"a": 0, "b": 1, "c": [2]})
    config = {"a": 0, "b": 1, "c": 2}
    for out in p:
        assert out == config, f"Expected {config} but got {out}"


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


def test_approx_eq():
    cases = [
        (1.0, 1.01, 0.1, True),
        (1.0, 1.001, 0.01, True),
        (1.0, 1.01, 0.001, False),
        (1.0, 1.00001, 0.000000001, False),
    ]
    for a, b, thresh, y in cases:
        out = approx_eq(a, b, thresh)
        assert y == out, f"Expected ({a} == {b}, thresh={thresh}) == {y} but got {out}"


def test_vec_approx_eq():
    cases = [
        (1.0, 1.01, 0.1, True),
        (1.0, 1.001, 0.01, True),
        (1.0, 1.01, 0.001, False),
        (T.inf, T.inf, 0.1, True),
        (-T.inf, -T.inf, 0.1, True),
        (T.inf, -T.inf, 0.1, False),
        (-T.inf, T.inf, 0.1, False),
        ([1, 2, 3, 4], [1.1, 2.1, 3.1, 4.1], 1, True),
    ]
    for a, b, thresh, y in cases:
        for out in [vec_approx_eq(T.tensor(a), T.tensor(b), thresh),
                    vec_approx_eq(np.array(a), np.array(b), thresh),]:
            assert y == out, f"Expected ({a} == {b}, thresh={thresh}) == {y} but got {out}"


def test_plot_images_at_positions():
    n_images = 100
    image_size = 20
    images = np.stack([np.ones((image_size, image_size, 3)) * (((i * 7) % 100)/100)
                       for i in range(n_images)])
    positions = np.stack([np.array([i % 6, i // 6]) * image_size
                          for i in range(n_images)])
    plot_images_at_positions(images, positions)
    plt.show()


def test_add_border():
    cases = [
        (np.array([[1]]), 1,
         np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])),
        (np.array([[1]]), 2,
         np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])),
        (np.array([[1]]), 3,
         np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])),
        (np.array([[1, 2, 3],
                   [4, 5, 6]]),
         1,
         np.array([[0, 0, 0, 0, 0],
                   [0, 1, 2, 3, 0],
                   [0, 4, 5, 6, 0],
                   [0, 0, 0, 0, 0]])),
        (np.array([[1, 2, 3],
                   [4, 5, 6]]),
         2,
         np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 2, 3, 0, 0],
                   [0, 0, 4, 5, 6, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])),
    ]
    for x, c, y in cases:
        out = add_border(x, c)
        assert np.allclose(out, y), f"Expected {y} but got {out}"

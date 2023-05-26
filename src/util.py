from __future__ import annotations

from math import floor, sqrt, ceil
import numpy as np
import torch as T
import itertools as it
from typing import *
import matplotlib.pyplot as plt
import time
import sys
from os import mkdir
from einops import reduce


def flatten(nested_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested dictionary of string keys into a single-level dictionary where
    nested keys are concatenated like so:
    a: {b: 1, c: 2} => {a.b: 1, a.c: 2}
    """
    out = {}
    for k, v in nested_dict.items():
        if isinstance(v, Dict):
            for vk, vv in flatten(v).items():
                out[f"{k}.{vk}"] = vv
        else:
            out[k] = v
    return out


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


class ParamTester:
    """
    Streamline iteration through sets of parameter values.

    Given a set of parameters, each with a list of possible values,
    iterate through each of the possible combinations of parameters
    using the order of parameters given (so we can decide which ones
    we want to test first) and the order of values given for each
    parameter.
    """
    def __init__(self, params: Dict[str, List[Any] | Any]):
        self.params = {
            k: vs if isinstance(vs, list) else [vs]
            for k, vs in params.items()
        }

    def __iter__(self):
        # reverse so we get the right prioritization
        for combo in it.product(*reversed(self.params.values())):
            yield {
                k: v
                for k, v in zip(reversed(self.params.keys()), combo)
            }


def split_endpoints(lengths: List[int]) -> List[Tuple[int, int]]:
    splits = np.cumsum([0] + lengths)
    return list(zip(splits[:-1], splits[1:]))


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


def pad_array(arr: np.ndarray, batch_size: int) -> np.ndarray:
    r = len(arr) % batch_size
    if r != 0:
        return np.concatenate((arr, np.empty(batch_size - r, dtype=object)))
    else:
        return arr


def batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    assert batch_size > 0
    b = []
    for x in iterable:
        if len(b) == batch_size:
            yield b
            b = [x]
        else:
            b.append(x)
    if b:  # leftover elts
        yield b


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


class Timing(object):
    """
    Use Timing blocks to time blocks of code.
    Adapted from DreamCoder.
    """
    def __init__(self, msg: str, file=sys.stdout, suppress_start=False):
        self.msg = msg
        self.file = file
        self.suppress_start = suppress_start

    def __enter__(self):
        if not self.suppress_start:
            print(f"{self.msg}...", file=self.file)
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.start
        print(f"{self.msg} took {dt:.4e} seconds", file=self.file)

    def time(self) -> float:
        return time.time() - self.start


def cut_ext(filename: str) -> str:
    return filename[:filename.rfind(".")]


def find_closing_bracket(s: str, brackets="[]") -> int:
    assert brackets in {"[]", "()", "{}"}
    c_open, c_close = brackets
    assert s[0] != c_open, f"Expected string to skip open bracket, but received: {s}"
    n_brackets = 0
    for i, c in enumerate(s):
        if c == c_open:
            n_brackets += 1
        elif c == c_close:
            if n_brackets == 0:
                return i
            else:
                n_brackets -= 1
    raise ValueError(f"Mismatched brackets in {s}")


def plot(imgs: List[np.array],
         shape: Optional[Tuple[int, int]] = None,
         labels: Optional[List[str]] = None,
         title="",
         fontsize=6,
         saveto=None):  # pragma: no cover
    # infer reasonable shape if none given
    if shape is None:
        n = len(imgs)
        height = floor(sqrt(n))
        width = ceil(n / height)
        shape = (height, width)

    assert len(imgs) <= shape[0] * shape[1], f"Received {len(imgs)} with shape {shape}"
    assert labels is None or len(imgs) == len(labels), f"Received {len(imgs)} images and {len(labels)} labels"

    # convert image to floats if needed
    if isinstance(imgs[0], np.ndarray) and imgs[0].dtype != float:
        imgs = [img.astype(float) for img in imgs]

    fig, ax = plt.subplots(*shape, figsize=(1.28 * shape[0] + 0.5,
                                            1.28 * shape[1] + 0.5))
    if shape == (1, 1):
        ax.imshow(imgs[0][0])
        if labels is not None:
            ax.set_title(labels[0], pad=3, fontsize=fontsize)
    else:
        # clear axis ticks
        for axis in ax.flat:
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

        # plot bitmaps
        axes: List[plt.Axes] = ax.flat
        for i, img in enumerate(imgs):
            axis = axes[i]
            axis.imshow(img[0])  # remove color dim
            if labels is not None:
                axis.set_title(labels[i], pad=3, fontsize=fontsize)

    fig.suptitle(title, fontsize=fontsize)
    plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
    if labels is not None:
        plt.subplots_adjust(wspace=0, hspace=0.2)
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
    if saveto:
        plt.savefig(saveto)
    else:
        plt.show()
    plt.close()


def stack_repeat(array: np.ndarray, k) -> np.ndarray:
    """Stacks `k` copies of `array`"""
    return np.repeat(array[None, ...], k, axis=0)


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


def approx_eq(a: float, b: float, threshold=10 ** -6) -> bool:
    return abs(a - b) <= threshold


def vec_approx_eq(a: T.Tensor | np.ndarray, b: T.Tensor | np.ndarray, threshold=10 ** -4) -> bool:
    if isinstance(a, np.ndarray):
        a = T.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = T.from_numpy(b)

    inf_mask = T.logical_not(T.isinf(a))
    return T.equal(T.isposinf(a), T.isposinf(b)) and \
        T.equal(T.isneginf(a), T.isneginf(b)) and \
        T.all(T.abs((a - b)[inf_mask]) <= threshold)


def try_mkdir(dir: str):
    try:
        f = open(dir, "r")
        f.close()
    except FileNotFoundError:
        print(f"{dir} directory not found, making dir...", file=sys.stderr)
        mkdir(dir)
    except IsADirectoryError:
        pass

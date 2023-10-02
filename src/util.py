from __future__ import annotations
import json
from math import floor, sqrt, ceil
from pprint import pp
import numpy as np
import torch as T
import itertools as it
from typing import *
import matplotlib.pyplot as plt
import time
import sys
from os import mkdir
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from func_timeout import func_timeout, FunctionTimedOut
import openai.error
from adjustText import adjust_text
from datetime import datetime


def timestamp():
    return datetime.now().isoformat()


class IdGen:
    """
    A simple ID generator
    """

    def __init__(self):
        self.id = 0

    def next(self):
        self.id += 1
        return self.id


def load_jsonl(filename: str) -> List[dict]:
    with open(filename, "r") as f:
        out = []
        for line in f.readlines():
            try:
                d = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                print(f"Failed to decode {line}")
                raise e
            out.append(d)
    return out


def pp_jsonl(filename: str, skip=1):
    with open(filename, "r") as f:
        for line in f.readlines()[::skip]:
            pp(json.loads(line))


def prompt_openai_with_exp_backoff(f, *args):
    backoff = 1
    while True:
        try:
            func_timeout(1000, f, args)
        except (openai.error.RateLimitError, openai.error.Timeout):
            print(f"Exceeded rate limit, blocking {backoff}s", openai.api_key)
            time.sleep(backoff)
            backoff *= 2
        except FunctionTimedOut:
            print(f"Timed out, blocking {backoff}s", openai.api_key)
            time.sleep(backoff)
            backoff *= 2
        except (openai.error.APIError, openai.error.APIConnectionError, openai.error.ServiceUnavailableError):
            print("openai.error.APIError, blocking 10s")
            time.sleep(10)
        except openai.error.InvalidRequestError as e:
            print("openai.error.InvalidRequestError", e)
            time.sleep(10)


def dict_to_text(d: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in d.items())


def invert_array(x: np.ndarray) -> np.ndarray:
    """
    Treating the array `x` as a function mapping indices to values,
    return the inverse function as an array
    """
    assert x.dtype == int
    assert np.array_equal(np.sort(x), np.arange(len(x)))
    y = np.zeros_like(x)
    y[x] = np.arange(len(x))
    return y


def plot_labeled_points(x, y, labels: List, title=None, **kwargs):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.scatter(x, y)
    text_labels = []
    for i, label in enumerate(labels):
        text_labels.append(ax.text(x[i], y[i], label, **kwargs))
    adjust_text(text_labels,
                force_text=0.05,
                arrowprops=dict(arrowstyle="-", color='0.6', alpha=0.5))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale()
    if title:
        plt.title(title)
    return ax


def center_image(image: np.ndarray) -> np.ndarray:
    # find the center of mass
    x, y, _ = np.nonzero(image)
    x_mid = (x.min() + x.max()) // 2
    y_mid = (y.min() + y.max()) // 2

    # translate the image so that the center of mass is at the center of the image
    x_offset = (image.shape[0] - 1) // 2 - x_mid
    y_offset = (image.shape[1] - 1) // 2 - y_mid

    return np.roll(image, (x_offset, y_offset), axis=(0, 1))


def bkg_black_to_white(image: np.ndarray) -> np.ndarray:
    return np.where(image > 0, image, 255)


def add_border(image: np.ndarray, thickness=1) -> np.ndarray:
    shape = (image.shape[0] + thickness * 2,
             image.shape[1] + thickness * 2,
             *image.shape[2:])
    out = np.zeros(shape, dtype=image.dtype)
    out[thickness:-thickness, thickness:-thickness] = image
    return out


def fig_images_at_positions(images: np.ndarray, positions: np.ndarray) -> plt.Figure:
    # translate positions so that the min x and y positions are 0
    positions[:, 0] -= positions[:, 0].min()
    positions[:, 1] -= positions[:, 1].min()

    # find the max x and y positions
    i_xlim = positions[:, 0].argmax()
    i_ylim = positions[:, 1].argmax()
    xlim = images[i_xlim].shape[0] + positions[i_xlim, 0]
    ylim = images[i_ylim].shape[1] + positions[i_ylim, 1]

    fig = plt.figure(figsize=(xlim / 100, ylim / 100))
    for image, pos in zip(images, positions):
        x, y = pos
        fig.figimage(image, xo=x, yo=y, origin='upper')
    return fig


def imscatter(images: np.ndarray, positions: np.ndarray, zoom=1, figsize=(10, 10), alpha=0.5):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    for image, position in zip(images, positions):
        im = OffsetImage(image, zoom=zoom, alpha=alpha)
        ab = AnnotationBbox(im, position, xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.update_datalim(positions)
    ax.autoscale()
    return ax


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


def plot_image_grid(imgs: List[np.ndarray],
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

    fig, ax = plt.subplots(*shape, figsize=(1.28 * shape[1] + 0.5,
                                            1.28 * shape[0] + 0.5))
    if shape == (1, 1):
        ax.imshow(imgs[0])
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
            axis.imshow(img)
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


def split_list(s: List[str], t: str) -> List[List[str]]:
    out = []
    while True:
        try:
            i = s.index(t)
            r, s = s[:i], s[i + 1:]
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
        for j in range(i + 1, len(vec)):
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

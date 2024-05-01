from __future__ import annotations
import json
from math import floor, sqrt, ceil
from pprint import pp
import re
from PIL import Image
import numpy as np
import pandas as pd
import torch as T
import itertools as it
from typing import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from einops import rearrange
import time
import sys
from os import mkdir, listdir, path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
from dataclasses import dataclass
from adjustText import adjust_text
from io import BytesIO


def scatterplot_image(coords: np.ndarray, figsize: int, **kwargs) -> np.ndarray:
    assert coords.ndim == 2, f"Expected 2d tensor, got {coords.ndim}d"
    assert coords.shape[1] == 2, f"Expected 2d coordinates, got {coords.shape[1]}d"

    # Create scatterplot
    fig = plt.Figure(figsize=(figsize, figsize))
    ax = fig.subplots()
    ax.scatter(coords[:, 0], coords[:, 1], s=2, **kwargs)
    plt.tight_layout()

    # Render the plot to a buffer
    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)

    # Convert buffer to a PIL image, then to a numpy array
    img = Image.open(buf)
    arr = np.asarray(img)
    plt.close()

    return arr


def combine_images_row(images: np.ndarray) -> np.ndarray:
    """
    Combine multiple images into a single image.  Given an array of images of shape [b, h, w, c],
    produce a single image with dimensions [h, b * w, c].
    """
    return rearrange(images, 'b h w c -> h (b w) c')


def combine_images_square(images: np.ndarray) -> np.ndarray:
    """
    Combine multiple images into a single image.  Given an array of images of shape [b, h, w, c],
    produce a single image with dimensions [h', w', c], where h' and w' are close to sqrt(b).
    """
    b, h, w, c = images.shape
    s = int(np.ceil(np.sqrt(b)))

    # Pad with zeros in case we don't have enough images to cover the square grid
    image = np.zeros((s ** 2, h, w, c), dtype=images.dtype)
    image[:b, :, :, :] = images

    return rearrange(image, '(sh sw) h w c -> (sh h) (sw w) c', sh=s, sw=s)


def animate_points(
        frames: List[Tuple[int, np.ndarray, np.ndarray]],
        title: str,
        background: Optional[np.ndarray] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        delay=200
):
    colors = np.array(["r", "g", "b", "y", "o", "b", "w"])
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect('equal')

    if background is not None:
        ax.imshow(background, extent=xlim + ylim, alpha=0.5)

    def update(frame):
        t, coords, groupings = frame
        assert coords.shape[1] == 2, f"Expected 2D points, got {coords.shape}"

        ax.set_title(f"{title}, frame: {t}")
        ax.title.set_fontsize(8)
        scatter.set_offsets(coords)
        scatter.set_color(colors[groupings])

        if xlim is None:
            ax.set_xlim(min(p[0] for p in coords), max(p[0] for p in coords))
        if ylim is None:
            ax.set_ylim(min(p[1] for p in coords), max(p[1] for p in coords))

        return scatter,

    anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=delay)
    plt.close()
    return anim


def plot_v_subplots(data: List[dict], keys: List[str]):
    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, 1, figsize=(12, 2 * n_keys))

    for ax, key in zip(axes, keys):
        ax.set_title(key)
        ax.plot([x[key] for x in data], label=key)
        if key.startswith("log"):
            ax.set_yscale("symlog")
        # if key.startswith("sparsity"):
        #     ax.set_ylim(0, 1)
        #     ax.set_ylabel("sparsity")

    plt.tight_layout()
    return fig


def plot_square_subplots(images: np.ndarray, title=""):
    assert images.ndim == 3, f"Expected 3d array, got {images.ndim}d"

    n_images = len(images)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    return fig


def count_calls(func):
    """
    A decorator that counts the number of times a function is called.
    """

    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


def ls_subdirs(dir_path: str) -> List[str]:
    return [
        d for d in listdir(dir_path)
        if path.isdir(path.join(dir_path, d))
    ]


@dataclass
class KVItem:
    id: str
    key: str
    value: Any

    @staticmethod
    def from_dict(d: Dict) -> List["KVItem"]:
        id = d["id"]
        return [
            KVItem(id=id, key=key, value=value)
            for key, value in d.items()
            if key != "id"
        ]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
        }


def test_from_dict():
    cases = [
        {"id": 0,
         "a": 1,
         "b": 2},
        [KVItem(0, "a", 1),
         KVItem(0, "b", 2)],
    ]
    for x, y in zip(cases[::2], cases[1::2]):
        assert KVItem.from_dict(x) == y


def test_json_dump():
    import json
    cases = [
        KVItem(0, "a", 1), {"id": 0, "key": "a", "value": 1},
        KVItem(0, "b", 2), {"id": 0, "key": "b", "value": 2},
    ]
    for x, y in zip(cases[::2], cases[1::2]):
        assert json.dumps(x) == y


def isnan(x):
    try:
        return np.isnan(x)
    except TypeError:
        return False


def incrementally_save_jsonl(it, filename: str, quiet=False) -> pd.DataFrame:
    if not filename.endswith(".jsonl"):
        filename += ".jsonl"
    with open(filename, "w") as f:
        for x in it:
            line = json.dumps(x, indent=None)
            if not quiet:
                print(line)
            f.write(line + "\n")
    return pd.read_json(filename, lines=True)


def pad_list(xs: List, n: int, nil: Any) -> List:
    """
    Make xs into a list of length n.  Remove entries if len(xs) > n, add nils if len(xs) < n.
    """
    if len(xs) < n:
        return xs + [nil] * (n - len(xs))
    else:
        return xs[:n]


def split_tests(text: str):
    """Split a code block full of test functions into a list of test functions"""
    text = strip_markdown(text)

    # split by decls
    blocks = ["def test_" + x
              for x in text.split("def test_")[1:]
              if x.strip()]
    out = []
    # only keep indented lines; stop at first non-indented line
    for block in blocks:
        lines = block.split("\n")
        block_text = lines[0] + "\n"
        for line in lines[1:]:
            if line.startswith("    "):
                block_text += line + "\n"
            else:
                break
        out.append(block_text)
    return out


def split_py_markdown(text: str) -> List[str]:
    """
    Split a string formatted as "```python<x1>``` ... ```python<xn>```" into
    the list of strings [x1, ..., xn].
    """
    # add an end marker for the last block if it doesn't have one
    n_starts = len(re.findall(r"```python", text))
    n_ends = len(re.findall(r"```", text))
    if n_starts > n_ends:
        text += "```"
    return [x.strip() for x in re.findall(r"```python\s*([^`]*)\s*```", text)]


def strip_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        return "\n".join(text.split("\n")[1:-1])
    else:
        return text


def test_strip_markdown():
    cases = [
        ("""
         ```python
         x + 1
         ```
         """,
         "         x + 1"),
        ("""
         
         ```python
         x + 1
         ```
         """,
         "         x + 1"),
        ("""
         ```
         x + 2
         ```
         
         
         """,
         "         x + 2"),
        ("x", "x"),
    ]
    for x, y in cases:
        out = strip_markdown(x)
        assert out == y, f"Expected {y} but got {out}"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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
        mkdir(dir)
    except IsADirectoryError:
        pass

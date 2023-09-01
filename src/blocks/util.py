import torch as T
import torch.nn.functional as F
from math import floor, ceil
import random
from collections import namedtuple
from typing import List, Tuple, Optional

Rect = namedtuple(typename='Rect', field_names=['x', 'y', 'w', 'h'])


def transpose_all(rects: List[Rect]) -> List[Rect]:
    return [
        Rect(x=r.y, y=r.x, w=r.h, h=r.w)
        for r in rects
    ]


def pad_mat(t: T.Tensor, h: int, w: int, padding_token: int = 0) -> T.Tensor:
    """Pad t to height h and width w"""
    assert (dims := len(t.shape)) == 2, \
        f'Expected a 2-dimensional tensor, but got a {dims}-dimensional tensor'
    assert h >= t.size(0) and w >= t.size(1), \
        f'Padded dimensions are smaller than tensor size: h={h}, w={w}, t.shape={t.shape}'
    return F.pad(t, (0, w - t.size(1), 0, h - t.size(0)), value=padding_token)


def fill_height(t: T.Tensor) -> int:
    assert len(t.shape) >= 2
    return (t.sum(dim=1) > 0).sum().item()


def fill_width(t: T.Tensor) -> int:
    assert len(t.shape) >= 2
    return (t.sum(dim=0) > 0).sum().item()


def uniq(l: List) -> List:
    joined = []
    for x in l:
        if x not in joined:
            joined.append(x)
    return joined


def unwrap_tensor(t):
    if isinstance(t, T.Tensor):
        return t.item()
    else:
        return t


def wrap_tensor(t):
    if isinstance(t, T.Tensor):
        return t
    else:
        return T.Tensor(t)


def pad(v, length: int, value: int):
    return F.pad(v, pad=(0, length - len(v)), value=value)


def filter_top_p(v, p=0.95):
    x = v.clone()
    values, indices = T.sort(x, descending=True)
    sums = T.cumsum(values, dim=-1)
    mask = sums >= p
    # right-shift indices to keep first sum >= p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    # filter out elements in v
    for b in range(x.shape[0]):
        x[b, indices[b, mask[b]]] = 0
    return x


def shuffled(it):
    random.shuffle(it)
    return it


def is_prefix(l1, l2):
    if len(l1) > len(l2):
        return False
    for x1, x2 in zip(l1, l2):
        if x1 != x2:
            return False
    return True


def to_toks(s):
    def tonum(s):
        try:
            return int(s)
        except:
            return s

    return [tonum(c) for c in s]


def clamp(x, lo, hi):
    assert hi >= lo, f'hi < lo: hi={hi}, lo={lo}'
    if x > hi: return hi
    if x < lo: return lo
    return x


def chunk(n, k):
    """
    Returns a partition of n items into k chunks.

    Output: a list of lengths, where the i-th length is the length of the i-th chunk
    e.g. chunk(10, 3) --> [4, 3, 3]
    """
    return [ceil(n / k) if i < n % k else floor(n / k)
            for i in range(k)]


def chunks(l, k, n):
    size = len(l)
    for i in range(n):
        start = (i * k) % size
        end = (start + k) % size
        yield l[start:end] if start < end else l[start:] + l[:end]


def chunk_pairs(l, k, n):
    """
    Iterator over n k-elt chunks of list l, yielding pairs of adjacent chunks
    """
    size = len(l)
    for i in range(n):
        start = (i * k) % size
        mid = (start + k) % size
        end = (mid + k) % size

        yield (l[start:mid] if start < mid else l[start:] + l[:mid],
               l[mid:end] if mid < end else l[mid:] + l[:end])


def img_to_tensor(lines, w=-1, h=-1):
    """Converts a list of strings into a float tensor"""
    if not lines: return T.Tensor([])

    lines_l = len(lines)
    lines_w = max(len(line) for line in lines)

    if h == -1: h = lines_l
    if w == -1: w = lines_w

    def cell(x, y):
        if y < lines_l and x < lines_w:
            try:
                return int(lines[y][x])
            except ValueError:
                return int(lines[y][x] == '#')
            except IndexError:
                return 0
        else:
            return 0

    return T.tensor([[cell(x, y) for x in range(w)]
                     for y in range(h)]).float()


def tensor_to_pts(tensor):
    return [(x, y) for x, y in tensor.nonzero().tolist()]


def unzip(l):
    return tuple(list(x) for x in zip(*l))


def make_bitmap(f, W, H):
    return T.tensor([[f((x, y))
                      for x in range(W)]
                     for y in range(H)]).float()


def split(l, pred):
    """
    Split l into two lists `a` and `b`, where
    all elts of `a` satisfy `pred` and all elts of `b` do not
    """
    sat, unsat = [], []
    for x in l:
        if pred(x):
            sat.append(x)
        else:
            unsat.append(x)
    return sat, unsat

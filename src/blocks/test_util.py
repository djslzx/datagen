from blocks.util import *


def test_fill_measure():
    cases = [
        (img_to_tensor(["###  "]), 1, 3),
        (img_to_tensor(["###  ",
                        "##   "]), 2, 3),
        (img_to_tensor([" ####",
                        "##   "]), 2, 5),
    ]
    for mat, height, width in cases:
        h = fill_height(mat)
        w = fill_width(mat)
        assert height == h and width == w, \
            f"[-] failed test_fill_measure:\n" \
            f"  Expected height={height} and width={width}, but got {h, w}"


def test_pad_tensor():
    cases = [
        (T.Tensor([[1, 1],
                   [2, 1]]),
         3, 4, 0,
         T.Tensor([[1, 1, 0, 0],
                   [2, 1, 0, 0],
                   [0, 0, 0, 0]])),
        (T.Tensor([[1, 1, 2],
                   [2, 1, 3]]),
         5, 3, 9,
         T.Tensor([[1, 1, 2],
                   [2, 1, 3],
                   [9, 9, 9],
                   [9, 9, 9],
                   [9, 9, 9]])),
    ]
    for t, h, w, tok, ans in cases:
        out = pad_mat(t, h, w, tok)
        assert T.equal(out, ans), f"Expected {ans}, but got {out}"

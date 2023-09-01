from blocks.ant import *


def test_connected():
    W, H = 8, 8
    tests = [
        ([], True),
        (["____",
          "____",
          "_#__",
          "____"], True),
        (["_#__",
          "____",
          "_#__",
          "____"], False),
        (["__#__",
          "_#__",
          "###_",
          "____"], True),
        (["_#__",
          "_##_",
          "____",
          "__##"], False),
    ]
    for img, ans in tests:
        t = util.img_to_tensor(img, w=W, h=H)
        pts = util.tensor_to_pts(t)
        o = connected(pts)
        assert o == ans, \
            f"Classified {t} as {o}, expected {ans}"


def test_classify():
    W, H = 8, 8
    tests = [
        (["____",
          "____",
          "_#__",
          "____"], 'Point'),
        (["____",
          "____",
          "_#__",
          "_#__"], 'Line'),
        (["____",
          "____",
          "_###",
          "____"], 'Line'),
        (["_#__",
          "__#_",
          "___#",
          "____"], 'Line'),
        (["_##_",
          "_##_",
          "_##_",
          "____"], 'Rect'),
        (["_##_",
          "_#__",
          "_##_",
          "____"], 'Sprite'),
        (["_#__",
          "_#__",
          "_##_",
          "____"], 'Sprite'),
        (["_#__",
          "_##_",
          "_#__",
          "____"], 'Sprite'),
    ]
    for img, ans in tests:
        t = util.img_to_tensor(img, w=W, h=H)
        pts = util.tensor_to_pts(t)
        o = classify(pts)
        assert o == ans, \
            f"Classified {t} as {o}, expected {ans}"

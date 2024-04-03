"""
Languages for testing novelty search with "point programs".
Programs are 2D points, and novelty is the Euclidean distance between them.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats

from grammar import Grammar
from featurizers import Featurizer
from lang.tree import Language, Tree
import util


class RealPoint(Language):
    grammar = r"""
        point: "(" num "," num ")" -> point
        num: NUMBER                -> pos
           | "-" NUMBER            -> neg

        %import common.NUMBER
        %import common.WS
        %ignore WS
    """
    types = {
        "point": ["Real", "Real", "Point"],
        "real": ["Real"],
    }

    def __init__(self, lim: Optional[float] = None, std: float = 1):
        super().__init__(
            parser_grammar=RealPoint.grammar,
            parser_start="point",
            root_type="Point",
            model=None,
            featurizer=PointFeaturizer(ndims=2)
        )
        assert lim is None or lim > 0
        self.lim = lim
        self.std = std
        self.x_distribution = GaussianSampler(0, self.std)
        self.y_distribution = GaussianSampler(0, self.std)

    def make_point(self, x: float, y: float) -> Tree:
        return self.parse(f"({x}, {y})")

    def none(self) -> Any:
        return np.array([0, 0])

    def simplify(self, t: Tree) -> Tree:
        raise NotImplementedError

    def _parse_child(self, t: Tree) -> float:
        if t.value == "pos":
            return float(t.children[0].value)
        elif t.value == "neg":
            return -float(t.children[0].value)

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> Any:
        assert isinstance(t, Tree), f"Expected to eval Tree, but got {t}"
        x, y = [self._parse_child(c) for c in t.children]
        return np.array([x, y])

    def fit(self, corpus: List[Tree], alpha=0.):
        xs, ys = [], []
        for tree in corpus:
            x, y = self.eval(tree)
            xs.append(x)
            ys.append(y)
        if len(xs) == 1:
            self.x_distribution = GaussianSampler(xs[0], self.std)
            self.y_distribution = GaussianSampler(ys[0], self.std)
        else:
            self.x_distribution = stats.gaussian_kde(xs, bw_method=self.std)
            self.y_distribution = stats.gaussian_kde(ys, bw_method=self.std)

    def _sample_coords(self) -> Tuple[int, int]:
        x = self.x_distribution.resample(1).item()
        y = self.y_distribution.resample(1).item()
        if self.lim is not None:
            x = min(self.lim, max(-self.lim, x))
            y = min(self.lim, max(-self.lim, y))
        return x, y

    def sample(self) -> Tree:
        x, y = self._sample_coords()
        return self.parse(f"({x}, {y})")

    def log_probability(self, t: Tree) -> float:
        x, y = self.eval(t)
        return self.x_distribution.logpdf(x) + self.y_distribution.logpdf(y)

    @property
    def str_semantics(self) -> Dict:
        return {
            "point": lambda x, y: f"({x}, {y})",
            "pos": lambda n: f"{n}",
            "neg": lambda n: f"-{n}",
        }


class NatPoint(Language):
    grammar = r"""
        point: "(" nat "," nat ")" -> point
        nat: "one"                 -> one
           | "inc" nat             -> inc
           
        %import common.WS
        %ignore WS
    """
    types = {
        "point": ["Nat", "Nat", "Point"],
        "inc": ["Nat", "Nat"],
        "one": ["Nat"],
    }

    def __init__(self):
        super().__init__(
            parser_grammar=NatPoint.grammar,
            parser_start="point",
            root_type="Point",
            model=Grammar.from_components(NatPoint.types, gram=2),
            featurizer=PointFeaturizer(ndims=2)
        )

    def simplify(self, t: Tree) -> Tree:
        raise NotImplementedError

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> Any:
        if t.value == "one":
            return 1
        elif t.value == "inc":
            return 1 + self.eval(t.children[0])
        elif t.value == "point":
            x, y = [self.eval(c) for c in t.children]
            return np.array([x, y])
        raise ValueError(f"Tree should be in {list(NatPoint.types.keys())}, but got {t.value}")

    @property
    def str_semantics(self) -> Dict:
        return {
            "point": lambda x, y: f"({x}, {y})",
            "inc": lambda n: f"(inc {n})",
            "one": lambda: "1",
        }

    def none(self) -> Any:
        return np.array([0, 0])


class RealMaze(RealPoint):
    def __init__(self, mask: np.ndarray, cell_size=1):
        """
        mask: defines maze shape, where 1s are walls
        cell_size: size of each cell in the mask, 1 by default
        """
        assert mask.ndim == 2
        n_rows, n_cols = mask.shape
        assert n_rows == n_cols

        super().__init__(lim=mask.shape[0] * cell_size)

        self.mask = mask
        self.n = n_rows

    def _is_valid_point(self, x: float, y: float) -> bool:
        """
        Check if the point (x, y) is in a valid position in the maze
        """
        if x < 0 or y < 0:
            return False

        x, y = int(x), int(y)
        in_bounds = 0 <= x < self.n and 0 <= y < self.n
        in_wall = bool(self.mask[y, x])

        return in_bounds and not in_wall

    def sample(self) -> Tree:
        # keep resampling until we get something that's within the bounds of the maze
        x, y = super()._sample_coords()
        while not self._is_valid_point(x, y):
            x, y = super()._sample_coords()
        return self.parse(f"({x}, {y})")


def str_to_float_mask(str_mask: List[str]) -> np.ndarray:
    """
    Convert a list of strings into a binary float array,
    where any '#' chars are interpreted as 1, any other char
    is interpreted as 0.

    Change from row-major form (row, col) to (x, y), where
      x = c
      y = n - r
    i.e., a single 90deg clockwise rotation.
    """
    m = np.array([
        [
            float(c == "#")
            for c in line
        ] for line in str_mask
    ])
    # rotate once clockwise
    return np.rot90(m, axes=(1, 0))


def test_str_to_float_mask():
    s = ["#_",
         "__"]
    m = str_to_float_mask(s)
    assert m[0, 0] == 0.
    assert m[0, 1] == 1.
    assert m[1, 0] == 0.
    assert m[1, 1] == 0.

    s = ["_#",
         "__"]
    m = str_to_float_mask(s)
    assert m[0, 0] == 0.
    assert m[0, 1] == 0.
    assert m[1, 0] == 0.
    assert m[1, 1] == 1.

    s = ["__",
         "#_"]
    m = str_to_float_mask(s)
    assert m[0, 0] == 1.
    assert m[0, 1] == 0.
    assert m[1, 0] == 0.
    assert m[1, 1] == 0.

    s = ["__",
         "_#"]
    m = str_to_float_mask(s)
    assert m[0, 0] == 0.
    assert m[0, 1] == 0.
    assert m[1, 0] == 1.
    assert m[1, 1] == 0.


def test_maze():
    str_mask = [
        "________",
        "_######_",
        "_#______",
        "_#____#_",
        "_#_####_",
        "_#____#_",
        "_###__#_",
        "___#__#_",
    ]
    mask = str_to_float_mask(str_mask)
    maze = RealMaze(mask)
    print(maze.mask)
    for i in range(10):
        tree = maze.sample()
        print(tree)


class PointFeaturizer(Featurizer):

    def __init__(self, ndims: int):
        self.ndims = ndims

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        return np.array(batch)

    @property
    def n_features(self) -> int:
        return self.ndims


class GaussianSampler:
    def __init__(self, mean, std):
        self.dist = stats.norm(loc=mean, scale=std)

    def resample(self, *args):
        # Generate a sample from the Gaussian distribution
        sample = self.dist.rvs()
        return sample

    def logpdf(self, x):
        return self.dist.logpdf(x)


def test_RealPoint_parse():
    cases = [
        "(0.1, 0.2)", [0.1, 0.2],
        "(0.1, -0.2)", [0.1, -0.2],
        "(-0.1, 0.2)", [-0.1, 0.2],
        "(-0.1, -0.2)", [-0.1, -0.2],
    ]
    RP = RealPoint()
    for x, y in zip(cases[::2], cases[1::2]):
        t = RP.parse(x)
        out = RP.eval(t)
        assert np.array_equal(out, y), f"Expected {y} but got {out}"


if __name__ == "__main__":
    G = NatPoint()
    xs = [
        "(1, 2)",
        "(1.2, 2)",
    ]
    for x in xs:
        t = G.parse(x)
        y = G.eval(t)
        print(f"{x} => {t} => {y}")

    for _ in range(10):
        t = G.sample()
        print(f"{t}")

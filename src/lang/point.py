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

    def __init__(
            self,
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            std: float = 1
    ):
        super().__init__(
            parser_grammar=RealPoint.grammar,
            parser_start="point",
            root_type="Point",
            model=None,
            featurizer=PointFeaturizer(ndims=2)
        )
        self.xlim = xlim
        self.ylim = ylim
        self.std = std
        self.x_distribution = GaussianSampler([0], self.std)
        self.y_distribution = GaussianSampler([0], self.std)

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

    def _trees_to_coords(self, trees: List[Tree]) -> np.ndarray:
        coords = []
        for tree in trees:
            x, y = self.eval(tree)
            coords.append((x, y))
        return np.array(coords)

    def fit(self, corpus: List[Tree], alpha=0.):
        coords = self._trees_to_coords(corpus)
        xs = coords[:, 0]
        ys = coords[:, 1]

        # update distributions
        self.x_distribution = GaussianSampler(xs, self.std)
        self.y_distribution = GaussianSampler(ys, self.std)

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        if val < lo:
            return lo
        elif val > hi:
            return hi
        else:
            return val

    def _sample_coords(self) -> Tuple[float, float]:
        x = self.x_distribution.sample()
        y = self.y_distribution.sample()
        if self.xlim is not None:
            x = RealPoint._clamp(x, *self.xlim)
        if self.ylim is not None:
            y = RealPoint._clamp(y, *self.ylim)
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
    def __init__(self, str_mask: List[str], std=1):
        """
        mask: defines maze shape, where 1s are walls
        cell_size: size of each cell in the mask, 1 by default
        """
        mask = str_to_float_mask(str_mask)
        assert mask.ndim == 2
        self.mask = self._rc_to_xy(mask)
        self.allowed_cells = np.ones_like(self.mask)
        nx, ny = self.mask.shape
        super().__init__(xlim=(0, nx), ylim=(0, ny), std=std)
        self.update_allowed(coords=np.zeros((1, 2)))

    @property
    def background(self) -> np.ndarray:
        """
        Return the mask as a bitmap
        """
        return self._xy_to_rc(self.mask)

    def _is_valid_point(self, x: float, y: float) -> bool:
        """
        Check if the point (x, y) is in a valid position in the maze
        """
        if x < 0 or y < 0:
            return False

        xlo, xhi = self.xlim
        ylo, yhi = self.ylim
        return ((xlo <= x < xhi and ylo <= y < yhi)
                and not bool(self.mask[int(x), int(y)])
                and bool(self.allowed_cells[int(x), int(y)]))

    def update_allowed(self, coords: np.ndarray):
        assert coords.shape[1] == 2, f"Expected 2d coords [n, 2], but got {coords.shape}"

        self.allowed_cells = np.zeros_like(self.mask)
        xlo, xhi = self.xlim
        ylo, yhi = self.ylim

        occupied_coords = {
            (int(x), int(y))
            for x, y in coords
        }
        for x, y in occupied_coords:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if xlo <= x + dx < xhi and ylo <= y + dy < yhi:
                        self.allowed_cells[x + dx, y + dy] = 1

    def fit(self, corpus: List[Tree], alpha=0.):
        super().fit(corpus, alpha)
        coords = self._trees_to_coords(corpus)
        self.update_allowed(coords)

    def sample(self) -> Tree:
        # keep resampling until we get something that's within the bounds of the maze
        x, y = super()._sample_coords()
        while not self._is_valid_point(x, y):
            x, y = super()._sample_coords()
        return self.parse(f"({x}, {y})")

    @staticmethod
    def _rc_to_xy(mat: np.ndarray) -> np.ndarray:
        # convert from r,c to x,y by rotating counter-clockwise 90deg
        assert mat.ndim == 2
        return np.rot90(mat, axes=(1, 0))

    @staticmethod
    def _xy_to_rc(mat: np.ndarray) -> np.ndarray:
        assert mat.ndim == 2
        return np.rot90(mat)


def str_to_float_mask(str_mask: List[str]) -> np.ndarray:
    """
    Convert a list of strings into a binary float array,
    where any '#' chars are interpreted as 1, any other char
    is interpreted as 0.
    """
    return np.array([
        [
            float(c == "#")
            for c in line
        ] for line in str_mask
    ])


class PointFeaturizer(Featurizer):

    def __init__(self, ndims: int):
        self.ndims = ndims

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        return np.array(batch)

    @property
    def n_features(self) -> int:
        return self.ndims


class GaussianSampler:
    def __init__(self, data: List[float], std):
        if len(data) == 1:
            self.dist_type = "single"
            self.dist = stats.norm(loc=data[0], scale=std)
        else:
            self.dist_type = "multi"
            self.dist = stats.gaussian_kde(data, bw_method=std)

    def sample(self, *args) -> float:
        # Generate a sample from the Gaussian distribution
        if self.dist_type == "single":
            return self.dist.rvs()
        else:
            return self.dist.resample(1).item()

    def logpdf(self, x) -> float:
        return self.dist.logpdf(x).item()


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

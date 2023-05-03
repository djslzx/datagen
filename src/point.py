"""
Languages for testing novelty search with "point programs".
Programs are 2D points, and novelty is the Euclidean distance between them.
"""
from typing import Any, Dict, List
import numpy as np
import scipy.stats as stats

from grammar import Grammar
from featurizers import Featurizer
from lang import Language, Tree
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

    def __init__(self):
        super().__init__(
            parser_grammar=RealPoint.grammar,
            parser_start="point",
            root_type="Point",
            model=None,
            featurizer=PointFeaturizer(ndims=2)
        )
        self.x_distribution = None
        self.y_distribution = None

    def none(self) -> Any:
        return np.array([0, 0])

    def simplify(self, t: Tree) -> Tree:
        raise NotImplementedError

    def _parse_child(self, t: Tree) -> float:
        if t.is_leaf():
            return float(t.value)
        else:
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
        self.x_distribution = stats.gaussian_kde(xs)
        self.y_distribution = stats.gaussian_kde(ys)

    def sample(self) -> Tree:
        x = self.x_distribution.resample(1).item()
        y = self.y_distribution.resample(1).item()
        return self.parse(f"({x}, {y})")

    @property
    def str_semantics(self) -> Dict:
        return {
            "point": lambda x, y: f"({x}, {y})",
            "pos": lambda n: f"{n}",
            "neg": lambda n: f"-{n}",
        }

    def log(self, popn: List[Tree]):
        pass


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
        "inc":   ["Nat", "Nat"],
        "one":  ["Nat"],
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
            "point": lambda x,y: f"({x}, {y})",
            "inc": lambda n: f"(inc {n})",
            "one": lambda: "1",
        }

    def none(self) -> Any:
        return np.array([0, 0])


class PointFeaturizer(Featurizer):

    def __init__(self, ndims: int):
        self.ndims = ndims

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        return np.array(batch)

    @property
    def n_features(self) -> int:
        return self.ndims


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
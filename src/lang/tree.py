"""
Defines Tree and Program
"""
from __future__ import annotations

from math import ceil
from typing import *
import lark
import numpy as np
from torch import Tensor
from tqdm import tqdm
from einops import rearrange
import pdb

from featurizers import Featurizer
from grammar import Grammar
import util


class Tree:
    """
    A program tree
    """

    def __init__(self, value, *children):
        self.value = value
        self.children: List[Tree] = list(children)

        # compute length at construction time
        # length: number of nodes in the tree
        if not children:
            self.length = 1
        else:
            self.length = 1 + sum(len(c) for c in children)

    @staticmethod
    def from_tuple(t: Tuple):
        if not isinstance(t, tuple):
            return Tree(t)

        sym, *args = t
        children = []
        for arg in args:
            if isinstance(arg, tuple):
                children.append(Tree.from_tuple(arg))
            else:
                children.append(Tree(arg))
        return Tree(sym, *children)

    @staticmethod
    def from_lark(ltree: lark.Tree | lark.Token):
        if isinstance(ltree, lark.Tree):
            return Tree(ltree.data, *(Tree.from_lark(x) for x in ltree.children))
        elif isinstance(ltree, lark.Token):
            return Tree(ltree.value)
        else:
            raise ValueError(f"Lark trees must be Trees or Tokens, but got {type(ltree)}")

    @staticmethod
    def from_sexp(s: str):
        def tokenize(s: str) -> List[str]:
            """Split s into space-separated tokens - include parens"""
            return s.replace("(", "( ").replace(")", " )").split()

        def group_parens(tokens: List[str]) -> Dict[int, int]:
            """Return a map from positions of open parens to the positions of their paired close parens"""
            ends = {}
            stack = []
            for i, token in enumerate(tokens):
                if token == "(":
                    stack.append(i)
                elif token == ")":
                    ends[stack.pop()] = i
            if stack:
                raise ValueError(f"Mismatched parentheses in expression: {''.join(tokens)}")
            return ends

        def translate(i: int, j: int) -> Tree:
            sym = None
            args = []
            r = i
            while r < j:
                if r not in ends:  # atom
                    if sym is None:
                        sym = tokens[r]
                    else:
                        args.append(Tree(tokens[r]))
                    r += 1
                else:  # s-exp
                    args.append(translate(r + 1, ends[r]))
                    r = ends[r] + 1
            return Tree(sym, *args)

        tokens = tokenize(s)
        ends = group_parens(tokens)
        assert tokens[0] == "(" and tokens[-1] == ")", f"Found unbracketed token sequence: {tokens}"
        return translate(1, len(tokens) - 1)

    def is_leaf(self) -> bool:
        return not self.children

    def to_sexp(self) -> str:
        if not self.is_leaf():
            s_args = [c.to_sexp() for c in self.children]
            args = " ".join(s_args)
            return f"({self.value} {args})"
        else:
            return str(self.value)

    def to_tuple(self) -> Tuple:
        if not self.is_leaf():
            args = [c.to_tuple() for c in self.children]
            return tuple([self.value] + args)
        else:
            return self.value

    def to_str(self, semantics: Dict) -> str:
        assert isinstance(semantics, dict), f"Expected dict but got {semantics} of type {type(semantics)}"
        sym = self.value
        args = [c.to_str(semantics) for c in self.children]
        if sym in semantics:
            f = semantics[sym]
            if callable(f):
                return f(*args)
            else:
                return f
        else:
            return sym

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not self.is_leaf():
            s_args = [c.to_sexp() for c in self.children]
            args = " ".join(s_args)
            return f"({self.value} {args})"
        else:
            return f"`{self.value}`"

    def __len__(self):
        """number of nodes in tree"""
        return self.length


class Language:
    """
    Defines a domain that can be used with novelty search.
    """

    Symbol = Union[tuple, str]
    Type = str

    # TODO: separate model from language

    def __init__(
            self,
            parser_grammar: str,
            parser_start: str,
            root_type: Optional[str],
            model: Optional[Grammar],
            featurizer: Featurizer
    ):
        self.parser = lark.Lark(parser_grammar, start=parser_start, parser='lalr')
        self.model = model
        if model is not None and root_type is None:
            raise ValueError("Root type must be defined for non-empty model")
        self.start = root_type
        if self.model: self.model.normalize_()
        self.featurizer = featurizer

    def none(self) -> Any:
        raise NotImplementedError

    def parse(self, s: str) -> Tree:
        """Parses a string into an AST in the language"""
        ltree = self.parser.parse(s)
        return Tree.from_lark(ltree)

    def simplify(self, t: Tree) -> Tree:
        """Simplifies a program in the language"""
        raise NotImplementedError

    def sample(self) -> Tree:
        """Probabilistically sample a tree from the language"""
        if self.model is None:
            raise ValueError("Cannot sample from an empty model")

        s = self.model.sample(self.start)
        t = Tree.from_tuple(s)
        return t

    def samples(self, n_samples: int, length_cap: int) -> List[Tree]:
        out = []
        while len(out) < n_samples:
            try:
                t = self.sample()
            except RecursionError:
                continue  # retry
            if len(t) <= length_cap:
                out.append(t)
        return out

    def fit(self, corpus: List[Tree], alpha):
        if self.model is None:
            raise ValueError("Cannot fit with empty model")
        ones = np.ones(len(corpus))
        if self.model.gram == 1:
            counts = sum_scans(corpus, weights=ones, scanner=unigram_scan)
            self.model.from_unigram_counts_(counts, alpha=alpha)
        elif self.model.gram == 2:
            counts = sum_scans(corpus, weights=ones, scanner=bigram_scan)
            counts.update(sum_scans(corpus, weights=ones, scanner=unigram_scan))
            self.model.from_bigram_counts_(counts, alpha=alpha)
        else:
            raise AttributeError(f"Cannot fit on grammar with gram={self.model.gram}")

    def log_probability(self, t: Tree) -> float:
        """Computes the probability of a tree in the language"""
        if self.model is None:
            raise ValueError("Cannot compute log probability of empty model")
        return self.model.log_probability(self.start, t.to_tuple()).item()

    def extract_features(self, trees: Collection[Tree], n_samples=1, batch_size=4, load_bar=False) -> np.ndarray:
        """
        Extract features from a collection of programs.
        """
        outputs, features = self.evaluate_features(trees, n_samples, batch_size, load_bar)
        return features

    def evaluate_features(
            self, 
            trees: Collection[Tree], 
            n_samples=1, 
            batch_size=4, 
            load_bar=False
    ) -> Tuple[Any, np.ndarray]:
        """
        Extract program outputs and features from a collection of programs.
        """

        def evaluate_trees():
            for x in trees:
                for _ in range(n_samples):
                    yield self.eval(x)

        outputs = []
        features = []
        n_batches = ceil(len(trees) * n_samples / batch_size)
        batches = util.batched(evaluate_trees(), batch_size=batch_size)
        if load_bar:
            batches = tqdm(batches, total=n_batches)
        for batch in batches:
            batch = np.array(batch)

            fv = self.featurizer.apply(batch)

            # add a batch dimension if we're missing one
            if fv.ndim == 1:
                fv = fv[None, :]
            features.extend(fv)

            if batch.ndim == 1:
                batch = batch[None, :]
            outputs.extend(batch)

        outputs = np.array(outputs)
        features = np.array(features)
        assert features.ndim == 2, f"Feature extraction should yield a single vector per tree"
        assert features.shape[0] == (len(trees) * n_samples), \
            f"Expected to get {len(trees)} * {n_samples} = {len(trees) * n_samples} feature vectors, but got out:{features.shape}"

        features = rearrange(features, "(s samples) features -> s (samples features)", 
                             s=len(trees), samples=n_samples)
        return outputs, features

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> Any:
        """Executes a tree in the language"""
        raise NotImplementedError

    def to_str(self, t: Tree) -> str:
        assert isinstance(t, Tree)
        return t.to_str(self.str_semantics)

    @property
    def str_semantics(self) -> Dict:
        raise NotImplementedError

    # def features(self, batch: List[Tree], env: Dict[str, Any]) -> np.ndarray:
    #     out = []
    #     for tree in batch:
    #         out.append(self.eval(tree, env))
    #     return self.featurizer.apply(out)


def unigram_scan(t: Tree, w=1.) -> Dict[str, int]:
    counts = {}

    def traverse(node: Tree):
        counts[node.value] = counts.get(node.value, 0) + w
        for c in node.children:
            traverse(c)

    traverse(t)
    return counts


def bigram_scan(t: Tree, w=1.) -> Dict[Tuple[str, int, str], int]:
    counts = {}

    def traverse(node: Tree):
        for i, c in enumerate(node.children):
            k = (node.value, i, c.value)
            counts[k] = counts.get(k, 0) + w
            traverse(c)

    traverse(t)
    return counts


def sum_scans(trees: List[Tree], weights: List[float] | np.ndarray,
              scanner: Callable[[Tree, float], Dict[Any, int]]) -> Dict[Any, int]:
    assert len(trees) == len(weights)
    counts = {}
    for tree, weight in zip(trees, weights):
        d = scanner(tree, weight)
        for k, v in d.items():
            counts[k] = counts.get(k, 0) + v
    return counts


class ParseError(Exception):
    pass


def test_unigram_scan():
    cases = [
        "(a)", {"a": 1.},
        "(a a a a)", {"a": 4.},
        "(a (a k) a (b k (c k) (c k)))", {"a": 3, "b": 1, "c": 2, "k": 4},
    ]
    for sexp, d in zip(cases[::2], cases[1::2]):
        t = Tree.from_sexp(sexp)
        out = unigram_scan(t)
        assert out == d, f"Expected {d} but got {out}"


def test_bigram_scan():
    cases = [
        "(+ (- 1 2) 3)", {("+", 0, "-"): 1,
                          ("+", 1, "3"): 1,
                          ("-", 0, "1"): 1,
                          ("-", 1, "2"): 1},
        "(+ (- 1 2) "
        "   (+ (- 1 2)"
        "      (- 1 2)))", {("+", 0, "-"): 2,
                            ("+", 1, "-"): 1,
                            ("+", 1, "+"): 1,
                            ("-", 0, "1"): 3,
                            ("-", 1, "2"): 3}
    ]
    for sexp, counts in zip(cases[::2], cases[1::2]):
        t = Tree.from_sexp(sexp)
        out = bigram_scan(t)
        assert out == counts, f"Expected {counts} but got {out}"


def test_tree_len():
    cases = [
        "(+ 2 3)", 3,
        "(+ (- 1 0) 3)", 5,
        "(+ (- 1 0) (* 2 2))", 7,
        "(+ (- 1 0) (* (+ 2 9) (- 2 3)))", 11,
    ]
    for sexp, size in zip(cases[::2], cases[1::2]):
        t = Tree.from_sexp(sexp)
        out = len(t)
        assert out == size, f"Expected {size} but got {out}"

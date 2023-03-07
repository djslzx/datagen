"""
Defines Tree and Program
"""
from __future__ import annotations
from typing import *
import lark
import numpy as np

from featurizers import Featurizer
from grammar import Grammar


class Tree:
    """
    A program tree
    """
    def __init__(self, value, *children):
        self.value = value
        self.children: List[Tree] = list(children)

    @staticmethod
    def from_tuple(t: Tuple):
        return Tree(*t)

    @staticmethod
    def from_lark(ltree: lark.Tree | lark.Token):
        if isinstance(ltree, lark.Tree):
            return Tree(ltree.data, *(Tree.from_lark(x) for x in ltree.children))
        elif isinstance(ltree, lark.Token):
            return Tree(ltree.value)
        else:
            raise ValueError("Lark trees must be Trees or Tokens")

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
            args = " ".join(c.to_sexp() for c in self.children)
            return f"({self.value} {args})"
        else:
            return self.value

    def to_tuple(self) -> Tuple:
        if not self.is_leaf():
            args = [c.to_tuple() for c in self.children]
            return tuple([self.value] + args)
        else:
            return self.value

    def to_str(self, semantics: Dict) -> str:
        sym = self.value
        args = [c.to_str(semantics) for c in self.children]
        if sym in semantics:
            return semantics[sym](*args)
        else:
            return sym

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not self.is_leaf():
            args = " ".join(c.to_sexp() for c in self.children)
            return f"({self.value} {args})"
        else:
            return f"<{self.value}>"

    def __len__(self):
        # number of nodes in tree
        return 1 + sum(len(c) for c in self.children)


class Language:
    """
    Defines a domain that can be used with novelty search.
    """

    Symbol = Union[tuple, str]
    Type = str

    def __init__(self, parser_grammar: str, start: str, model: Grammar, featurizer: Featurizer):
        self.start = start
        self.parser_grammar = parser_grammar
        self.parser = lark.Lark(parser_grammar, start=start, parser='lalr')
        self.model = model
        self.featurizer = featurizer

    def parse(self, s: str) -> Tree:
        """Parses a string into an AST in the language"""
        ltree = self.parser.parse(s, start=self.start)
        return Tree.from_lark(ltree)

    def simplify(self, t: Tree) -> Tree:
        """Simplifies a program in the language"""
        raise NotImplementedError

    def sample(self) -> Tree:
        """Probabilistically sample a tree from the language"""
        return Tree.from_tuple(self.model.sample(self.start))

    def fit(self, corpus: List[Tree]):
        counts = bigram_scans(corpus, weights=np.ones(len(corpus)))
        self.model.from_bigram_counts_(counts)

    def eval(self, t: Tree, env: Dict[str, Any]) -> Any:
        """Executes a tree in the language"""
        raise NotImplementedError

    def features(self, batch: List[Tree], env: Dict[str, Any]) -> np.ndarray:
        out = []
        for tree in batch:
            out.append(self.eval(tree, env))
        return self.featurizer.apply(out)


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


def bigram_scans(trees: List[Tree], weights: List[float] | np.ndarray) -> Dict[Tuple[str, int, str], int]:
    assert len(trees) == len(weights)
    counts = {}
    for tree, weight in zip(trees, weights):
        b = bigram_scan(tree, weight)
        for k, v in b.items():
            counts[k] = counts.get(k, 0) + v
    return counts


def sum_counts(a: Dict[Any, float], b: Dict[Any, float]) -> Dict[Any, float]:
    return {k: a.get(k, 0) + b.get(k, 0)
            for k in a.keys() | b.keys()}


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

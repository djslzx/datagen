"""
Defines Tree and Program
"""
from __future__ import annotations
from typing import *
import lark
import numpy as np

from featurizers import Featurizer
from grammar import Grammar

"""
Program drawn from Grammar of Language
Grammar specifies Language

Language
  parser: lark.Parser
  model: Grammar
  
  parse(Str) -> Program:
    use self.parser
  fit(List[Str]):
    use self.count
  simplify(Program) -> Program:
    use self.parser to get s-exp,
    then use rust module
  eval(Program) -> Value 
    
  sample() -> Program

LSystem, Regex <: Language
"""


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

    def __repr__(self):
        return self.to_sexp()

    def __str__(self):
        return self.to_sexp()

    def __len__(self):
        # number of nodes in tree
        return 1 + sum(len(c) for c in self.children)


class Language:
    """
    Defines a domain that can be used with novelty search.
    """

    Symbol = (tuple | str)
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

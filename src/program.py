"""
Defines Tree and Program
"""
from __future__ import annotations
from typing import *

from cfg import CFG


class Tree:
    V = TypeVar("V")  # value type
    Y = TypeVar("Y")  # function output type

    def __init__(self, value: V, children: List[V]):
        self.value = value
        self.children = children

    @staticmethod
    def leaf(value: V) -> Tree:
        return Tree(value, [])

    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.to_tuple())

    def map(self, f: Callable[[V], Y]) -> Tree:
        return Tree(f(self.value), [child.map(f) for child in self.children])

    def apply(self, f: Callable[[V, List[Y]], Y]):
        return f(self.value, [child.apply(f) for child in self.children])

    def to_tuple(self) -> Tuple:
        return self.apply(lambda *args: args)  # *args is already a tuple


class Program:
    def __init__(self):
        pass

    @property
    def grammar(self) -> CFG:
        """The program type's grammar"""
        raise NotImplementedError

    def parse(self, s: str) -> Tree:
        """Parses a string into an AST in the program grammar"""
        raise NotImplementedError

    def flatten(self, t: Tree) -> str:
        """Flattens a tree in the program grammar into a string"""
        raise NotImplementedError

    def execute(self, t: Tree, env: Dict[str, Any]) -> Any:
        """Executes a tree in the program grammar"""
        raise NotImplementedError



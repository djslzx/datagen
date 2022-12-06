from __future__ import annotations
import random
from typing import Dict, List, Iterator, Tuple
from math import sin, cos, radians
import numpy as np
import skimage.draw
import itertools as it
import pdb

from cfg import CFG, Grammar, LearnedGrammar

LSYSTEM_MG = CFG("L-SYSTEM", {
    "L-SYSTEM": [
        ["AXIOM", ";", "RULES"],
    ],
    "AXIOM": [
        ["NT", "AXIOM_W_NT"],
        ["NT"],
        ["T", "AXIOM"],
    ],
    "AXIOM_W_NT": [
        ["NT_OR_T", "AXIOM_W_NT"],
        ["NT_OR_T"],
    ],
    "RULES": [
        ["RULE", ",", "RULES"],
        ["RULE"],
    ],
    "RULE": [
        ["LHS", "~", "RHS"],
    ],
    "LHS": [
        ["NT"],
    ],
    "RHS": [
        ["[", "RHS", "]", "RHS"],
        ["[", "RHS", "]"],
        ["NT", "RHS_W_NT"],
        ["NT"],
        ["T", "RHS"],
    ],
    "RHS_W_NT": [
        ["[", "RHS", "]", "RHS"],
        ["[", "RHS", "]"],
        ["NT_OR_T", "RHS_W_NT"],
        ["NT_OR_T"],
    ],
    "NT_OR_T": [
        ["NT"],
        ["T"]
    ],
    "NT": [
        ["F"],
        # ["f"],
        # ["X"],
    ],
    "T": [
        ["+"],
        ["-"],
    ],
})


class LSystem:

    def __init__(self):
        pass

    def expand(self, s: str) -> str:  # pragma: no cover
        assert False, f"Should be implemented in child {type(self).__name__}"

    def expansions(self, iters: int) -> Iterator[str]:
        """Returns a generator over the 0-th through `iters`-th expansions."""
        word = self.axiom
        yield word
        for _ in range(iters):
            word = self.expand(word)
            yield word

    def nth_expansion(self, n: int) -> str:
        """Returns the n-th expansion."""
        word = self.axiom
        for _ in range(n):
            word = self.expand(word)
        return word

    def expand_until(self, length: int) -> Tuple[int, str]:
        """
        Apply rules to the axiom until the number of `F` tokens is >= length
        """
        word = self.axiom
        depth = 0
        while len(word) < length:
            cache = word
            word = self.expand(word)
            if word == cache:
                break
            depth += 1
        return depth, word

    @staticmethod
    def draw(s: str, d: float, theta: float, n_rows: int = 512, n_cols: int = 512) -> np.ndarray:  # pragma: no cover
        """
        Draw the turtle interpretation of the string `s` onto a `n_rows` x `n_cols` array,
        using scikit-image's drawing library (with anti-aliasing).
        """
        r, c = n_rows//2, n_cols//2  # start at center of canvas
        heading = 90  # start facing up (logo)
        stack = []
        canvas = np.zeros((n_rows, n_cols), dtype=np.uint8)
        for char in s:
            if char == 'F':
                r1 = r + int(d * sin(radians(heading)))
                c1 = c + int(d * cos(radians(heading)))
                rs, cs, val = skimage.draw.line_aa(r, c, r1, c1)
                # mask out out-of-bounds indices
                mask = (0 <= rs) & (rs < n_rows) & (0 <= cs) & (cs < n_cols)
                rs, cs, val = rs[mask], cs[mask], val[mask]
                canvas[rs, cs] = val * 255
                r, c = r1, c1
            elif char == 'f':
                r += int(d * sin(radians(heading)))
                c += int(d * cos(radians(heading)))
            elif char == '+':
                heading += theta
            elif char == '-':
                heading -= theta
            elif char == '[':
                stack.append((r, c, heading))
            elif char == ']':
                r, c, heading = stack.pop()
        return canvas


class D0LSystem(LSystem):
    """
    A deterministic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self, axiom: str, productions: Dict[str, str]):
        super().__init__()
        self.axiom = axiom
        self.productions = productions

    def __str__(self) -> str:  # pragma: no cover
        rules = []
        for pred, succs in self.productions.items():
            for i, succ in enumerate(succs):
                rules.append(
                    f'{pred} -> {succ}'
                )
        return f'axiom: {self.axiom}\n' + 'rules: [\n  ' + '\n  '.join(rules) + '\n]\n'

    def expand(self, s: str) -> str:
        # Assume identity production if predecessor is not in self.productions
        return ''.join(self.productions.get(c, c) for c in s)


class S0LSystem(LSystem):
    """
    A stochastic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    # TODO: add check-rep?

    def __init__(self,
                 axiom: str,
                 productions: Dict[str, List[str]],
                 distribution: str | Dict[str, List[float]] = "uniform"):
        super().__init__()
        self.axiom = axiom
        self.productions = productions

        # check if distribution is a string
        if distribution == "uniform":
            self.distribution = {
                pred: np.ones(len(succs)) / len(succs)
                for pred, succs in productions.items()
            }
        else:
            self.distribution = {
                pred: (lambda x: x / np.sum(x))(np.array(weights))
                for pred, weights in distribution.items()
            }

    def expand(self, s: str) -> str:
        return ''.join(random.choices(population=self.productions.get(c, [c]),
                                      weights=self.distribution.get(c, [1]),
                                      k=1)[0]
                       for c in s)

    def __str__(self) -> str:  # pragma: no cover
        rules = []
        for pred, succs in self.productions.items():
            for i, succ in enumerate(succs):
                weight = self.distribution[pred][i]
                rules.append(
                    f'{pred} -[{weight:.3f}]-> {succ}'
                )
        return f'axiom: {self.axiom}\n' + \
               'rules: [\n  ' + '\n  '.join(rules) + '\n]\n'

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other) -> bool:
        # doesn't handle different orderings
        return (isinstance(other, S0LSystem) and
                self.axiom == other.axiom and
                self.productions == other.productions and
                all(np.array_equal(self.distribution[pred], other.distribution[pred])
                    for pred in self.productions.keys()))

    def to_sentence(self) -> List[str]:
        """
        Convert the L-system to a sentence outputted by a metagrammar.
        Needed to fit metagrammars to libraries of L-systems.
        """
        return (list(self.axiom) + [';'] + list(it.chain.from_iterable(
            [[pred, '~', *succ, ',']
             for pred, succs in self.productions.items()
             for succ in succs])))[:-1]

    def to_code(self) -> str:
        return "".join(self.to_sentence())

    @staticmethod
    def from_sentence(s: List[str] | Tuple[str]) -> 'S0LSystem':
        """
        Accepts a single string with spaces between distinct tokens, and outputs an L-system.
        The list should have the form 'AXIOM; RULE, RULE, ...', where RULE has the form 'LHS ~ RHS'.
        """
        assert isinstance(s, List) or isinstance(s, Tuple), f"Expected list/tuple of strings but found {type(s)}"
        s = " ".join(s)
        s_axiom, s_rules = s.strip().split(';')
        axiom = s_axiom.replace(' ', '')
        s_rules = s_rules.strip()

        rules = {}
        for s_rule in s_rules.split(','):
            if not s_rule.strip():
                continue

            lhs, rhs = s_rule.split('~')
            lhs = lhs.strip()
            rhs = rhs.replace(',', '').strip()
            rhs = ''.join(rhs.split())

            if lhs in rules:
                rules[lhs].append(rhs)
            else:
                rules[lhs] = [rhs]

        return S0LSystem(axiom, rules, "uniform")

    def to_tree(self) -> Tuple:
        """Converts the grammar to a tuple encoding a tree"""
        # TODO: implement a general-purpose parser?

        def parse_axiom(s: str) -> Tuple:
            assert s, "Received empty string"
            if len(s) == 1:
                hd, tl = s[0], None
            else:
                hd, tl = s[0], s[1:]
            assert hd in ["F", "+", "-"]
            symbol = ("symbol_nonterm" if hd == "F" else "symbol_term", hd)
            if not tl:  # leaf
                return ("axiom_atom", symbol)
            else:
                return ("axiom_extend", symbol, parse_axiom(tl))

        def find_closing_bracket(s: str) -> int:
            n_brackets = 0
            for i, c in enumerate(s):
                if c == "[":
                    n_brackets += 1
                elif c == "]":
                    if n_brackets == 0:
                        return i
                    else:
                        n_brackets -= 1
            raise ValueError(f"Mismatched brackets in {s}")

        def parse_rhs(s: str) -> Tuple | str:
            if len(s) == 0:
                return "rhs_empty"
            elif len(s) == 1:
                hd, tl = s[0], None
            else:
                hd, tl = s[0], s[1:]

            assert hd in ["F", "+", "-", "["]

            if hd == "[":
                # find position of paired closing bracket; assumes balanced brackets
                end = find_closing_bracket(tl)
                inner = tl[:end]
                outer = tl[end+1:]  # ignore closing bracket
                return ("rhs_bracket", parse_rhs(inner), parse_rhs(outer))

            symbol = ("symbol_nonterm" if hd == "F" else "symbol_term", hd)
            if not tl:  # leaf
                return ("rhs_symbol", symbol, "rhs_empty")
            else:
                return ("rhs_symbol", symbol, parse_rhs(tl))

        def parse_rules(rules: Dict[CFG.Word, List[CFG.Sentence]]) -> Tuple:
            rule_trees = []
            for lhs, rhss in rules.items():
                for rhs in rhss:
                    rhs_tree = parse_rhs(rhs)
                    rule_trees.append(("rule", ("lhs", lhs), rhs_tree))
            rules_tree = ('rules_atom', rule_trees[-1])
            for tree in reversed(rule_trees[:-1]):
                rules_tree = ("rules_extend", tree, rules_tree)
            return rules_tree

        axiom_expr = parse_axiom(self.axiom)
        rules_expr = parse_rules(self.productions)
        return ('l_system', axiom_expr, rules_expr)

    @staticmethod
    def metagrammar() -> Grammar:
        """
        root -> axiom rules
        symbol -> nonterm | term
        axiom -> symbol axiom | axiom
        rules -> rule rules | rule
        rule -> lhs rhs
        lhs -> nonterm
        rhs -> [ rhs ] rhs | symbol rhs | empty
        term -> + | -
        nonterm -> F
        """
        # Types are capitalized; functions are lowercase
        components = {
            "l_system": ["Axiom", "Rules", "LSystem"],
            "symbol_nonterm": ["Nonterm", "Symbol"],
            "symbol_term": ["Term", "Symbol"],
            "axiom_extend": ["Symbol", "Axiom", "Axiom"],
            "axiom_atom": ["Symbol", "Axiom"],
            "rules_extend": ["Rule", "Rules", "Rules"],
            "rules_atom": ["Rule", "Rules"],
            "rule": ["Lhs", "Rhs", "Rule"],
            "lhs": ["Nonterm", "Lhs"],
            "rhs_bracket": ["Rhs", "Rhs", "Rhs"],
            "rhs_symbol": ["Symbol", "Rhs", "Rhs"],
            "rhs_empty": ["Rhs"],
            "+": ["Term"],
            "-": ["Term"],
            "F": ["Nonterm"],
        }
        semantics = {
            "l_system": lambda axiom, rules: S0LSystem(axiom, rules, "uniform"),
            "symbol_nonterm": lambda nonterm: nonterm,
            "symbol_term": lambda term: term,
            "axiom_extend": lambda symbol, axiom: symbol + axiom,
            "axiom_atom": lambda symbol: symbol,
            "rules_extend": lambda rule, rules: rule + "," + rules,
            "rules_atom": lambda rule: rule,
            "rule": lambda lhs, rhs: lhs + "~" + rhs,
            "lhs": lambda nonterm: nonterm,
            "rhs_bracket": lambda rhs1, rhs2: "[" + rhs1 + "]" + rhs2,
            "rhs_symbol": lambda sym, rhs: sym + rhs,
            "rhs_empty": lambda: "",
            "+": lambda: "+",
            "-": lambda: "-",
            "F": lambda: "F"
        }
        return Grammar.from_components(components, 1).normalize_()

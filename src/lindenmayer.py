from __future__ import annotations
import random
from typing import *
from math import sin, cos, radians
import numpy as np
import skimage.draw
import itertools as it

from cfg import CFG, PCFG
import util

MG = CFG("LSystem", {
    "LSystem": [
        ["Axiom", ";", "Rules"],
    ],
    "Nonterminal": [
        ["F"],
    ],
    "Terminal": [
        ["+"],
        ["-"],
    ],
    "Axiom": [
        ["Nonterminal", "Axiom"],
        ["Terminal", "Axiom"],
        [],
    ],
    "Rules": [
        ["Rule", ",", "Rules"],
        ["Rule"],
    ],
    "Rule": [
        ["Nonterminal", "~", "Rhs"],
    ],
    "Rhs": [
        ["[", "Rhs", "]", "Rhs"],
        ["Nonterminal", "Rhs"],
        ["Terminal", "Rhs"],
        [],
    ],
})


def parse_lsystem(s: str, f: Callable) -> Any:
    """
    Greedily parse the string s into a tree of (type, id) nodes, where
    - type is the nonterminal used, and
    - id is the index of the rule used in the nonterminal's list of rules
    """
    assert isinstance(s, str), f"Got unexpected type: {type(s)}"

    def parse_symbol(s: str) -> Tuple:
        assert len(s) == 1, f"Found long symbol: {s}"
        if s == "F":
            return f("Nonterminal", 0)
        else:
            return f("Terminal", 0 if s == "+" else 1)

    def parse_axiom(s: str) -> Tuple:
        if len(s) == 0:
            return f("Axiom", 2)
        hd, tl = s[0], s[1:]
        assert hd in ["F", "+", "-"], f"Unexpected head: '{hd}'"
        i = (0 if hd == "F" else 1)  # choose Axiom's Nonterm or Term rule
        return f("Axiom", i, parse_symbol(hd), parse_axiom(tl))

    def parse_rhs(s: str) -> Tuple | str:
        if len(s) == 0:
            return f("Rhs", 3)
        hd, tl = s[0], s[1:]
        assert hd in ["F", "+", "-", "["], f"Unexpected head: '{hd}'"
        if hd == "[":
            # find position of paired closing bracket; assumes balanced brackets
            end = util.find_closing_bracket(tl)
            inner = tl[:end]
            outer = tl[end + 1:]  # ignore closing bracket
            return f("Rhs", 0, parse_rhs(inner), parse_rhs(outer))

        i = (1 if hd == "F" else 2)
        return f("Rhs", i, parse_symbol(hd), parse_rhs(tl))

    def parse_rule(s: str) -> Tuple:
        lhs, rhs = s.split("~")
        return f("Rule", 0, f("Nonterminal", 0), parse_rhs(rhs))

    def parse_rules(s: str) -> Tuple:
        rule_exprs = [parse_rule(rule) for rule in s.split(",")]
        expr = f("Rules", 1, rule_exprs[-1])
        for rule_expr in reversed(rule_exprs[:-1]):
            expr = f("Rules", 0, rule_expr, expr)
        return expr

    axiom, rules = s.split(";")
    axiom_expr = parse_axiom(axiom)
    rules_expr = parse_rules(rules)

    return f("LSystem", 0, axiom_expr, rules_expr)


def parse_lsystem_ast(s: str) -> Tuple:
    return parse_lsystem(s, lambda *x: tuple(x))


def apply_to_tree(root: Tuple, f: Callable) -> Any:
    symbol, i, *args = root
    if not args:
        return f(symbol, i)
    else:
        return f(symbol, i, *[apply_to_tree(arg, f) for arg in args])


def parse_lsystem_str_as_counts(s: str) -> Dict[str, np.ndarray]:
    counts = empty_mg_counts()

    def count(nt, i, *args):
        counts[nt][i] += 1

    parse_lsystem(s, count)
    return counts


def empty_mg_counts() -> Dict[str, np.ndarray]:
    return {
        nt: np.zeros(len(MG.rules[nt]))
        for nt in MG.nonterminals
    }


def count_rules(corpus: List[str]) -> Dict[str, np.ndarray]:
    sum_counts = empty_mg_counts()
    for word in corpus:
        counts = parse_lsystem_str_as_counts(word)
        for k, v in counts.items():
            sum_counts[k] += v
    return sum_counts


def trained_metagrammar(corpus: List[str], alpha=0.1) -> PCFG:
    counts = count_rules(corpus)
    weights = {
        nt: (vec + alpha) / np.sum(vec + alpha)
        for nt, vec in counts.items()
    }
    return PCFG.from_CFG(MG, weights)


def empty_bigram_counts(f: Callable[[str, int, int], str]) -> Dict[str, np.ndarray]:
    # todo: test
    return {
        f(nt, i, j): np.zeros(len(MG.rules[succ]))
        for nt in MG.nonterminals
        for i, succs in enumerate(MG.successors(nt))
        for j, succ in enumerate(succ for succ in succs if MG.is_nonterminal(succ))
    }


def bigram_map(g: CFG) -> Dict[Tuple[str, int, int], str]:
    """Construct the map from (nt, i, j) to the annotated bigram nonterminal"""
    # todo: test
    counts = {nt: 0 for nt in g.nonterminals}
    mapping = {}
    for p in g.nonterminals:
        for i, s in enumerate(g.successors(p)):
            j = 0  # count only nonterminal successors
            for tok in s:
                if g.is_nonterminal(tok):
                    counts[tok] += 1
                    mapping[p, i, j] = f"{tok}_{counts[tok]}"
                    j += 1
    return mapping


def bigram_counts(s: str) -> Dict[str, np.ndarray]:
    # todo: test
    map_fn = bigram_map(MG)
    counts = empty_bigram_counts(lambda *k: map_fn[k])

    def count(symbol, i, *args):
        for j, arg in enumerate(args):
            k = map_fn[symbol, i, j]
            counts[k][arg] += 1
        return i

    parse_lsystem(s, count)
    return counts


def trained_bigram_metagrammar(corpus: List[str], alpha=0.1) -> PCFG:
    map_fn = bigram_map(MG)
    counts = empty_bigram_counts(lambda *k: map_fn[k])
    for word in corpus:
        counts = bigram_counts(word)
        for k, v in counts.items():
            counts[k] += v
    weights = {
        k: (vec + alpha) / np.sum(vec + alpha)
        for k, vec in counts.items()
    }
    # add start symbol with weight 1
    weights[MG.start] = np.ones(1)
    grammar = MG.to_bigram()
    return PCFG.from_CFG(grammar, weights)


def bigram_log_pr(g: PCFG, s: str) -> np.ndarray:
    assert g.log_mode
    counts = bigram_counts(s)
    return sum(
        g.weights[key][i] * count
        for key, count_vec in counts.items()
        for i, count in enumerate(count_vec)
    )


class LSystem:

    def __init__(self):
        pass

    def expand(self, s: str) -> str:  # pragma: no cover
        raise NotImplementedError(f"Should be implemented in child {type(self).__name__}")

    @property
    def axiom(self) -> str:
        raise NotImplementedError(f"Should be implemented in child {type(self).__name__}")

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
        self._axiom = axiom
        self.productions = productions

    @property
    def axiom(self) -> str:
        return self._axiom

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
        self._axiom = axiom
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

    @property
    def axiom(self) -> str:
        return self._axiom

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

"""
Handles parsing and optimization of L-system strings of the form 'F;F~F...'
using lark and egg (packaged as `eggy` for the DSL used here).

This module defines a metagrammar over L-systems that guides parsing, in
addition to types and semantics for the rules in the metagrammar.

Because `egg` requires ASTs (represented as s-expressions) as input and
produces AST strings as output, but the learner requires ASTs as tuples,
this module contains multiple translation functions that convert between
lark parse trees, s-expression strings, and tuple trees.

Using `egg` to simplify an L-system string looks as follows:

  Lsys str => Lsys lark AST => Lsys s-exp => [egg]
  [egg] => simpl s-exp => simpl tuple AST => simpl str
"""

from __future__ import annotations

from typing import *
import lark
import eggy
import sys


# grammar over lsystems
lsystem_metagrammar = r"""
    lsystem: axiom ";" rules   -> lsystem
    axiom: symbols             -> axiom
    symbols: symbol symbols    -> symbols
           | symbol            -> symbol
    symbol: "[" symbols "]"    -> bracket
          | NT                 -> nonterm
          | T                  -> term
    rules: rule "," rules      -> rules
         | rule                -> rule
    rule: NT "~" symbols       -> arrow
    NT: "F"
      | "f"
    T: "+"
     | "-"

    %import common.WS
    %ignore WS
"""

rule_types = {
    "lsystem": ["Axiom", "Rules", "LSystem"],
    "axiom": ["Symbols", "Axiom"],
    "symbols": ["Symbol", "Symbols", "Symbols"],
    "symbol": ["Symbol", "Symbols"],
    "bracket": ["Symbols", "Symbol"],
    "nonterm": ["Nonterm", "Symbol"],
    "term": ["Term", "Symbol"],
    "rules": ["Rule", "Rules", "Rules"],
    "rule": ["Rule", "Rules"],
    "arrow": ["Nonterm", "Symbols", "Rule"],
    "F": ["Nonterm"],
    "f": ["Nonterm"],
    "+": ["Term"],
    "-": ["Term"],
}

rule_str_semantics = {
    "lsystem": lambda ax, rs: f"{ax};{rs}",
    "axiom": lambda xs: xs,
    "symbols": lambda x, xs: f"{x}{xs}",
    "symbol": lambda x: x,
    "bracket": lambda xs: f"[{xs}]",
    "nonterm": lambda nt: nt,
    "term": lambda t: t,
    "rules": lambda r, rs: f"{r},{rs}",
    "rule": lambda r: r,
    "arrow": lambda nt, xs: f"{nt}~{xs}",
    "F": lambda: "F",
    "f": lambda: "f",
    "+": lambda: "+",
    "-": lambda: "-",
}

parser = lark.Lark(lsystem_metagrammar, start='lsystem', parser='lalr')


def parse_lsys_as_ltree(s: str) -> lark.Tree:
    return parser.parse(s)


def ltree_to_ttree(node: lark.Tree | lark.Token) -> Tuple:
    """Convert a lark tree into a nested tuple of strings"""
    if isinstance(node, lark.Tree):
        return tuple([node.data] + [ltree_to_ttree(x) for x in node.children])
    else:
        return node.value


def parse_lsys(s: str) -> Tuple:
    return ltree_to_ttree(parse_lsys_as_ltree(s))


def ltree_to_sexp(node: lark.Tree | lark.Token) -> str:
    """Convert a lark tree into an s-expression string"""
    if isinstance(node, lark.Tree):
        args = [node.data] + [ltree_to_sexp(x) for x in node.children]
        return f"({' '.join(args)})"
    else:
        return node.value


def ttree_to_sexp(ttree: Tuple) -> str:
    node, *args = ttree
    if args:
        s_args = " ".join([node] + [ttree_to_sexp(arg) for arg in args])
        return f"({s_args})"
    else:
        return node


def tokenize(s: str) -> List[str]:
    return s.replace("(", "( ").replace(")", " )").split()


def group_parens(tokens: List[str]) -> Dict[int, int]:
    # TODO: test
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


def sexp_to_ttree(s: str) -> Tuple:
    # TODO: test
    tokens = tokenize(s)
    ends = group_parens(tokens)

    def to_ast_h(i: int, j: int) -> Tuple:
        args = []
        r = i
        while r < j:
            if r not in ends:  # atom
                args.append(tokens[r])
                r += 1
            else:  # s-exp
                args.append(to_ast_h(r + 1, ends[r]))
                r = ends[r] + 1
        return tuple(args)

    assert tokens[0] == "(" and tokens[-1] == ")", f"Found unbracketed token sequence: {tokens}"
    return to_ast_h(1, len(tokens) - 1)


def eval_ttree_as_str(ttree: Tuple) -> str:
    if isinstance(ttree, tuple):
        symbol, *args = ttree
    else:
        symbol, args = ttree, []
    str_args = [eval_ttree_as_str(arg) for arg in args]
    return rule_str_semantics[symbol](*str_args)


def count_unigram(ttree: Tuple) -> Dict[str, int]:
    counts = {}

    def traverse(node: Tuple):
        symbol, *args = node
        counts[symbol] = counts.get(symbol, 0) + 1
        for arg in args:
            traverse(arg)

    traverse(ttree)
    return counts


def count_bigram(ttree: Tuple) -> Dict[Tuple[str, int, str], int]:
    counts = {}

    def traverse(node: Tuple):
        name, *args = node
        for i, arg in enumerate(args):
            k = (name, i, arg[0])
            counts[k] = counts.get(k, 0) + 1
            traverse(arg)

    traverse(ttree)
    # counts["$", 0, ttree[0]] = 1      # add in transitions from start
    return counts


def multi_count_bigram(ttrees: List[Tuple]) -> Dict[Tuple[str, int, str], int]:
    counts = {}
    for ttree in ttrees:
        b = count_bigram(ttree)
        for k, v in b.items():
            counts[k] = counts.get(k, 0) + v
    return counts


def sum_counts(a: Dict[Any, float], b: Dict[Any, float]) -> Dict[Any, float]:
    return {k: a.get(k, 0) + b.get(k, 0)
            for k in a.keys() | b.keys()}


# Simplification

class ParseError(Exception):
    pass


def simplify(s: str) -> str:
    """Simplifies an L-system repr `s` using egg."""
    ltree = parse_lsys_as_ltree(s)
    sexp = ltree_to_sexp(ltree)
    sexp_simpl = eggy.simplify(sexp)
    if "nil" in sexp_simpl:
        if sexp_simpl != "nil":
            print(f"WARNING: found nil in unsimplified expression: {sexp_simpl}", file=sys.stderr)
        raise ParseError(f"Unexpected 'nil' token in simplified expr: {sexp_simpl}")
    ttree = sexp_to_ttree(sexp_simpl)
    s_simpl = eval_ttree_as_str(ttree)
    s_dedup = dedup_rules(s_simpl)
    return s_dedup


def simplify_ttree(ttree: Tuple) -> Tuple:
    sexp = ttree_to_sexp(ttree)
    sexp_simpl = eggy.simplify(sexp)
    ttree_simpl = sexp_to_ttree(sexp_simpl)
    return ttree_simpl


def dedup_rules(s: str) -> str:
    s_axiom, s_rules = s.split(";")
    rules = set(s_rules.split(","))
    s_rules = ",".join(sorted(rules, key=lambda x: (len(x), x)))
    return f"{s_axiom};{s_rules}"


def demo_simplify():
    strs = [
        "F;F~F",
        "F;F~+-+--+++--F",
        "F;F~-+F+-",
        "F;F~[F]F",
        "F;F~[FF]FF",
        "F;F~[+F-F]+F-F",
        "F;F~[F]",
        "F;F~[FF+FF]",
        "F;F~F,F~F,F~F",
        "F;F~F,F~+-F,F~F",
        "F;F~F,F~+F-",
        "F;F~F,F~+F-,F~F",
        "F;F~F,F~FF,F~F,F~FF",
        "F;F~F[+F]F,F~F,F~F[+F]F",
        "F;F~[-+-+---]F[++++]",
        "+;F~F",
        "[++];F~F",
        "[++];F~[F]",
        "[++];F~[F][+++]",
        "F;F~+",
        "F;F~F,F~+",
        "F;F~+,F~+",
        "F;F~F,F~+,F~+",
    ]
    for s in strs:
        try:
            s_simpl = simplify(s)
            print(f"{s} => {s_simpl}")
        except ParseError:
            print(f"Got nil on {s}")


def demo_to_sexp():
    strs = [
        "F;F~+",
        "F;F~F,F~+",
        "F;F~+,F~+",
        "F;F~F,F~+,F~+",
    ]
    for s in strs:
        print(to_sexp(s))


def to_sexp(s: str) -> str:
    lt = parse_lsys_as_ltree(s)
    sxp = ltree_to_sexp(lt)
    return sxp


def test_count_unigram():
    cases = [
        "(a)", {"a": 1},
        "(a a a a)", {"a": 4},
        "(a (a k) a (b k (c k) (c k)))", {"a": 3, "b": 1, "c": 2, "k": 4},
    ]
    for sexp, d in zip(cases[::2], cases[1::2]):
        ttree = sexp_to_ttree(sexp)
        out = count_unigram(ttree)
        assert out == d, f"Expected {d} but got {out}"


if __name__ == '__main__':
    demo_simplify()
    # demo_to_sexp()

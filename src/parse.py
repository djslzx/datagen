from __future__ import annotations

from typing import *
import lark
import eggy
import numpy as np
from lindenmayer import S0LSystem


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


def ltree_to_ttree(node: lark.Tree | lark.Token):
    """Convert a lark tree into a nested tuple of strings"""
    if isinstance(node, lark.Tree):
        return tuple([node.data] + [ltree_to_ttree(x) for x in node.children])
    else:
        return node.value


def ltree_to_sexp(node: lark.Tree | lark.Token) -> str:
    """Convert a lark tree into an s-expression string"""
    if isinstance(node, lark.Tree):
        args = [node.data] + [ltree_to_sexp(x) for x in node.children]
        return f"({' '.join(args)})"
    else:
        return node.value


def unigram_counts(tree: lark.Tree) -> Dict[str, np.ndarray]:
    raise NotImplementedError
    # counts: Dict[str, np.ndarray] = {nt: np.zeros(len(mg.rules[nt]))
    #                                  for nt in mg.nonterminals}
    #
    # def traverse(node: lark.Tree | lark.Token):
    #     if isinstance(node, lark.Tree):
    #         nt, i = name_to_nti(node.data)
    #         counts[nt][i] += 1
    #         for c in node.children:
    #             traverse(c)
    #     else:
    #         nt, i = name_to_nti(node.value)
    #         counts[nt][i] += 1
    #
    # traverse(tree)
    # return counts


def bigram_counts(tree: lark.Tree) -> Dict[str, Dict[str, int]]:
    raise NotImplementedError


def tokenize(s: str) -> List[str]:
    return s.replace("(", "( ").replace(")", " )").split()


def group_parens(tokens: List[str]) -> Dict[int, int]:
    ends = {}
    stack = []
    for i, token in enumerate(tokens):
        if token == "(":
            stack.append(i)
        elif token == ")":
            ends[stack.pop()] = i
    return ends


def sexp_to_ttree(s: str) -> Tuple:
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


def simplify(s: str) -> str:
    """Simplifies an L-system repr `s` using egg."""
    ltree = parse_lsys_as_ltree(s)
    sexp = ltree_to_sexp(ltree)
    sexp_simpl = eggy.simplify(sexp)
    ttree = sexp_to_ttree(sexp_simpl)
    s_simpl = eval_ttree_as_str(ttree)
    s_dedup = dedup_rules(s_simpl)
    return s_dedup


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
    ]
    for s in strs:
        s_simpl = simplify(s)
        print(f"{s} => {s_simpl}")


if __name__ == '__main__':
    demo_simplify()
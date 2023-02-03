from __future__ import annotations

from typing import *
import lark
import eggy
import numpy as np

from cfg import CFG

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

lsystem_meta_cfg = CFG("LSystem", {
    "LSystem": ["Axiom ; Rules"],
    "Axiom": ["Symbols"],
    "Symbols": ["Symbol Symbols", "Symbol"],
    "Symbol": ["[ Symbols ]", "Nonterm", "Term"],
    "Rules": ["Rule , Rules", "Rule"],
    "Rule": ["Nonterm ~ Symbols"],
    "Nonterm": ["F", "f"],
    "Term": ["+", "-"]
})

# declare rule names
_rule_names = {
    "LSystem": ["lsystem"],
    "Axiom": ["axiom"],
    "Symbols": ["symbols", "symbol"],
    "Symbol": ["bracket", "nonterm", "term"],
    "Rules": ["rules", "rule"],
    "Rule": ["arrow"],
    "Nonterm": ["F", "f"],
    "Term": ["+", "-"]
}

# build translation dicts from rule name decls
_d_name_to_nti = {
    name: (nt, i)
    for nt, names in _rule_names.items()
    for i, name in enumerate(names)
}
_d_nti_to_name = {v: k for k, v in _d_name_to_nti.items()}


def name_to_nti(s: str) -> Tuple[str, int]:
    """Given the name of the i-th rule of nonterminal NT, return (NT, i)"""
    return _d_name_to_nti[s]


def nti_to_name(s: str, i: int) -> str:
    """Given (NT, i), return the name of the i-th rule of nonterminal NT"""
    return _d_nti_to_name[s, i]


def tree_to_tuple(node: lark.Tree | lark.Token):
    """Convert a lark tree into a nested tuple of strings"""
    if isinstance(node, lark.Tree):
        return tuple([node.data] + [tree_to_tuple(x) for x in node.children])
    else:
        return node.value


def tree_to_sexp(node: lark.Tree | lark.Token) -> str:
    """Convert a lark tree into an s-expression string"""
    if isinstance(node, lark.Tree):
        args = [node.data] + [tree_to_sexp(x) for x in node.children]
        return f"({' '.join(args)})"
    else:
        return node.value


def unigram_counts(tree: lark.Tree) -> Dict[str, np.ndarray]:
    mg = lsystem_meta_cfg  # TODO: take this as an argument
    counts: Dict[str, np.ndarray] = {nt: np.zeros(len(mg.rules[nt]))
                                     for nt in mg.nonterminals}

    def traverse(node: lark.Tree | lark.Token):
        if isinstance(node, lark.Tree):
            nt, i = name_to_nti(node.data)
            counts[nt][i] += 1
            for c in node.children:
                traverse(c)
        else:
            nt, i = name_to_nti(node.value)
            counts[nt][i] += 1

    traverse(tree)
    return counts


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


def tokens_to_sexp(tokens: List[str]) -> Tuple:
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


def map_sexp(sexp: Tuple, f: Callable) -> Tuple:
    if isinstance(sexp, tuple):
        val, *children = sexp
    else:
        val, children = sexp, []
    return tuple([f(val)] + [map_sexp(child, f) for child in children])


if __name__ == '__main__':
    strs = [ 
        # "F;F~F",
        # "F;F~+-+--+++--F",
        # "F;F~-+F+-",
        # "F;F~[F]F",
        # "F;F~[FF]FF",
        # "F;F~[+F-F]+F-F",
        "F;F~[F]",
        "F;F~[FF+FF]",
        # "F;F~F,F~F,F~F",
        # "F;F~F,F~+-F,F~F",
        # "F;F~F,F~+F-",
        # "F;F~F,F~+F-,F~F",
        # "F;F~F,F~FF,F~F,F~FF",
        # "F;F~F[+F]F,F~F,F~F[+F]F",
    ]
    parser = lark.Lark(lsystem_metagrammar, start='lsystem', parser='lalr')
    for s in strs:
        ast = parser.parse(s)
        # print(ast.pretty())
        sexp = tree_to_sexp(ast)
        simplified = eggy.simplify(sexp)
        print(f"{s} ->\n"
              f"  {sexp} ->\n"
              f"  {simplified}")

        # tokens = tokenize(simplified)
        # sexp = tokens_to_sexp(tokens)
        # print(sexp)

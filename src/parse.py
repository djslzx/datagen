from __future__ import annotations

from typing import *
import lark
import eggy

lsystem_metagrammar = r"""
    lsystem: axiom ";" rules   -> lsystem
    axiom: symbols             -> axiom
    ?symbols: symbol           -> symbol
            | "[" symbols "]"  -> bracket
            | symbols symbols  -> symbols
    ?symbol: NT                -> nonterm
           | T                 -> term
    rules: rule                -> rule
         | rule "," rules      -> rules
    rule: NT "~" symbols       -> arrow
    NT: "F"
      | "f"
    T: "+"
     | "-"

    %import common.WS
    %ignore WS
"""


def tree_to_tuple(node: lark.Tree | lark.Token):
    if isinstance(node, lark.Tree):
        return tuple([node.data] + [tree_to_tuple(x) for x in node.children])
    else:
        return node.value


def tree_to_sexp(node: lark.Tree | lark.Token) -> str:
    if isinstance(node, lark.Tree):
        args = [node.data] + [tree_to_sexp(x) for x in node.children]
        return f"({' '.join(args)})"
    else:
        return node.value


def tree_to_counts(tree: lark.Tree) -> Dict:
    keys = ["lsystem", "axiom", "symbol", "bracket", "symbols",
            "rule", "rules", "arrow", "nonterm", "term", "F", "f", "+", "-"]
    counts = {key: 0 for key in keys}

    def traverse(node: lark.Tree | lark.Token):
        if isinstance(node, lark.Tree):
            counts[node.data] += 1
            for c in node.children:
                traverse(c)
        else:
            counts[node.value] += 1

    traverse(tree)
    return counts


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


mapping = [
    (("LSystem", 0), "sys"),
    (("LSystem", 1), "sys-e"),
    (("Nonterminal", 0), "F"),
    (("Terminal", 0), "+"),
    (("Terminal", 1), "-"),
    (("Axiom", 0), "ax-nt"),
    (("Axiom", 1), "ax-t"),
    (("Axiom", 2), "ax-e"),
    (("Rules", 0), "rules-add"),
    (("Rules", 1), "rules-one"),
    (("Rule", 0), "rule"),
    (("Rhs", 0), "rhs-bracket"),
    (("Rhs", 1), "rhs-nt"),
    (("Rhs", 2), "rhs-t"),
    (("Rhs", 3), "rhs-e"),
]
forward = {x: y for x, y in mapping}
backward = {y: x for x, y in mapping}


if __name__ == '__main__':
    strs = [ 
        "F;F~F",
        "F;F~+-+--+++--F",
        "F;F~F,F~F,F~F",
        "F;F~F,F~+-F,F~F",
        "F;F~F,F~+F-,F~F",
        "F;F~F,F~FF,F~F,F~FF",
        "F;F~F[+F]F,F~F,F~F[+F]F",
    ]
    parser = lark.Lark(lsystem_metagrammar, start='lsystem', parser='lalr')
    for s in strs:
        ast = parser.parse(s)
        sexp = tree_to_sexp(ast)
        simplified = eggy.simplify(sexp)
        # counts = tree_to_counts(ast)
        # print(ast.pretty())
        print(f"{s} -> {sexp} -> {simplified}")

    # for s in simplified:
    #     tokens = tokenize(s)
    #     sexp = tokens_to_sexp(tokens)
    #     print(sexp)

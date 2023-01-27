from __future__ import annotations
from lindenmayer import S0LSystem
from lark import Lark, Transformer, Tree, Token

lsystem_grammar = r"""
    lsystem: axiom ";" rules   -> lsystem
    axiom: symbols             -> axiom
    ?symbols: symbol           -> singleton
            | "[" symbols "]"  -> bracket
            | symbols symbols  -> concat
    ?symbol: NT                -> nt 
           | T                 -> t
    rules: rule                -> one_rule
         | rule "," rules      -> add_rule
    rule: NT "~" symbols       -> make_rule
    NT: "F"
      | "f"
    T: "+"
     | "-"

    %import common.WS
    %ignore WS
"""

# TODO: convert lark parse tree to tuple tree
# TODO: use tuple tree to count occurrences of rules


class LSystemTF(Transformer):
    def lsystem(self, items):
        assert len(items) == 2
        s = ";".join(str(x) for x in items)
        return S0LSystem.from_sentence(list(s))
    def axiom(self, items):
        return items[0]
    def concat(self, items):
        xs, ys = items
        return f"{xs}{ys}"
    def bracket(self, items):
        return f"[{items[0]}]"
    def rule(self, items):
        lhs, rhs = items
        return f"{lhs}~{rhs}"
    def one_rule(self, items):
        return items[0]
    def add_rule(self, items):
        r, rs = items
        return f"{r},{rs}"
    def NONTERMINAL(self, items):
        return items[0]
    def TERMINAL(self, items):
        return items[0]

def tuplify(node: Tree | Token):
    if isinstance(node, Tree):
        return tuple([node.data] + [tuplify(x) for x in node.children])
    else:
        return node.value

def exprify(node: Tree | Token) -> str:
    if isinstance(node, Tree):
        args = [node.data] + [exprify(x) for x in node.children]
        return f"({' '.join(args)})"
    else:
        return node.value

if __name__ == '__main__':
    strs = [ 
        # "-+;F~F,F~F[-F]F[-F[-F]F]F,F~F",
        # "F;F~F[+F]F,F~F,F~F[+F]F",
        # "F;F~F[+F]F,F~F[+F[+F]F]F,F~F",
        # "F;F~F,F~F[-F]F[+F[-F]F]F,F~F",
        # "+-;F~F,F~F",
        "F;F~F",
        "F;F~FF",
        "F;F~+,F~FF",
        "F;F~F[F[F]]+-+--+++--F",
        "F;F~F[+F]F,F~F,F~F[+F]F",
    ]
    parser = Lark(lsystem_grammar, start='lsystem', parser='lalr')
    tf = LSystemTF()
    for s in strs:
        ast = parser.parse(s)
        print(ast.pretty())
        print(exprify(ast))
        # print(ast)
        # print(tf.transform(ast))
        # print(s)

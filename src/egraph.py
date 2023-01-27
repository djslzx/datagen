import pdb

from lindenmayer import *
import util

samples = [
    "-+;F~F,F~F[-F]F[-F[-F]F]F,F~F",
    "F;F~F[+F]F,F~F,F~F[+F]F",
    "F;F~F[+F]F,F~F[+F[+F]F]F,F~F",
    "F;F~F,F~F[-F]F[+F[-F]F]F,F~F",
    ";F~F[-F]F[-F[-F]F]F,F~F",
    "F;F~F[-F[+F]F]F,F~F,F~,F~F",
    "FF;F~F,F~F,F~,F~F[+F]F[-F]F,F~",
    "+-;F~F,F~F",
    ";F~+-",
    "F;F~FF",
    "F;F~+,F~FF",
    "F;F~F[F[F]]+-+--+++--F",
    ";F~[F]F",
    ";F~[F][F]",
]

mapping = {
    ("LSystem", 0): "sys",
    ("LSystem", 1): "sys-e",
    ("Nonterminal", 0): "F",
    ("Terminal", 0): "+",
    ("Terminal", 1): "-",
    ("Axiom", 0): "ax-nt",
    ("Axiom", 1): "ax-t",
    ("Axiom", 2): "ax-e",
    ("Rules", 0): "rules-add",
    ("Rules", 1): "rules-one",
    ("Rule", 0): "rule",
    ("Rhs", 0): "rhs-bracket",
    ("Rhs", 1): "rhs-nt",
    ("Rhs", 2): "rhs-t",
    ("Rhs", 3): "rhs-e",
}


def to_named_ast(ast: Tree) -> Tree:
    def transform(*node):
        symbol, i, *args = node
        if not args:
            return mapping[symbol, i]
        else:
            return mapping[symbol, i], *args
    return apply_to_tree(ast, transform)


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


def to_ast(tokens: List[str]) -> Tree:
    ends = group_parens(tokens)

    def to_ast_h(i: int, j: Optional[int]) -> Tree:
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


asts = [
    "(a)",
    "(a (b))",
    "(a (b c) d)",
    "(a (b c (d e) (f)) g)",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rules-add (rule F (rhs-nt F rhs-e)) (rules-one (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e))))))))",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rhs-nt F rhs-e)))) (rules-one (rule F (rhs-nt F rhs-e))))))",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F rhs-e)) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t - (rhs-nt F rhs-e)) (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F (rhs-bracket (rhs-t - (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rhs-nt F rhs-e)))))) (rules-one (rule F (rhs-nt F rhs-e))))))",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t - (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rhs-nt F rhs-e)))) (rules-add (rule F (rhs-nt F rhs-e)) (rules-add (rule F rhs-e) (rules-one (rule F (rhs-nt F rhs-e)))))))",
    # "(sys (ax-nt F (ax-nt F ax-e)) (rules-add (rule F (rhs-nt F rhs-e)) (rules-add (rule F rhs-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F (rhs-bracket (rhs-t - (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))))) (rules-one (rule F rhs-e))))))",
    # "(sys (ax-nt F ax-e) (rules-one (rule F (rhs-nt F (rhs-nt F rhs-e)))))",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-t + rhs-e)) (rules-one (rule F (rhs-nt F (rhs-nt F rhs-e))))))",
    # "(sys (ax-nt F ax-e) (rules-one (rule F (rhs-nt F (rhs-bracket (rhs-nt F (rhs-bracket (rhs-nt F rhs-e) rhs-e)) (rhs-nt F rhs-e))))))",
]

for ast in asts:
    tokens = tokenize(ast)
    print(tokens)
    print(group_parens(tokens))
    print(to_ast(tokens))
    print()

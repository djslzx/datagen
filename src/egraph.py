import pdb

from lindenmayer import *
from learner import to_flat_string
import util

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


def to_named_ast(ast: Tree) -> Tree:
    def transform(*node):
        symbol, i, *args = node
        if not args:
            return forward[symbol, i]
        else:
            return forward[symbol, i], *args
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

    def to_ast_h(i: int, j: int) -> Tree:
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


def map_ast(ast: Tree, f: Callable) -> Tree:
    if isinstance(ast, tuple):
        val, *children = ast
    else:
        val, children = ast, []
    return tuple([f(val)] + [map_ast(child, f) for child in children])


samples = [
    # "-+;F~F,F~F[-F]F[-F[-F]F]F,F~F",
    # "F;F~F[+F]F,F~F,F~F[+F]F",
    # "F;F~F[+F]F,F~F[+F[+F]F]F,F~F",
    # "F;F~F,F~F[-F]F[+F[-F]F]F,F~F",
    # ";F~F[-F]F[-F[-F]F]F,F~F",
    # "F;F~F[-F[+F]F]F,F~F,F~,F~F",
    # "FF;F~F,F~F,F~,F~F[+F]F[-F]F,F~",
    # "+-;F~F,F~F",
    # ";F~+-",
    # "F;F~FF",
    # "F;F~+,F~FF",
    # "F;F~F[F[F]]+-+--+++--F",
    # ";F~[F]F",
    # ";F~[F][F]",
    "F;F~F[+F]F,F~F,F~F[+F]F",
]

strs = [
    "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rules-add (rule F (rhs-nt F rhs-e)) (rules-one (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e))))))))",
    "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rules-add (rule F (rhs-nt F rhs-e)) (rules-one (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e))))))))",
    # "(a)",
    # "(a (b))",
    # "(a (b c) d)",
    # "(a (b c (d e) (f)) g)",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rules-add (rule F (rhs-nt F rhs-e)) (rules-one (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e))))))))",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rhs-nt F rhs-e)))) (rules-one (rule F (rhs-nt F rhs-e))))))",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F rhs-e)) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t - (rhs-nt F rhs-e)) (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F (rhs-bracket (rhs-t - (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rhs-nt F rhs-e)))))) (rules-one (rule F (rhs-nt F rhs-e))))))",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t - (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))) (rhs-nt F rhs-e)))) (rules-add (rule F (rhs-nt F rhs-e)) (rules-add (rule F rhs-e) (rules-one (rule F (rhs-nt F rhs-e)))))))",
    # "(sys (ax-nt F (ax-nt F ax-e)) (rules-add (rule F (rhs-nt F rhs-e)) (rules-add (rule F rhs-e) (rules-add (rule F (rhs-nt F (rhs-bracket (rhs-t + (rhs-nt F rhs-e)) (rhs-nt F (rhs-bracket (rhs-t - (rhs-nt F rhs-e)) (rhs-nt F rhs-e)))))) (rules-one (rule F rhs-e))))))",
    # "(sys (ax-nt F ax-e) (rules-one (rule F (rhs-nt F (rhs-nt F rhs-e)))))",
    # "(sys (ax-nt F ax-e) (rules-add (rule F (rhs-t + rhs-e)) (rules-one (rule F (rhs-nt F (rhs-nt F rhs-e))))))",
    # "(sys (ax-nt F ax-e) (rules-one (rule F (rhs-nt F (rhs-bracket (rhs-nt F (rhs-bracket (rhs-nt F rhs-e) rhs-e)) (rhs-nt F rhs-e))))))",
]

# convert Lsystem encoding string into a parse tree, then output the parse tree as a string
for sample in samples:
    ast = parse_lsystem_to_ast(sample)
    nast = to_named_ast(ast)
    nast_str = str(nast).replace("'", "").replace(",", "")
    print(sample, nast_str, sep='\n')
    strs.append(nast_str)

# read the parse tree string in back into an L-system ast
for s in strs:
    tokens = tokenize(s)
    ast = to_ast(tokens)
    lsys_ast = map_ast(ast, lambda x: "_".join(str(e) for e in backward[x]))
    lsys_str = to_flat_string(lsys_ast)

    print(s)
    # print(group_parens(tokens))
    # print(to_ast(tokens))
    # print(lsys_ast)
    print(lsys_str)

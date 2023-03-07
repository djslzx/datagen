from __future__ import annotations
from typing import *
import numpy as np

import parse
from lang import Language, Tree, Grammar
from featurizers import TextClassifier


class Regex(Language):
    """
    Models the domain of regular expressions using the grammar from the paper
    "Learning to learn generative programs with Memoised Wake-Sleep" (Hewitt et al.)

    Token types:
    .       : any character
    \\w      : alphanumeric char
    \\d      : digit
    \\u      : uppercase char
    \\l      : lowercase char
    \\s      : whitespace char
    (φ contains specific probabilities for each allowed character)

    Operator  types:
    E?      : E (ϕ_?) | ϵ (1 - ϕ_?)
    E*      : E+ (ϕ_*) | ϵ (1 - ϕ_*)
    E+      : EE*
    E₁ | E₂ : E₁ (ϕ_|) | E₂ (1 - ϕ_|)
    (φ contains production probabilities)
    """

    metagrammar = r"""
        e: e "?"         -> maybe
         | e "*"         -> star
         | e "+"         -> plus
         | "(" e ")"     -> bracket
         | e "|" e       -> or
         | e e           -> seq
         | "."           -> dot
         | "\w"          -> alpha
         | "\d"          -> digit
         | "\p"          -> upper
         | "\l"          -> lower
         | "\s"          -> whitespace
         | /./           -> literal
    """
    types = {
        # operators
        "maybe": ["Regex", "Regex"],
        "star": ["Regex", "Regex"],
        "plus": ["Regex", "Regex"],
        "bracket": ["Regex", "Regex"],
        "or": ["Regex", "Regex", "Regex"],
        "seq": ["Regex", "Regex", "Regex"],

        # character classes
        "dot": ["Regex"],
        "alpha": ["Regex"],
        "digit": ["Regex"],
        "upper": ["Regex"],
        "lower": ["Regex"],
        "whitespace": ["Regex"],
        "literal": ["Regex"],
    }
    upper = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    lower = list("abcdefghijklmnopqrstuvwxyz")
    digit = list("0123456789")
    alpha = upper + lower + digit
    whitespace = list("\t\r\n ")
    dot = alpha + whitespace + list("~`!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?")
    char_classes = {
        "dot": dot,
        "alpha": alpha,
        "digit": digit,
        "upper": upper,
        "lower": lower,
        "whitespace": whitespace,
    }

    def __init__(self, eval_weights: Dict[str, np.ndarray] = None):
        """
        The grammar contains weights over the space of regular expressions,
        whereas the `eval_weights`, analogous to φ from the paper, determine
        execution semantics.

        TODO: should the evaluator eval_weights be learned as in Luke Hewitt's paper?
        """
        super().__init__(parser_grammar=Regex.metagrammar,
                         start="e",
                         model=Grammar.from_components(Regex.types, gram=2),
                         featurizer=TextClassifier())
        self.eval_weights = eval_weights if eval_weights is not None else Regex.uniform_weights()

    @staticmethod
    def uniform_weights() -> Dict[str, np.ndarray]:
        def uniform(k: int) -> np.ndarray:
            return np.ones(k) / k

        return {
            # operators (nodes)
            "maybe": uniform(2),
            "star": uniform(2),
            "plus": uniform(1),
            "or": uniform(2),

            # character classes (leaves)
            "dot": uniform(len(Regex.dot)),
            "alpha": uniform(len(Regex.alpha)),
            "digit": uniform(len(Regex.digit)),
            "upper": uniform(len(Regex.upper)),
            "lower": uniform(len(Regex.lower)),
            "whitespace": uniform(len(Regex.whitespace)),
            # literals don't need probabilities
        }

    def eval(self, t: Tree, env: Dict[str, str]) -> str:
        def flip() -> int:
            return np.random.multinomial(n=1, pvals=self.eval_weights[t.value]).argmax()

        if t.is_leaf():
            # leaf must be a character class
            i = flip()
            return Regex.char_classes[t.value][i]
        else:
            # node must be an operator
            if t.value == "maybe":
                c = t.children[0]
                return "" if flip() == 0 else self.eval(c, env)
            elif t.value == "star":
                c = t.children[0]
                # optimize as while loop
                out = ""
                while flip() == 0:
                    out += self.eval(c, env)
                return out
            elif t.value == "plus":
                # E+ => EE* = (plus c) => (seq c (star c))
                c = t.children[0]
                return self.eval(c, env) + self.eval(Tree("star", c), env)
            elif t.value == "bracket":
                c = t.children[0]
                return self.eval(c, env)
            elif t.value == "or":
                a, b = t.children
                return self.eval(a, env) if flip() == 0 else self.eval(b, env)
            elif t.value == "seq":
                a, b = t.children
                return self.eval(a, env) + self.eval(b, env)
            else:
                raise ValueError(f"Regex internal nodes must be operators, "
                                 f"but found char class {t.value} in tree {t}")

    def simplify(self, t: Tree) -> Tree:
        raise NotImplementedError


if __name__ == "__main__":
    examples = [
        ".",
        r"-\p-(\p-)+",
        r"-\p-\p-.*",
        r"sch\d\d\d@sfusd.\l\l\l",
    ]
    r = Regex()
    for ex in examples:
        t = r.parse(ex)
        print(t)

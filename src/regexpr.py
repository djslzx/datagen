from __future__ import annotations
from typing import *
import numpy as np

from lang import Language, Tree, Grammar
from featurizers import TextClassifier


class Regex(Language):
    """
    Models the domain of regular expressions using the grammar from the paper
    "Learning to learn generative programs with Memoised Wake-Sleep" (Hewitt et al.)

    Token types:
    \\.      : any character (need backslash to distinguish from literal .)
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
         | "\."           -> dot
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
        "literal": ["Char", "Regex"],
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
    # add all allowed characters to Char type for literals
    types.update({x: ["Char"] for x in dot})

    def __init__(self, eval_weights: Dict[str, np.ndarray] = None):
        """
        The grammar contains weights over the space of regular expressions,
        whereas the `eval_weights`, analogous to φ from the paper, determine
        execution semantics.

        TODO: should the evaluator eval_weights be learned as in Luke Hewitt's paper?
        """
        super().__init__(parser_grammar=Regex.metagrammar,
                         parser_start="e",
                         root_type="Regex",
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
            # plus: E+ is implemented as EE*
            "or": uniform(2),

            # character classes (leaves)
            "dot": uniform(len(Regex.dot)),
            "alpha": uniform(len(Regex.alpha)),
            "digit": uniform(len(Regex.digit)),
            "upper": uniform(len(Regex.upper)),
            "lower": uniform(len(Regex.lower)),
            "whitespace": uniform(len(Regex.whitespace)),
            "literal": uniform(1),
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
            elif t.value == "literal":
                return t.children[0].value
            else:
                raise ValueError(f"Regex internal nodes must be operators, "
                                 f"but found char class {t.value} in tree {t}")

    @property
    def str_semantics(self) -> Dict:
        return {
            "maybe": lambda r: f"{r}?",
            "star": lambda r: f"{r}*",
            "plus": lambda r: f"{r}+",
            "bracket": lambda r: f"({r})",
            "or": lambda r, s: f"{r}|{s}",
            "seq": lambda r, s: f"{r}{s}",
            "dot": lambda: r"\.",
            "alpha": lambda: r"\w",
            "digit": lambda: r"\d",
            "upper": lambda: r"\p",
            "lower": lambda: r"\l",
            "whitespace": lambda: r"\s",
            "literal": lambda x: x,
        }

    def simplify(self, t: Tree) -> Tree:
        raise NotImplementedError


if __name__ == "__main__":
    examples = [
        r"\.",
        r"-\p-(\p-)+",
        r"-\p-\p-\.*",
        r"sch\d\d\d@sfusd.\l\l\l",
    ]
    r = Regex()
    for ex in examples:
        p = r.parse(ex)
        print(ex, r.to_str(p))
        for _ in range(10):
            print(r.eval(p, env={}))

    print("Fitting...")
    corpus = [r.parse(ex) for ex in examples]
    r.fit(corpus, alpha=1)

    print("Sampling...")
    for _ in range(10):
        p = r.sample()
        print(p, r.to_str(p))

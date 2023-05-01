from __future__ import annotations

import pickle
from typing import Dict, Any
import numpy as np
from sys import stderr

from lang import Language, Tree, Grammar
from featurizers import TextClassifier
from examples import regex_handcoded_examples


class Regex(Language):
    r"""
    Models the domain of regular expressions using the grammar from the paper
    "Learning to learn generative programs with Memoised Wake-Sleep" (Hewitt et al.),
    with some extras (seq, bracket)
    """

    metagrammar = r"""
        e: e "?"         -> maybe
         | e "*"         -> star
         | e "+"         -> plus
         | "(" e ")"     -> bracket
         | e "|" e       -> or
         | e e           -> seq
         | "."           -> any
         | "\w"          -> alpha
         | "\d"          -> digit
         | "\p"          -> upper
         | "\l"          -> lower
         | "\s"          -> whitespace
         | ANY           -> literal
         | "\\" ANY       -> escaped
        
        ANY: /./ 
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
        "any": ["Regex"],
        "alpha": ["Regex"],
        "digit": ["Regex"],
        "upper": ["Regex"],
        "lower": ["Regex"],
        "whitespace": ["Regex"],
        "literal": ["Char", "Regex"],
        "escaped": ["EscapeChar", "Regex"],
    }

    upper = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    lower = list("abcdefghijklmnopqrstuvwxyz")
    digit = list("0123456789")
    alpha = upper + lower + digit
    whitespace = list("\t ")
    any = alpha + whitespace + list("~`!@#$%^&-_=[]{}\\;:'\",<>/")
    escaped = list(".()|?*+")
    char_classes = {
        "any": any,
        "alpha": alpha,
        "digit": digit,
        "upper": upper,
        "lower": lower,
        "whitespace": whitespace,
        "escaped": escaped,
    }
    # add all allowed characters to Char, EscapeChar types for literals
    types.update({x: ["Char"] for x in any + escaped})
    types.update({x: ["EscapeChar"] for x in escaped})

    def __init__(self, eval_weights: Dict[str, np.ndarray] = None, gram=2):
        """
        The grammar contains weights over the space of regular expressions,
        whereas the `eval_weights`, analogous to φ from the paper, determine
        execution semantics.

        TODO: should the evaluator eval_weights be learned as in Luke Hewitt's paper?
        """
        super().__init__(parser_grammar=Regex.metagrammar,
                         parser_start="e",
                         root_type="Regex",
                         model=Grammar.from_components(Regex.types, gram=gram),
                         featurizer=TextClassifier())
        self.eval_weights = eval_weights if eval_weights is not None else Regex.uniform_weights()

    def none(self) -> Any:
        return ""

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
            "any": uniform(len(Regex.any)),
            "alpha": uniform(len(Regex.alpha)),
            "digit": uniform(len(Regex.digit)),
            "upper": uniform(len(Regex.upper)),
            "lower": uniform(len(Regex.lower)),
            "whitespace": uniform(len(Regex.whitespace)),
            "literal": uniform(1),
            "escaped": uniform(1),
        }

    def eval(self, t: Tree, env: Dict[str, str] = None) -> str:
        if env is None: env = {}

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
            elif t.value == "escaped":
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
            "any": lambda: r".",
            "alpha": lambda: r"\w",
            "digit": lambda: r"\d",
            "upper": lambda: r"\p",
            "lower": lambda: r"\l",
            "whitespace": lambda: r"\s",
            "literal": lambda x: x,
            "escaped": lambda x: f"\\{x}",
        }

    def simplify(self, t: Tree) -> Tree:
        print("WARNING: simplification for regular expressions not implemented", file=stderr)
        return t


def examples():
    """
    Returns a dictionary D with keys 'name' and 'data':
    - D['name'] is the name of the concept
    - D['data'] is a list of strings (ostensibly) drawn from a regular expression
    """
    obj = pickle.load(open("../datasets/csv/csv.p", "rb"))
    return obj


if __name__ == "__main__":
    examples = [
        r"0",
        r"0",
        r"0",
        r"0",
        r"0",
        # r"\(909\) \d\d\d-\d\d\d\d",
        # r"\.",
        # r"-\p-(\p-)+",
        # r"-\p-\p-\.*",
        # r"sch\d\d\d@sfusd.\l\l\l",
        # r".{10}",
        # r"\.{2,3}",
        # r"(\w){2,10}",
        # r"(\w)\1{2,}",
    ]  # + regex_handcoded_examples
    r = Regex()
    for ex in examples:
        p = r.parse(ex)
        s = r.to_str(p)
        print(ex, s, p)
        for _ in range(10):
            print(r.eval(p, env={}))

    print("Fitting...")
    corpus = [r.parse(ex) for ex in examples]
    r.fit(corpus, alpha=1e-10)
    print(r.model)

    print("Sampling...")
    for _ in range(10):
        p = r.sample()
        print(r.to_str(p), p)
        for _ in range(3):
            print("  " + r.eval(p, env={}))

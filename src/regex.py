from __future__ import annotations
from typing import *

import parse
from parse import lark

regex_metagrammar = r"""
    e: e "?"         -> maybe
     | e "+"         -> plus
     | e "*"         -> star
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
    "maybe": ["Regex", "Regex"],
    "plus": ["Regex", "Regex"],
    "star": ["Regex", "Regex"],
    "bracket": ["Regex", "Regex"],
    "or": ["Regex", "Regex", "Regex"],
    "seq": ["Regex", "Regex", "Regex"],
    "dot": ["Regex"],
    "alpha": ["Regex"],
    "digit": ["Regex"],
    "upper": ["Regex"],
    "lower": ["Regex"],
    "whitespace": ["Regex"],
    "literal": ["Regex"],
}

str_semantics = {
    "maybe": lambda e: f"{e}?",
    "plus": lambda e: f"{e}+",
    "star": lambda e: f"{e}*",
    "bracket": lambda e: f"({e})",
    "or": lambda a, b: f"{a}|{b}",
    "seq": lambda a, b: f"{a}{b}",
    "dot": lambda: ".",
    "alpha": lambda: r"\w",
    "digit": lambda: r"\d",
    "upper": lambda: r"\p",
    "lower": lambda: r"\l",
    "whitespace": lambda: r"\s",
    "literal": lambda e: e,
}

if __name__ == "__main__":
    examples = [
        ".",
        r"-\p-(\p-)+",
        r"-\p-\p-.*",
        r"sch\d\d\d@sfusd.\l\l\l",
    ]

    mg = regex_metagrammar
    p = lark.Lark(grammar=mg, start="e", parser="lalr")
    for example in examples:
        x = p.parse(example)
        ttree = parse.ltree_to_ttree(x)
        s = parse.eval_ttree_as_str(str_semantics, ttree)
        print(s, example)
        print(x.pretty())


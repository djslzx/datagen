from __future__ import annotations

from typing import List, Dict, Any

from lang.tree import Language, Tree, Grammar, ParseError, Featurizer


class Ant(Language):
    """
    Mujoco ant domain from Programmatic RL without Oracles.
    """
    grammar = r"""
        e: "if" "(" b ")" "then" vec "else" "("? e ")"? -> if
         | vec                                          -> c
        b: NUMBER "+" vec "* X >= 0"                    -> b
        vec: "[" (NUMBER ","?)* "]"                     -> vec
        
        %import common.NUMBER
        %import common.WS
        %ignore WS
    """

    def __init__(self):
        super().__init__(
            parser_grammar=Ant.grammar,
            parser_start="e",
            root_type="Reals",
            model=None,
            featurizer=AntFeaturizer(),
        )

    def sample(self) -> Tree:
        raise NotImplementedError

    def fit(self, corpus: List[Tree], alpha):
        raise NotImplementedError

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> Any:
        raise NotImplementedError

    @property
    def str_semantics(self) -> Dict:
        return {
            "if": lambda b, c, e: f"if ({b}) then {c} else ({e})",
            "c": lambda v: f"{v}",
            "b": lambda n, v: f"{n} + {v} * X >= 0",
            "vec": lambda *xs: "[ " + " ".join(xs) + " ]",
        }


class AntFeaturizer(Featurizer):
    """
    Ant features: points along trajectory of ant?
    """
    pass


if __name__ == "__main__":
    lang = Ant()
    s = "if (1.0 + [0 1 2] * X >= 0) then [1 2 3] else [4, 5, 6]"
    tree = lang.parse(s)
    print(tree, lang.to_str(tree), sep="\n")

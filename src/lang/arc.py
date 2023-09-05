from typing import Dict, Any

from lang.tree import Language, Tree, Grammar, ParseError
from featurizers import Featurizer, ResnetFeaturizer
import util
import blocks


class Blocks(Language):
    """
    Defines the ARC domain as a language for novelty search.
    """
    types = {
        # bool
        "nil": ["Bool"],
        "not": ["Bool", "Bool"],
        "lt": ["Int", "Int", "Bool"],
        "and": ["Bool", "Bool", "Bool"],
        "or": ["Bool", "Bool", "Bool"],
        # int
        "int": ["Int", "Int"],
        "xmax": ["Int"],
        "ymax": ["Int"],
        "z": ["Int", "Int"],
        "plus": ["Int", "Int", "Int"],
        "minus": ["Int", "Int", "Int"],
        "times": ["Int", "Int", "Int"],
        "if": ["Bool", "Int", "Int", "Int"],
        # points
        "point": ["Int", "Int", "Point"],
        # color
        "color": ["Int", "Color"],
        # bitmaps
        "line": ["Point", "Point", "Color", "Bmp"],
        "rect": ["Point", "Point", "Color", "Bmp"],
        "sprite": ["Int", "Point", "Bmp"],
        # transforms
        "seq": ["Bmp", "Bmp", "Bmp"],
        "apply": ["Transform", "Bmp", "Bmp"],
        "repeat": ["Transform", "Int", "Transform"],
        "hflip": ["Transform"],
        "vflip": ["Transform"],
        "translate": ["Int", "Int", "Transform"],
        "compose": ["Transform", "Transform", "Transform"],
    }
    types.update({k: ["Color"] for k in range(5)})
    types.update({k: ["Int"] for k in range(10)})
    metagrammar = r"""
        bmp: "(" "seq" bmp bmp ")"            -> seq
           | "(" "apply" transform bmp ")"    -> apply
           | "(" "line" point point color ")" -> line
           | "(" "rect" point point color ")" -> rect
           | "(" "sprite" int point ")"       -> sprite
        transform: "hflip" 
                 | "vflip" 
                 | "(" "translate" int int ")"           -> translate
                 | "(" "compose" transform transform ")" -> compose
                 | "(" "repeat" transform int ")"        -> repeat
        color: NUMBER
        int: NUMBER 
           | "xmax" 
           | "ymax" 
           | "(" "z" NUMBER ")"            -> z
           | "(" "if" bool int int ")"     -> if
           | "(" "plus" int int ")"        -> plus
           | "(" "minus" int int ")"       -> minus
           | "(" "times" int int ")"       -> times
        bool: "nil" 
            | "(" "not" bool ")"      -> not
            | "(" "and" bool bool ")" -> and
            | "(" "or" bool bool ")"  -> or
            | "(" "lt" int int ")"    -> lt
        point: "(" "point" int int ")" -> point
        
        %import common.WS
        %import common.NUMBER
        %ignore WS
    """

    def __init__(self):
        model = Grammar.from_components(Blocks.types, gram=1)
        super().__init__(
            parser_grammar=Blocks.metagrammar,
            parser_start="bmp",
            root_type="Bmp",
            model=model,
            featurizer=ResnetFeaturizer(),
        )

    def eval(self, t: Tree, env: Dict[str, Any] = None):
        print(t)

    @property
    def str_semantics(self) -> Dict:
        semantics = {
            "int": lambda x: str(x),
            "color": lambda x: str(x),
            "nil": lambda: "nil",
            "not": lambda x: f"(not {x})",
            "lt": lambda x, y: f"(lt {x} {y})",
            "and": lambda x, y: f"(and {x} {y})",
            "or": lambda x, y: f"(or {x} {y})",
            "xmax": lambda: "xmax",
            "ymax": lambda: "ymax",
            "z": lambda x: f"(z {x})",
            "plus": lambda x, y: f"(plus {x} {y})",
            "minus": lambda x, y: f"(minus {x} {y})",
            "times": lambda x, y: f"(times {x} {y})",
            "if": lambda b, x, y: f"(if {b} {x} {y})",
            "line": lambda p1, p2, c: f"(line {p1} {p2} {c})",
            "point": lambda x, y: f"(point {x} {y})",
            "rect": lambda p1, p2, c: f"(rect {p1} {p2} {c})",
            "sprite": lambda i, p: f"(sprite {i} {p})",
            "seq": lambda x, y: f"(seq {x} {y})",
            "apply": lambda t, x: f"(apply {t} {x})",
            "repeat": lambda t, n: f"(repeat {t} {n})",
            "hflip": lambda: "hflip",
            "vflip": lambda: "vflip",
            "translate": lambda x, y: f"(translate {x} {y})",
            "compose": lambda t1, t2: f"(compose {t1} {t2})",
        }
        semantics.update({k: lambda: str(k) for k in range(10)})
        return semantics


if __name__ == "__main__":
    b = Blocks()
    print(b.model)
    examples = [
        "(rect (point 1 2) (point 1 2) 1)",
        "(line (point 1 2) (point 3 4) 1)",
        "(seq (line (point 1 2) (point 3 4) 1) "
        "     (rect (point 1 2) (point 1 2) 1))",
        "(apply hflip (line (point 1 2) (point 1 4) 1))",
    ]
    for s in examples:
        t = b.parse(s)
        print(t.to_sexp())
        print(b.to_str(t))
        print(b.eval(t))

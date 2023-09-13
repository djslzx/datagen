from typing import Dict, Any
from matplotlib import pyplot as plt
import einops as ein

from lang.tree import Language, Tree, Grammar
from featurizers import Featurizer, ResnetFeaturizer
import blocks.grammar as grammar
import util


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
        transform: "hflip" -> hflip
                 | "vflip" -> vflip
                 | "(" "translate" int int ")"           -> translate
                 | "(" "compose" transform transform ")" -> compose
                 | "(" "repeat" transform int ")"        -> repeat
        color: NUMBER
        int: NUMBER 
           | "xmax" -> xmax
           | "ymax" -> ymax
           | "(" "z" NUMBER ")"            -> z
           | "(" "if" bool int int ")"     -> if
           | "(" "plus" int int ")"        -> plus
           | "(" "minus" int int ")"       -> minus
           | "(" "times" int int ")"       -> times
        bool: "nil" -> nil
            | "(" "not" bool ")"      -> not
            | "(" "and" bool bool ")" -> and
            | "(" "lt" int int ")"    -> lt
        point: "(" "point" int int ")" -> point
        
        %import common.WS
        %import common.NUMBER
        %ignore WS
    """

    def __init__(self, gram: int, featurizer: Featurizer):
        assert gram in {1, 2}
        model = Grammar.from_components(Blocks.types, gram=gram)
        super().__init__(
            parser_grammar=Blocks.metagrammar,
            parser_start="bmp",
            root_type="Bmp",
            model=model,
            featurizer=featurizer,
        )

    def _to_obj(self, t: Tree, env: Dict[str, Any] = None) -> grammar.Expr:
        if t.is_leaf():
            if t.value == "hflip":
                return grammar.HFlip()
            elif t.value == "vflip":
                return grammar.VFlip()
            elif t.value == "xmax":
                return grammar.XMax()
            elif t.value == "ymax":
                return grammar.YMax()
            raise ValueError(f"Unexpected leaf: {t}")
        elif t.value == "int" or t.value == "color":
            return grammar.Num(n=int(t.children[0].value))
        elif t.value == "nil":
            return grammar.Nil()
        elif t.value == "not":
            c = t.children[0]
            b = self._to_obj(c, env)
            return grammar.Not(b)
        elif t.value == "lt":
            a, b = t.children
            return grammar.Lt(self._to_obj(a, env), self._to_obj(b, env))
        elif t.value == "and":
            a, b = t.children
            return grammar.And(self._to_obj(a, env), self._to_obj(b, env))
        elif t.value == "z":
            i = self._to_obj(t.children[0], env)
            return grammar.Z(i)
        elif t.value == "plus":
            a, b = t.children
            return grammar.Plus(self._to_obj(a, env), self._to_obj(b, env))
        elif t.value == "minus":
            a, b = t.children
            return grammar.Minus(self._to_obj(a, env), self._to_obj(b, env))
        elif t.value == "times":
            a, b = t.children
            return grammar.Times(self._to_obj(a, env), self._to_obj(b, env))
        elif t.value == "if":
            b, x, y = t.children
            return grammar.If(self._to_obj(b, env), self._to_obj(x, env), self._to_obj(y, env))
        elif t.value == "point":
            x, y = t.children
            return self._to_obj(x, env), self._to_obj(y, env)
        elif t.value == "line":
            p1, p2, c = t.children
            x1, y1 = self._to_obj(p1, env)
            x2, y2 = self._to_obj(p2, env)
            c = self._to_obj(c, env)
            return grammar.CornerLine(x1, y1, x2, y2, c)
        elif t.value == "rect":
            p1, p2, c = t.children
            x1, y1 = self._to_obj(p1, env)
            x2, y2 = self._to_obj(p2, env)
            c = self._to_obj(c, env)
            return grammar.CornerRect(x1, y1, x2, y2, c)
        elif t.value == "seq":
            x, y = t.children
            return grammar.Join(self._to_obj(x, env), self._to_obj(y, env))
        elif t.value == "apply":
            f, x = t.children
            return grammar.Apply(self._to_obj(f, env), self._to_obj(x, env))
        elif t.value == "repeat":
            f, n = t.children
            return grammar.Repeat(self._to_obj(f, env), self._to_obj(n, env))
        elif t.value == "translate":
            x, y = t.children
            return grammar.Translate(self._to_obj(x, env), self._to_obj(y, env))
        elif t.value == "compose":
            f, g = t.children
            return grammar.Compose(self._to_obj(f, env), self._to_obj(g, env))

    def eval(self, t: Tree, env: Dict[str, Any] = None):
        o = self._to_obj(t, env)
        evaluator = grammar.Eval(env=env, height=16, width=16)
        bmp = o.accept(evaluator)
        return ein.repeat(bmp, "h w -> h w c", c=3)

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
    b = Blocks(featurizer=ResnetFeaturizer(), gram=2)
    print(b.model)
    examples = [
        "(rect (point 1 2) (point 1 2) 1)",
        "(rect (point 1 1) (point xmax ymax) 1)",
        "(rect (point 1 1) (point 15 15) 1)",
        "(line (point 1 2) (point 3 4) 1)",
        "(seq (line (point 1 2) (point 3 4) 1) "
        "     (rect (point 1 2) (point 1 2) 1))",
        "(apply hflip (line (point 1 2) (point 1 4) 1))",
    ]
    for s in examples:
        t = b.parse(s)
        print(t.to_sexp())
        print(b.to_str(t))
        img = b.eval(t)
        plt.imshow(img)
        plt.show()

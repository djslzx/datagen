from typing import Dict, Any, Optional
from collections import ChainMap
import numpy as np
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
        # bitmaps
        "line": ["Point", "Point", "Int", "Bmp"],
        "rect": ["Point", "Point", "Int", "Bmp"],
        "seq": ["Bmp", "Bmp", "Bmp"],
        "apply": ["Transform", "Bmp", "Bmp"],
        # transforms
        "hflip": ["Transform"],
        "vflip": ["Transform"],
        "translate": ["Int", "Int", "Transform"],
        "compose": ["Transform", "Transform", "Transform"],
        "repeat": ["Transform", "Int", "Transform"],
        # points
        "point": ["Int", "Int", "Point"],
        # int
        "int": ["LiteralInt", "Int"],
        "xmax": ["Int"],
        "ymax": ["Int"],
        "z": ["LiteralInt", "Int"],
        "plus": ["Int", "Int", "Int"],
        "minus": ["Int", "Int", "Int"],
        "times": ["Int", "Int", "Int"],
        "if": ["Bool", "Int", "Int", "Int"],
        # lit
        "lit": ["Const", "LiteralInt"],
        # bool
        "nil": ["Bool"],
        "not": ["Bool", "Bool"],
        "and": ["Bool", "Bool", "Bool"],
        "lt": ["Int", "Int", "Bool"],
    }
    types.update({k: ["Const"] for k in range(10)})

    parser_grammar = r"""        
        bmp: "(" "line" point point int ")" -> line
           | "(" "rect" point point int ")" -> rect
           | "(" "seq" bmp bmp ")"          -> seq
           | "(" "apply" transform bmp ")"  -> apply
        transform: "hflip"                               -> hflip
                 | "vflip"                               -> vflip
                 | "(" "translate" int int ")"           -> translate
                 | "(" "compose" transform transform ")" -> compose
                 | "(" "repeat" transform int ")"        -> repeat
        point: "(" "point" int int ")"     -> point
        int: n                             -> int
           | "xmax"                        -> xmax
           | "ymax"                        -> ymax
           | "(" "z" n ")"                 -> z
           | "(" "plus" int int ")"        -> plus
           | "(" "minus" int int ")"       -> minus
           | "(" "times" int int ")"       -> times
           | "(" "if" bool int int ")"     -> if
        n: NUMBER                          -> lit
        bool: "nil"                        -> nil
            | "(" "not" bool ")"           -> not
            | "(" "and" bool bool ")"      -> and
            | "(" "lt" int int ")"         -> lt

        
        %import common.WS
        %import common.NUMBER
        %ignore WS
    """

    def __init__(self, gram: int, featurizer: Featurizer, env: Optional[Dict] = None, height=16, width=16):
        assert gram in {1, 2}
        model = Grammar.from_components(Blocks.types, gram=gram)
        super().__init__(
            parser_grammar=Blocks.parser_grammar,
            parser_start="bmp",
            root_type="Bmp",
            model=model,
            featurizer=featurizer,
        )
        self.env = {} if env is None else env
        self.height = height
        self.width = width

    def _to_obj(self, t: Tree, env: Dict[str, Any] = None):
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
        elif t.value == "lit":
            return int(t.children[0].value)
        elif t.value == "int":
            c = t.children[0]
            return grammar.Num(self._to_obj(c, env))
        elif t.value == "nil":
            return grammar.Nil()
        elif t.value == "not":
            c = t.children[0]
            return grammar.Not(self._to_obj(c, env))
        elif t.value == "lt":
            a, b = t.children
            return grammar.Lt(self._to_obj(a, env), self._to_obj(b, env))
        elif t.value == "and":
            a, b = t.children
            return grammar.And(self._to_obj(a, env), self._to_obj(b, env))
        elif t.value == "z":
            c = t.children[0]
            return grammar.Z(self._to_obj(c, env))
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
        env = ChainMap(self.env, env)
        try:
            o = self._to_obj(t, env)
        except ValueError:
            raise ValueError(f"Failed to evaluate expression {t.to_sexp()}")

        evaluator = grammar.Eval(env=env, height=16, width=16)
        try:
            bmp = o.accept(evaluator).numpy().astype(np.uint8)
            return ein.repeat(bmp, "h w -> h w c", c=3)
        except AssertionError:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    @property
    def str_semantics(self) -> Dict:
        semantics = {
            "int": lambda n: f"(int {n})",
            "lit": lambda n: f"<{n}>",
            "nil": lambda: "nil",
            "not": lambda x: f"(not {x})",
            "lt": lambda x, y: f"(lt {x} {y})",
            "and": lambda x, y: f"(and {x} {y})",
            "or": lambda x, y: f"(or {x} {y})",
            "xmax": lambda: "xmax",
            "ymax": lambda: "ymax",
            "z": lambda i: f"(z {i})",
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
        return semantics


if __name__ == "__main__":
    b = Blocks(featurizer=ResnetFeaturizer(), gram=2, env={"z": list(range(10))})
    examples = [
        # "(rect (point 1 2) (point 1 2) 1)",
        # "(rect (point 1 1) (point xmax ymax) 1)",
        # "(rect (point 1 1) (point 15 15) 1)",
        # "(line (point 1 2) (point 3 4) 1)",
        # "(seq (line (point 1 2) (point 3 4) 1) "
        # "     (rect (point 1 2) (point 1 2) 1))",
        # "(apply hflip (line (point 1 2) (point 1 4) 1))",
        "(rect (point 1 1) (point (z 1) (z 2)) 1)"
    ]
    for x in examples:
        t = b.parse(x)
        print(t.to_sexp())
        print(b.to_str(t))
        print()
        # img = b.eval(t, )
        # plt.imshow(img)
        # plt.show()

    b.fit(corpus=[b.parse(s) for s in examples], alpha=0.01)
    for i in range(10):
        t = b.sample()
        print(t.to_sexp())
        print(b.to_str(t))
        print()
    #     img = b.eval(t)
    #     plt.imshow(img)
    #     plt.show()

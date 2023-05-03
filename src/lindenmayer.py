from __future__ import annotations
import random
from typing import *
from math import sin, cos, radians
import numpy as np
import skimage.draw
from scipy.ndimage import gaussian_filter
import itertools as it
from sys import stderr, maxsize

from eggy import simplify
from lang import Language, Tree, Grammar, ParseError
from featurizers import ResnetFeaturizer
import util


class LSystem:

    def __init__(self):
        pass

    def expand(self, s: str) -> str:  # pragma: no cover
        raise NotImplementedError(f"Should be implemented in child {type(self).__name__}")

    @property
    def axiom(self) -> str:
        raise NotImplementedError(f"Should be implemented in child {type(self).__name__}")

    def expansions(self, iters: int) -> Iterator[str]:
        """Returns a generator over the 0-th through `iters`-th expansions."""
        word = self.axiom
        yield word
        for _ in range(iters):
            word = self.expand(word)
            yield word

    def nth_expansion(self, n: int) -> str:
        """Returns the n-th expansion."""
        word = self.axiom
        for _ in range(n):
            word = self.expand(word)
        return word

    def expand_until(self, length: int) -> Tuple[int, str]:
        """
        Apply rules to the axiom until the number of `F` tokens is >= length
        """
        word = self.axiom
        depth = 0
        while len(word) < length:
            cache = word
            word = self.expand(word)
            if word == cache:
                break
            depth += 1
        return depth, word

    @staticmethod
    def draw(s: str, d: float, theta: float, n_rows: int = 512, n_cols: int = 512, aa=True) -> np.ndarray:  # pragma: no cover
        """
        Draw the turtle interpretation of the string `s` onto a `n_rows` x `n_cols` array,
        using scikit-image's drawing library (with anti-aliasing).
        """
        r, c = n_rows//2, n_cols//2  # parser_start at center of canvas
        heading = 90  # parser_start facing up (logo)
        stack = []
        canvas = np.zeros((n_rows, n_cols), dtype=np.uint8)
        for char in s:
            if char == 'F':
                r1 = r + int(d * sin(radians(heading)))
                c1 = c + int(d * cos(radians(heading)))
                # only draw if at least one coordinate is within the canvas
                if ((0 <= r1 < n_rows and 0 <= c1 < n_cols) or
                    (0 <= r < n_rows and 0 <= c < n_cols)):
                    if aa:
                        rs, cs, val = skimage.draw.line_aa(r, c, r1, c1)
                        mask = (0 <= rs) & (rs < n_rows) & (0 <= cs) & (cs < n_cols)  # mask out out-of-bounds indices
                        rs, cs, val = rs[mask], cs[mask], val[mask]
                        canvas[rs, cs] = val * 255
                    else:
                        rs, cs = skimage.draw.line(r, c, r1, c1)
                        mask = (0 <= rs) & (rs < n_rows) & (0 <= cs) & (cs < n_cols)  # mask out out-of-bounds indices
                        rs, cs = rs[mask], cs[mask]
                        canvas[rs, cs] = 255
                r, c = r1, c1
            elif char == 'f':
                r += int(d * sin(radians(heading)))
                c += int(d * cos(radians(heading)))
            elif char == '+':
                heading += theta
            elif char == '-':
                heading -= theta
            elif char == '[':
                stack.append((r, c, heading))
            elif char == ']':
                r, c, heading = stack.pop()
        return util.stack_repeat(canvas, 3)


class D0LSystem(LSystem):
    """
    A deterministic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self, axiom: str, productions: Dict[str, str]):
        super().__init__()
        self._axiom = axiom
        self.productions = productions

    @property
    def axiom(self) -> str:
        return self._axiom

    def __str__(self) -> str:  # pragma: no cover
        rules = []
        for pred, succs in self.productions.items():
            for i, succ in enumerate(succs):
                rules.append(
                    f'{pred} -> {succ}'
                )
        return f'axiom: {self.axiom}\n' + 'rules: [\n  ' + '\n  '.join(rules) + '\n]\n'

    def expand(self, s: str) -> str:
        # Assume identity production if predecessor is not in self.productions
        return ''.join(self.productions.get(c, c) for c in s)


class S0LSystem(LSystem):
    """
    A stochastic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self,
                 axiom: str,
                 productions: Dict[str, List[str]],
                 distribution: str | Dict[str, List[float]] = "uniform"):
        super().__init__()
        self._axiom = axiom
        self.productions = productions

        # check if distribution is a string
        if distribution == "uniform":
            self.distribution = {
                pred: np.ones(len(succs)) / len(succs)
                for pred, succs in productions.items()
            }
        else:
            self.distribution = {
                pred: (lambda x: x / np.sum(x))(np.array(weights))
                for pred, weights in distribution.items()
            }

    @property
    def axiom(self) -> str:
        return self._axiom

    def expand(self, s: str) -> str:
        return ''.join(random.choices(population=self.productions.get(c, [c]),
                                      weights=self.distribution.get(c, [1]),
                                      k=1)[0]
                       for c in s)

    def __str__(self) -> str:  # pragma: no cover
        rules = []
        for pred, succs in self.productions.items():
            for i, succ in enumerate(succs):
                weight = self.distribution[pred][i]
                rules.append(
                    f'{pred} -[{weight:.3f}]-> {succ}'
                )
        return f'axiom: {self.axiom}\n' + \
               'rules: [\n  ' + '\n  '.join(rules) + '\n]\n'

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other) -> bool:
        # doesn't handle different orderings
        return (isinstance(other, S0LSystem) and
                self.axiom == other.axiom and
                self.productions == other.productions and
                all(np.array_equal(self.distribution[pred], other.distribution[pred])
                    for pred in self.productions.keys()))

    def to_sentence(self) -> List[str]:
        """
        Convert the L-system to a sentence outputted by a metagrammar.
        Needed to fit metagrammars to libraries of L-systems.
        """
        return (list(self.axiom) + [';'] + list(it.chain.from_iterable(
            [[pred, '~', *succ, ',']
             for pred, succs in self.productions.items()
             for succ in succs])))[:-1]

    def to_str(self) -> str:
        return "".join(self.to_sentence())

    @staticmethod
    def from_str(s: str) -> 'S0LSystem':
        return S0LSystem.from_sentence(list(s))

    @staticmethod
    def from_sentence(s: List[str] | Tuple[str]) -> 'S0LSystem':
        """
        Accepts a single string with spaces between distinct tokens, and outputs an L-system.
        The list should have the form 'AXIOM; RULE, RULE, ...', where RULE has the form 'LHS ~ RHS'.
        """
        assert isinstance(s, List) or isinstance(s, Tuple), f"Expected list/tuple of strings but found {type(s)}"
        s = " ".join(s)
        s_axiom, s_rules = s.strip().split(';')
        axiom = s_axiom.replace(' ', '')
        s_rules = s_rules.strip()

        rules = {}
        for s_rule in s_rules.split(','):
            if not s_rule.strip():
                continue

            lhs, rhs = s_rule.split('~')
            lhs = lhs.strip()
            rhs = rhs.replace(',', '').strip()
            rhs = ''.join(rhs.split())

            if lhs in rules:
                rules[lhs].append(rhs)
            else:
                rules[lhs] = [rhs]

        return S0LSystem(axiom, rules, "uniform")


class LSys(Language):
    """
    Defines the L-System domain used for novelty search.
    """
    sol_metagrammar = r"""
        lsystem: axiom ";" rules   -> lsystem
        axiom: symbols             -> axiom
        symbols: symbol symbols    -> symbols
               | symbol            -> symbol
        symbol: "[" symbols "]"    -> bracket
              | NT                 -> nonterm
              | T                  -> term
        rules: rule "," rules      -> rules
             | rule                -> rule
        rule: NT "~" symbols       -> arrow
        NT: /[Ff]/
        T: "+"
         | "-"

        %import common.WS
        %ignore WS
    """
    sol_types = {
        "lsystem": ["Axiom", "Rules", "LSystem"],
        "axiom": ["Symbols", "Axiom"],
        "symbols": ["Symbol", "Symbols", "Symbols"],
        "symbol": ["Symbol", "Symbols"],
        "bracket": ["Symbols", "Symbol"],
        "nonterm": ["Nonterm", "Symbol"],
        "term": ["Term", "Symbol"],
        "rules": ["Rule", "Rules", "Rules"],
        "rule": ["Rule", "Rules"],
        "arrow": ["Nonterm", "Symbols", "Rule"],
        "F": ["Nonterm"],
        "f": ["Nonterm"],
        "+": ["Term"],
        "-": ["Term"],
    }
    # sol_types.update({
    #     token: ["Nonterm"] for token in "LRXYAB"
    # })

    dol_metagrammar = r"""
        lsystem: axiom ";" rule   -> lsystem
        axiom: symbols            -> axiom
        symbols: symbol symbols   -> symbols
               | symbol           -> symbol
        symbol: "[" symbols "]"   -> bracket
              | NT                -> nonterm
              | T                 -> term
        rule: NT "~" symbols      -> arrow
        NT: "F"
        T: "+" | "-"

        %import common.WS
        %ignore WS
    """
    dol_types = {
        "lsystem": ["Axiom", "Rule", "LSystem"],
        "axiom": ["Symbols", "Axiom"],
        "symbols": ["Symbol", "Symbols", "Symbols"],
        "symbol": ["Symbol", "Symbols"],
        "bracket": ["Symbols", "Symbol"],
        "nonterm": ["Nonterm", "Symbol"],
        "term": ["Term", "Symbol"],
        "arrow": ["Nonterm", "Symbols", "Rule"],
        "F": ["Nonterm"],
        "+": ["Term"],
        "-": ["Term"],
    }

    def __init__(self, theta: float, step_length: int, render_depth: int, n_rows: int, n_cols: int, aa=True,
                 kind="stochastic", quantize=False, disable_last_layer=False, softmax_outputs=True):
        self.kind = kind
        assert kind in {"stochastic", "deterministic"}, f"LSys must be 'stochastic' or 'deterministic', but got {kind}"
        if kind == "stochastic":
            parser_grammar = LSys.sol_metagrammar
            parser_types = LSys.sol_types
        else:
            parser_grammar = LSys.dol_metagrammar
            parser_types = LSys.dol_types
        super().__init__(parser_grammar=parser_grammar,
                         parser_start="lsystem",
                         root_type="LSystem",
                         model=Grammar.from_components(parser_types, gram=2),
                         featurizer=ResnetFeaturizer(quantize=quantize,
                                                     disable_last_layer=disable_last_layer,
                                                     softmax_outputs=softmax_outputs))
        self.theta = theta
        self.step_length = step_length
        self.render_depth = render_depth
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.aa = aa

    def none(self) -> Any:
        return np.zeros((self.n_rows, self.n_cols))

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> np.ndarray:
        s = self.to_str(t)
        if env is None: env = {}

        # fetch env variables if present, otherwise use object fields
        theta = env["theta"] if "theta" in env else self.theta
        render_depth = env["render_depth"] if "render_depth" in env else self.render_depth
        step_length = env["step_length"] if "step_length" in env else self.step_length
        n_rows = env["n_rows"] if "n_rows" in env else self.n_rows
        n_cols = env["n_cols"] if "n_cols" in env else self.n_cols
        aa = env["aa"] if "aa" in env else self.aa

        lsys = S0LSystem.from_str(s)
        sample = lsys.nth_expansion(render_depth)
        return LSystem.draw(sample, d=step_length, theta=theta, n_rows=n_rows, n_cols=n_cols, aa=aa)

    @property
    def str_semantics(self) -> Dict:
        semantics = {
            "lsystem": lambda ax, rs: f"{ax};{rs}",
            "axiom": lambda xs: xs,
            "symbols": lambda x, xs: f"{x}{xs}",
            "symbol": lambda x: x,
            "bracket": lambda xs: f"[{xs}]",
            "nonterm": lambda nt: nt,
            "term": lambda t: t,
            "rules": lambda r, rs: f"{r},{rs}",
            "rule": lambda r: r,
            "arrow": lambda nt, xs: f"{nt}~{xs}",
            "F": lambda: "F",
            "f": lambda: "f",
            "+": lambda: "+",
            "-": lambda: "-",
        }
        semantics.update({
            token: (lambda: token)
            for token in "LRXYAB"
        })
        return semantics

    def simplify(self, t: Tree) -> Tree:
        """Simplify using egg and deduplicate rules"""
        sexp = t.to_sexp()
        sexp_simpl = simplify(sexp)
        if "nil" in sexp_simpl:
            if sexp_simpl != "nil":
                print(f"WARNING: found nil in unsimplified expression: {sexp_simpl}", file=stderr)
            raise NilError(f"Unexpected 'nil' token in simplified expr: {sexp_simpl}")
        s_simpl = self.to_str(Tree.from_sexp(sexp_simpl))
        s_dedup = LSys.dedup_rules(s_simpl)
        return self.parse(s_dedup)

    @staticmethod
    def dedup_rules(s: str) -> str:
        s_axiom, s_rules = s.split(";")
        rules = set(s_rules.split(","))
        s_rules = ",".join(sorted(rules, key=lambda x: (len(x), x)))
        return f"{s_axiom};{s_rules}"

    def __str__(self) -> str:
        excluded_keys = {"model", "parser"}
        return "\n".join([f"<StochLSys:"] +
                         [f"  {key}={val}"
                          for key, val in self.__dict__.items()
                          if key not in excluded_keys])


class NilError(ParseError):
    pass


def test_lsys_simplify():
    cases = {
        "F;F~F": "F;F~F",
        "F;F~+-+--+++--F": "F;F~F",
        "F;F~-+F+-": "F;F~F",
        "F;F~[F]F": "F;F~F",
        "F;F~[FF]FF": "F;F~FF",
        "F;F~[+F-F]+F-F": "F;F~+F-F",
        "F;F~[F]": "F;F~[F]",
        "F;F~[FF+FF]": "F;F~[FF+FF]",
        "F;F~F,F~F,F~F": "F;F~F",
        "F;F~F,F~+-F,F~F": "F;F~F",
        "F;F~F,F~+F-": "F;F~F,F~+F-",
        "F;F~F,F~+F-,F~F": "F;F~F,F~+F-",
        "F;F~F,F~FF,F~F,F~FF": "F;F~F,F~FF",
        "F;F~F[+F]F,F~F,F~F[+F]F": "F;F~F,F~F[+F]F",
        "F;F~[-+-+---]F[++++]": "F;F~F",
        "+;F~F": "nil",
        "[++];F~F": "nil",
        "[++];F~[F]": "nil",
        "[++];F~[F][+++]": "nil",
        "F;F~+": "nil",
        "F;F~F,F~+": "F;F~F",
        "F;F~+,F~+": "nil",
        "F;F~F,F~+,F~+": "F;F~F",
    }
    L = LSys(theta=90, step_length=3, render_depth=3, n_rows=128, n_cols=128)
    for x, y in cases.items():
        t_x = L.parse(x)
        try:
            out = L.to_str(L.simplify(t_x))
            assert out == y, f"Expected {x} => {y} but got {out}"
        except NilError:
            assert y == "nil", f"Got NilError on unexpected input {x}"


if __name__ == "__main__":
    import view
    from torch import from_numpy, Tensor, stack
    from featurizers import ResnetFeaturizer
    np.set_printoptions(threshold=maxsize)

    examples = [
        # "F-F;F~-" + "F" * n for n in range(1, 10)
        # "F;F~FF",
        # "F;F~F[+F][-F]F",
        "FFF+FFF[[+FF]];F~F+F++FF+F",
        "-FF[+F+FFFF]+++F-++F-FF[F]F+++F--FFFF[F]-[-[+-+FF+[F]]F+-FFF+F[[+F]]]F;F~F+-F+FFF-FFFFFF+F[F]",
        "F-F;F~-FFFFFFFF",
        # "F;F~+--+F",
        # "F;F~+--+F,F~F",
        # "F;F~[+F][-F]F,F~FF",
    ]
    params = {
        "theta": 45,
        "step_length": 3,
        "render_depth": 3,
        "n_rows": 128,
        "n_cols": 128,
    }
    L = LSys(**params, kind="deterministic")
    print(L)
    # view.plot_lsys_at_depths(L, examples, "", 3, depths=(1, 6))
    M = [L.eval(L.parse(x), {"aa": True}) for x in examples]
    # print("aa:", np.unique(M))
    # util.plot(M, title="aa")

    # M_no_aa = [L.eval(L.parse(x), {"aa": False}) for x in examples]
    # print("no aa:", np.unique(M_no_aa))
    # util.plot(M_no_aa, title="no aa")

    # test effect of gaussian blur
    ft = ResnetFeaturizer()
    preprocessed = ft.preprocess(stack([from_numpy(x) for x in M]))
    util.plot(preprocessed, title="resnet preprocessed")
    for i in range(6):
        filtered = [gaussian_filter(x, sigma=i) for x in M]
        util.plot(filtered, title=f"gaussian filter, sigma={i}")

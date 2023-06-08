from __future__ import annotations
import random
from pprint import pp
from typing import Dict, List, Iterator, Tuple, Any
from math import sin, cos, radians
import numpy as np
import skimage.draw
import itertools as it
from sys import stderr, maxsize
import colorsys

import eggy
from lang.lang import Language, Tree, Grammar, ParseError
from featurizers import Featurizer
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
    def draw(s: str, d: float, theta: float,
             n_rows: int = 512, n_cols: int = 512, aa=True,
             vary_color=True, hue_start=240, hue_end=300, hue_step=0.5) -> np.ndarray:  # pragma: no cover
        """
        Draw the turtle interpretation of the string `s` onto a `n_rows` x `n_cols` array,
        using scikit-image's drawing library (with anti-aliasing).

        Color strokes by recency, varying hue within the range (hue_start, hue_end),
        with hues in 0-360 degrees.  Hue is incremented by hue_step for each stroke.
        """
        r, c = n_rows//2, n_cols//2  # parser_start at center of canvas
        heading = 90  # parser_start facing up (logo)
        stack = []
        canvas = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
        hue_angle = 0
        for char in s:
            if char == 'F':
                # choose hue based on recency
                if vary_color:
                    hue = hue_start + (hue_angle % (hue_end - hue_start))
                    rgb = 255 * np.array(colorsys.hsv_to_rgb(hue / 360, 1, 1))
                    hue_angle += hue_step
                else:
                    rgb = np.array([255, 255, 255])

                r1 = r + int(d * sin(radians(heading)))
                c1 = c + int(d * cos(radians(heading)))
                # only draw if at least one coordinate is within the canvas
                if ((0 <= r1 < n_rows and 0 <= c1 < n_cols) or
                    (0 <= r < n_rows and 0 <= c < n_cols)):
                    if aa:
                        rs, cs, intensities = skimage.draw.line_aa(r, c, r1, c1)
                        mask = (0 <= rs) & (rs < n_rows) & (0 <= cs) & (cs < n_cols)  # mask out out-of-bounds indices
                        rs, cs, intensities = rs[mask], cs[mask], intensities[mask]
                        canvas[rs, cs] = np.outer(intensities, rgb)
                    else:
                        rs, cs = skimage.draw.line(r, c, r1, c1)
                        mask = (0 <= rs) & (rs < n_rows) & (0 <= cs) & (cs < n_cols)  # mask out out-of-bounds indices
                        rs, cs = rs[mask], cs[mask]
                        canvas[rs, cs] = rgb
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
        return canvas


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
        if s.count(';') > 1:
            return S0LSystem.from_sentence(list(s.split(';',maxsplit=1)[1]))
        else:
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
    ANGLES = [20, 45, 60, 90]
    EXTRA_NONTERMINALS = [] # ["f", "L", "R"]

    sol_metagrammar = r"""
        lsystem: angle ";" axiom ";" rules -> lsystem
        axiom: symbols                     -> axiom
        symbols: symbol symbols            -> symbols
               | symbol                    -> symbol
        symbol: "[" symbols "]"            -> bracket
              | NT                         -> nonterm
              | T                          -> term
        rules: rule "," rules              -> rules
             | rule                        -> rule
        rule: NT "~" symbols               -> arrow
        angle: NUMBER                      -> angle
        NT: "F" | "f" | "L" | "R"
        T: "+" | "-"

        %import common.WS
        %import common.NUMBER
        %ignore WS
    """
    sol_types = {
        "lsystem": ["Angle", "Axiom", "Rules", "LSystem"],
        "axiom": ["Symbols", "Axiom"],
        "symbols": ["Symbol", "Symbols", "Symbols"],
        "symbol": ["Symbol", "Symbols"],
        "bracket": ["Symbols", "Symbol"],
        "nonterm": ["Nonterm", "Symbol"],
        "term": ["Term", "Symbol"],
        "rules": ["Rule", "Rules", "Rules"],
        "rule": ["Rule", "Rules"],
        "arrow": ["Nonterm", "Symbols", "Rule"],
        "angle": ["Num", "Angle"],
        "F": ["Nonterm"],
        "+": ["Term"],
        "-": ["Term"],
    }
    sol_types.update({nt: ["Nonterm"] for nt in EXTRA_NONTERMINALS})
    sol_types.update({angle: ["Num"] for angle in ANGLES})

    dol_metagrammar = r"""
        lsystem: angle ";" axiom ";" rule   -> lsystem
        axiom: symbols                      -> axiom
        symbols: symbol symbols             -> symbols
               | symbol                     -> symbol
        symbol: "[" symbols "]"             -> bracket
              | NT                          -> nonterm
              | T                           -> term
        rule: NT "~" symbols                -> arrow
        angle: NUMBER                       -> angle
        NT: "F" | "f" | "L" | "R"
        T: "+" | "-"

        %import common.WS
        %import common.NUMBER
        %ignore WS
    """
    dol_types = {
        "lsystem": ["Angle", "Axiom", "Rule", "LSystem"],
        "axiom": ["Symbols", "Axiom"],
        "symbols": ["Symbol", "Symbols", "Symbols"],
        "symbol": ["Symbol", "Symbols"],
        "bracket": ["Symbols", "Symbol"],
        "nonterm": ["Nonterm", "Symbol"],
        "term": ["Term", "Symbol"],
        "arrow": ["Nonterm", "Symbols", "Rule"],
        "angle": ["Num", "Angle"],
        "F": ["Nonterm"],
        "+": ["Term"],
        "-": ["Term"],
    }
    dol_types.update({nt: ["Nonterm"] for nt in EXTRA_NONTERMINALS})
    dol_types.update({angle: ["Num"] for angle in ANGLES})

    def __init__(self, kind: str, featurizer: Featurizer, step_length: int, render_depth: int,
                 n_rows=128, n_cols=128, aa=True, vary_color=True):
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
                         featurizer=featurizer)
        self.step_length = step_length
        self.render_depth = render_depth
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.aa = aa
        self.vary_color = vary_color

    def none(self) -> Any:
        return np.zeros((self.n_rows, self.n_cols))

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> np.ndarray:
        s = self.to_str(t)
        if env is None: env = {}

        # fetch env variables if present, otherwise use object fields
        theta = float(t.children[0].children[0].value)  # extract angle (hacky)
        render_depth = env["render_depth"] if "render_depth" in env else self.render_depth
        step_length = env["step_length"] if "step_length" in env else self.step_length
        n_rows = env["n_rows"] if "n_rows" in env else self.n_rows
        n_cols = env["n_cols"] if "n_cols" in env else self.n_cols
        aa = env["aa"] if "aa" in env else self.aa
        vary_color = env["vary_color"] if "vary_color" in env else False

        lsys = S0LSystem.from_str(s)
        sample = lsys.nth_expansion(render_depth)
        return LSystem.draw(
            sample,
            d=step_length,
            theta=theta,
            n_rows=n_rows,
            n_cols=n_cols,
            aa=aa,
            vary_color=vary_color,
        )

    @property
    def str_semantics(self) -> Dict:
        semantics = {
            "lsystem": lambda angle, ax, rs: f"{angle};{ax};{rs}",
            "axiom": lambda xs: xs,
            "symbols": lambda x, xs: f"{x}{xs}",
            "symbol": lambda x: x,
            "bracket": lambda xs: f"[{xs}]",
            "nonterm": lambda nt: nt,
            "term": lambda t: t,
            "rules": lambda r, rs: f"{r},{rs}",
            "rule": lambda r: r,
            "arrow": lambda nt, xs: f"{nt}~{xs}",
            "angle": lambda a: a,
        }
        # constants
        for token in ["F", "+", "-"] + LSys.EXTRA_NONTERMINALS + LSys.ANGLES:
            semantics[token] = str(token)
        return semantics

    def simplify(self, t: Tree) -> Tree:
        """Simplify using egg and deduplicate rules"""
        sexp = t.to_sexp()
        sexp_simpl = eggy.simplify_lsystem(sexp)
        if "nil" in sexp_simpl:
            if sexp_simpl != "nil":
                print(f"WARNING: found nil in expression: {sexp} => {sexp_simpl}", file=stderr)
            raise NilError(f"Unexpected 'nil' in simplified expr: {sexp} => {sexp_simpl}")
        s_simpl = self.to_str(Tree.from_sexp(sexp_simpl))
        if self.kind == "stochastic":
            s_dedup = LSys.dedup_rules(s_simpl)
            return self.parse(s_dedup)
        return self.parse(s_simpl)

    @staticmethod
    def dedup_rules(s: str) -> str:
        s_angle, s_axiom, s_rules = s.split(";")
        rules = set(s_rules.split(","))
        s_rules = ",".join(sorted(rules, key=lambda x: (len(x), x)))
        return f"{s_angle};{s_axiom};{s_rules}"

    def __str__(self) -> str:
        excluded_keys = {"model", "parser"}
        return "\n".join([f"<LSys:"] +
                         [f"  {key}={val}"
                          for key, val in self.__dict__.items()
                          if key not in excluded_keys]) + ">"


class NilError(ParseError):
    pass


if __name__ == "__main__":
    from torch import from_numpy, stack
    from featurizers import ResnetFeaturizer
    np.set_printoptions(threshold=maxsize)
    L = LSys(step_length=3, render_depth=4, n_rows=128, n_cols=128, kind="deterministic")
    print(L)

    templates = [
        "F;F~F[+F][-F]F",
        "F;F~FF+-F[+][-][[+--]]",
    ]
    examples = []
    for t in templates:
        for angle in LSys.ANGLES:
            examples.append(f"{angle};{t}")

    programs = [L.parse(x) for x in examples]
    simplified_programs = [L.simplify(p) for p in programs]
    for x, p in zip(examples, simplified_programs):
        print(f"{x} => {L.to_str(p)}")

    def sample_and_show(lsys: LSys, side_len: int):
        samples = [lsys.sample() for _ in range(side_len ** 2)]
        for x in samples:
            s = lsys.to_str(x)
            print(f"{s}: {x}")
        util.plot([lsys.eval(x) for x in samples], shape=(side_len, side_len))

    L.model.normalize_()
    sample_and_show(L, 20)  # uniform random
    uniform_samples = [L.sample() for _ in range(20 ** 2)]

    L.fit(programs, alpha=1)
    sample_and_show(L, 20)  # fitted to unsimplified samples

    L.model.normalize_()
    L.fit(simplified_programs, alpha=1)
    sample_and_show(L, 20)  # fitted to simplified

    L.model.normalize_()
    L.fit(uniform_samples, alpha=1)
    sample_and_show(L, 20)  # fitted to uniformly generated data, unsimplified

    L.model.normalize_()
    uniform_simplified = []
    for p in uniform_samples:
        try:
            x = L.simplify(p)
            uniform_simplified.append(x)
        except NilError:
            pass
    L.fit(uniform_simplified, alpha=1)
    sample_and_show(L, 20)  # fitted to uniformly generated data, simplified

    # view.plot_lsys_at_depths(L, examples, "", n_imgs_per_plot=len(LSys.ANGLES), depths=(1, 6))
    #
    # M = [L.eval(p, {"aa": True}) for p in programs]
    # shape = (len(templates), len(LSys.ANGLES))
    # util.plot(M, shape=shape, labels=[L.to_str(p) for p in programs], title="aa")
    #
    # M_no_aa = [L.eval(p, {"aa": False}) for p in programs]
    # print("no aa:", np.unique(M_no_aa))
    # util.plot(M_no_aa, title="no aa", shape=shape)
    #
    # # test effect of gaussian blur
    # ft = ResnetFeaturizer()
    # preprocessed = ft.preprocess(stack([from_numpy(x) for x in M]))
    # util.plot(preprocessed, title="resnet preprocessed", shape=shape)
    # for i in range(3):
    #     filtered = [gaussian_filter(x, sigma=i) for x in M]
    #     util.plot(filtered, title=f"gaussian filter, sigma={i}", shape=shape)

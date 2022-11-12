import random
from typing import Dict, List, Iterator, Union, Tuple
from math import sin, cos, radians
import numpy as np
import skimage.draw as skdraw
import matplotlib.pyplot as plt
import pdb
import itertools as it
import time

import util
from cfg import PCFG

# TODO: define LSYSTEM_MG as a CFG, then turn it into a PCFG when
#  it's being used so we don't have to worry about immutability
# TODO: add test cases for the different kinds of expressions we want to cover with this metagrammar
LSYSTEM_MG = PCFG(
    start="L-SYSTEM",
    weights="uniform",
    rules={
        "L-SYSTEM": [
            ["AXIOM", ";", "RULES"],
        ],
        "AXIOM": [
            ["NT", "AXIOM_W_NT"],
            ["NT"],
            ["T", "AXIOM"],
        ],
        "AXIOM_W_NT": [
            ["NT_OR_T", "AXIOM_W_NT"],
            ["NT_OR_T"],
        ],
        "RULES": [
            ["RULE", ",", "RULES"],
            ["RULE"],
        ],
        "RULE": [
            ["LHS", "~", "RHS"],
        ],
        "LHS": [
            ["NT"],
        ],
        "RHS": [
            ["[", "RHS", "]", "RHS"],
            ["[", "RHS", "]"],
            ["NT", "RHS_W_NT"],
            ["NT"],
            ["T", "RHS"],
        ],
        "RHS_W_NT": [
            ["[", "RHS", "]", "RHS"],
            ["[", "RHS", "]"],
            ["NT_OR_T", "RHS_W_NT"],
            ["NT_OR_T"],
        ],
        "NT_OR_T": [
            ["NT"],
            ["T"]
        ],
        "NT": [
            ["F"],
            # ["f"],
            # ["X"],
        ],
        "T": [
            ["+"],
            ["-"],
        ],
    },
)


class LSystem:

    def __init__(self):
        pass

    def expand(self, s: str) -> str:
        assert False, f"Should be implemented in child {type(self).__name__}"

    def expansions(self, iters: int) -> Iterator[str]:
        """Returns a generator over `iters` expansions."""
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
    def draw(s: str, d: float, theta: float, n_rows: int = 512, n_cols: int = 512) -> np.ndarray:
        """
        Draw the turtle interpretation of the string `s` onto a `n_rows` x `n_cols` array,
        using scikit-image's drawing library (with anti-aliasing).
        """
        r, c = n_rows//2, n_cols//2  # start at center of canvas
        heading = 90  # start facing up (logo)
        stack = []
        canvas = np.zeros((n_rows, n_cols))
        for char in s:
            if char == 'F':
                r1 = r + int(d * sin(radians(heading)))
                c1 = c + int(d * cos(radians(heading)))
                rs, cs, val = skdraw.line_aa(r, c, r1, c1)
                # mask out out-of-bounds indices
                mask = (0 <= rs) & (rs < n_rows) & (0 <= cs) & (cs < n_cols)
                rs, cs, val = rs[mask], cs[mask], val[mask]
                canvas[rs, cs] = val * 255
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
        self.axiom = axiom
        self.productions = productions

    def __str__(self) -> str:
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
                 distribution: Union[str, Dict[str, List[float]]] = "uniform"):
        super().__init__()
        self.axiom = axiom
        self.productions = productions

        # check if distribution is a string
        if distribution == "uniform":
            distribution = {
                pred: [1 / len(succs)] * len(succs)
                for pred, succs in productions.items()
            }
        else:
            distribution = {
                pred: util.normalize(weights)
                for pred, weights in distribution.items()
            }

        # check that distribution sums to 1 for any predecessor
        assert all(abs(sum(weights) - 1) < 0.01
                   for _, weights in distribution.items()), \
            "All rules with the same predecessor should have" \
            " probabilities summing to 1"
        self.distribution = distribution

    def __hash__(self):
        return hash(" ".join(self.to_sentence()))

    def expand(self, s: str) -> str:
        return ''.join(random.choices(population=self.productions.get(c, [c]),
                                      weights=self.distribution.get(c, [1]),
                                      k=1)[0]
                       for c in s)

    def __str__(self) -> str:
        rules = []
        for pred, succs in self.productions.items():
            for i, succ in enumerate(succs):
                weight = self.distribution[pred][i]
                rules.append(
                    f'{pred} -[{weight:.3f}]-> {succ}'
                )
        return f'axiom: {self.axiom}\n' + \
            'rules: [\n  ' + '\n  '.join(rules) + '\n]\n'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        return (self.axiom, self.productions, self.distribution) == \
            (other.axiom, other.productions, other.distribution)

    def to_sentence(self) -> List[str]:
        """
        Convert the L-system to a sentence outputted by a metagrammar.
        Needed to fit metagrammars to libraries of L-systems.
        """
        return (list(self.axiom) + [';'] + list(it.chain.from_iterable(
            [[pred, '~', *succ, ',']
             for pred, succs in self.productions.items()
             for succ in succs])))[:-1]

    def to_code(self) -> str:
        return "".join(self.to_sentence())

    @staticmethod
    def from_sentence(s: List[str]) -> 'S0LSystem':
        """
        Accepts a list of strings, or a single string with spaces between
        distinct tokens, and outputs an L-system. The list should have the
        form 'AXIOM; RULE, RULE, ...', where RULE has the form 'LHS ~ RHS'.
        """
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


def test_from_sentence():
    cases = [
        ('F ; F ~ f F'.split(),
         S0LSystem('F', {'F': ['fF']}, 'uniform')),
        ('F ; F ~ f F ,'.split(),
         S0LSystem('F', {'F': ['fF']}, 'uniform')),
        ('F ; F ~ A , A ~ A b , A ~ b b , A ~ A A'.split(),
         S0LSystem('F', {'F': ['A'], 'A': ['Ab', 'bb', 'AA']}, 'uniform')),
        ('F + F ; F ~ f F ,'.split(),
         S0LSystem('F+F', {'F': ['fF']}, 'uniform')),
    ]
    for s, y in cases:
        out = S0LSystem.from_sentence(s)
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_from_sentence")


def test_to_sentence():
    cases = [
        (S0LSystem('F', {'F': ['fF']}, 'uniform'),
         'F ; F ~ f F'.split()),
        (S0LSystem('F', {'F': ['A'], 'A': ['Ab', 'bb', 'AA']}, 'uniform'),
         'F ; F ~ A , A ~ A b , A ~ b b , A ~ A A'.split()),
        (S0LSystem('F-F-F', {'F': ['fF']}, 'uniform'),
         'F - F - F ; F ~ f F'.split()),
    ]
    for g, y in cases:
        out = g.to_sentence()
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_to_sentence")


def draw_systems():
    systems: Dict[str, LSystem] = {
        'koch': D0LSystem(
            axiom='F-F-F-F',
            productions={
                'F': 'F-F+F+FF-F-F+F'
            },
        ),
        'islands': D0LSystem(
            axiom='F+F+F+F',
            productions={
                'F': 'F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF',
                'f': 'ffffff',
            },
        ),
        'branch': D0LSystem(
            axiom='F',
            productions={
                'F': 'F[+F]F[-F]F'
            },
        ),
        'wavy-branch': D0LSystem(
            axiom='F',
            productions={
                'F': 'FF-[-F+F+F]+[+F-F-F]'
            },
        ),
        'stochastic-branch': S0LSystem(
            axiom='F',
            productions={
                'F': ['F[+F]F[-F]F',
                      'F[+F]F',
                      'F[-F]F']
            },
            distribution={
                'F': [0.33,
                      0.33,
                      0.34]
            },
        ),
        'random-walk': S0LSystem(
            axiom='A',
            productions={
                'A': ['dFA'],
                'd': ['+', '++', '+++', '++++'],
                'F': ['F'],
            },
            distribution={
                'A': [1],
                'd': [0.25, 0.25, 0.25, 0.25],
                'F': [1],
            },
        ),
        'triplet': S0LSystem(
            axiom='F',
            productions={
                'F': ['FF',
                      '[-F]F',
                      '[+F]F'],
            },
            distribution='uniform'
        ),
    }

    for name, angle, levels, samples in [
        ('koch', 90, 4, 1),
        ('islands', 90, 3, 1),
        ('branch', 25.7, 5, 1),
        ('branch', 73, 5, 1),
        ('wavy-branch', 22.5, 5, 1),
        ('stochastic-branch', 22.5, 5, 5),
        # ('random-walk', 90, 99, 1),
        # ('triplet', 35, 5, 6),
        # ('random', 45, 6, 3),
    ]:
        for sample in range(samples):
            system = systems[name]
            print(system)
            for _, word in enumerate(system.expansions(levels)):
                mat = LSystem.draw(s=word, d=5, theta=angle)
                plt.imshow(mat)
                plt.show()
            print()


def test_render():
    systems = [
        S0LSystem(axiom="F",
                  productions={"F": ["F+F", "F-F"]}),
    ]
    for system in systems:
        _, s = system.expand_until(100)
        time_str = int(time.time())
        mat = S0LSystem.draw(s, d=3, theta=43, n_rows=512, n_cols=512)
        plt.imshow(mat)
        plt.show()


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: lindenmayer.py DIR")
    #     sys.exit(1)

    # test_to_sticks()
    # draw_systems(out_dir=sys.argv[1])
    test_from_sentence()
    test_to_sentence()
    draw_systems()
    test_render()

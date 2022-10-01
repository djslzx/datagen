import turtle
import svgwrite
import sys
import random
from typing import Dict, List, Iterator, Union, Tuple
from math import sin, cos, radians, sqrt
import pdb
import itertools as it
import util


def setup_turtle():
    if turtle.isvisible():
        turtle.screensize(1000, 1000)
        turtle.pensize(1)
        turtle.mode('logo')
        turtle.hideturtle()
        turtle.speed(0)
        turtle.tracer(0, 0)
        turtle.clear()
        turtle.setpos(0, 0)
        turtle.setheading(0)


class Stick:

    def __init__(self, x: float, y: float, d: float,
                 cos_theta: float, sin_theta: float):
        self.x = x
        self.y = y
        self.d = d
        self.cos_theta = cos_theta
        self.sin_theta = sin_theta

    def __eq__(self, other) -> bool:
        return isinstance(other, Stick) and \
            all(util.approx_eq(self.__dict__[key], other.__dict__[key])
                for key in self.__dict__.keys())

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Stick(x={self.x}, y={self.y}, d={self.d}, " \
            f"cos_theta={self.cos_theta:.3f}, sin_theta={self.sin_theta:.3f})"

    def endpoints(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (self.x, self.y),
            (self.x + self.d * self.cos_theta,
             self.y + self.d * self.sin_theta),
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
        Apply rules to the word until the number of `F` tokens is >= length
        """
        word = self.axiom
        depth = 0
        while word.count('F') < length:
            cache = word
            word = self.expand(word)
            if word == cache:
                break
            depth += 1
        return depth, word

    def to_sticks(s: str, d: float, theta: float) -> List[Stick]:
        """
        Converts the string `s` to a collection of sticks.
        - `d`: the length of each stick
        - `theta`: the angle of a turn, in degrees
        """
        # TODO: should the sticks represent angles directly or as multiples of
        # the input parameters (d, theta)?
        x, y = 0, 0
        heading = 90            # use logo mode: start facing up
        sticks = []
        stack = []
        for c in s:
            if c == 'F':
                sticks.append(Stick(
                    x=x,
                    y=y,
                    d=d,
                    cos_theta=cos(radians(heading)),
                    sin_theta=sin(radians(heading)),
                ))
                x += d * cos(radians(heading))
                y += d * sin(radians(heading))
            elif c == 'f':
                x += d * cos(radians(heading))
                y += d * sin(radians(heading))
            elif c == '+':
                heading += theta
            elif c == '-':
                heading -= theta
            elif c == '[':
                stack.append((x, y, heading))
            elif c == ']':
                x, y, heading = stack.pop()
        return sticks

    def to_svg(sticks: List[Stick], filename: str):
        if not sticks:
            return

        points = [stick.endpoints() for stick in sticks]

        # translate negative points in image over to positive values
        min_x = int(min(x
                        for (ax, ay), (bx, by) in points
                        for x in [ax, bx]))
        min_y = int(min(y
                        for (ax, ay), (bx, by) in points
                        for y in [ay, by]))
        max_y = int(max(y
                        for (ax, ay), (bx, by) in points
                        for y in [ay, by]))

        # flip and translate points
        points = [((ax - min_x + 1,
                    max_y - ay - min_y + 1),
                   (bx - min_x + 1,
                    max_y - by - min_y + 1))
                  for (ax, ay), (bx, by) in points]

        assert all(v >= 0
                   for (ax, ay), (bx, by) in points
                   for v in [ax, ay, bx, by])

        # create SVG drawing
        dwg = svgwrite.Drawing(filename=filename)
        dwg.add(dwg.rect(size=('100%', '100%'), fill='white', class_='bkg'))
        lines = dwg.add(dwg.g(id='sticks', stroke='black', stroke_width=1))
        for a, b in points:
            lines.add(dwg.line(start=a, end=b))
        dwg.save()

    def render(s: str, d: float, theta: float, filename: str):
        LSystem.to_svg(
            sticks=LSystem.to_sticks(s=s, d=d, theta=theta),
            filename=f'{filename}.svg',
        )

    def render_with_turtle(s: str, d: float, theta: float, filename: str):
        """Renders the L-System using Turtle graphics."""
        setup_turtle()
        stack = []
        for c in s:
            if c == 'F':
                # move forward and draw a line
                turtle.pendown()
                turtle.forward(d)
                turtle.penup()
            elif c == 'f':
                # move forward without drawing
                turtle.forward(d)
            elif c == '+':
                turtle.left(theta)
            elif c == '-':
                turtle.right(theta)
            elif c == '[':
                # push turtle state onto stack
                stack.append((turtle.pos(), turtle.heading()))
            elif c == ']':
                # pop turtle state off of stack
                pos, heading = stack.pop()
                turtle.setpos(*pos)
                turtle.setheading(heading)

        turtle.update()
        turtle.getcanvas().postscript(
            file=f'{filename}.ps',
            colormode='color',
            height=1500,
            width=1500,
        )
        turtle.clear()
        turtle.setpos(0, 0)
        turtle.setheading(0)


class D0LSystem(LSystem):
    """
    A deterministic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self, axiom: str, productions: Dict[str, str]):
        super().__init__()
        self.axiom = axiom
        self.productions = productions

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
                 distribution: Union[str, Dict[str, List[float]]]):
        super().__init__()
        self.axiom = axiom
        self.productions = productions

        # check if distribution is a string
        if isinstance(distribution, str) and distribution == "uniform":
            distribution = {
                pred: [1 / len(succs)] * len(succs)
                for pred, succs in productions.items()
            }
        else:
            distribution = util.normalize_weights(distribution)

        # check that distribution sums to 1 for any predecessor
        assert all(abs(sum(weights) - 1) < 0.01
                   for _, weights in distribution.items()), \
            "All rules with the same predecessor should have" \
            " probabilities summing to 1"
        self.distribution = distribution

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

    def __eq__(self, other) -> bool:
        return (self.axiom, self.productions, self.distribution) == \
            (other.axiom, other.productions, other.distribution)

    def to_sentence(self) -> List[str]:
        """
        Convert the L-system to a sentence outputted by a metagrammar.
        Needed to fit metagrammars to libraries of L-systems.
        """
        return ([self.axiom, ';'] + list(it.chain.from_iterable(
            [[pred, '~', *succ, ',']
             for pred, succs in self.productions.items()
             for succ in succs])))[:-1]

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
    ]
    for g, y in cases:
        out = g.to_sentence()
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_to_sentence")


def test_to_sticks():
    cases = [
        ("F", 1, 90,
         [Stick(x=0, y=0, d=1, cos_theta=0, sin_theta=1)]),
        ("FF", 1, 90,
         [
             Stick(x=0, y=0, d=1, cos_theta=0, sin_theta=1),
             Stick(x=0, y=1, d=1, cos_theta=0, sin_theta=1),
         ]),
        ("F+F", 1, 90,
         [
             Stick(x=0, y=0, d=1, cos_theta=0, sin_theta=1),
             Stick(x=0, y=1, d=1, cos_theta=-1, sin_theta=0),
         ]),
        ("F+F+F+F", 1, 90,
         [
             Stick(0, 0, 1, 0, 1),
             Stick(0, 1, 1, -1, 0),
             Stick(-1, 1, 1, 0, -1),
             Stick(-1, 0, 1, 1, 0),
         ]),
        ("F-F-F-F", 1, 90,
         [
             Stick(0, 0, 1, 0, 1),
             Stick(0, 1, 1, 1, 0),
             Stick(1, 1, 1, 0, -1),
             Stick(1, 0, 1, -1, 0),
         ]),
        ("-F", 1, 30,
         [
             Stick(0, 0, 1, 1/2, sqrt(3)/2),
         ]),
        ("+FF", 1, 30,
         [
             Stick(0, 0, 1, -1/2, sqrt(3)/2),
             Stick(-1/2, sqrt(3)/2, 1, -1/2, sqrt(3)/2),
         ]),
    ]
    for s, d, theta, ans in cases:
        out = LSystem.to_sticks(s, d, theta)
        assert ans == out, \
            f"Expected {ans} for input {s, d, theta},\n got {out}"
    print(" [+] passed test_to_sticks")


def draw_systems(out_dir: str):
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
            for level, word in enumerate(system.expansions(levels)):
                print(word)
                LSystem.render_with_turtle(
                    word,
                    d=5,
                    theta=angle,
                    filename=f'{out_dir}/{name}-{angle}'
                    f'[{sample}]-{level:02d}'
                )
            print()


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: lindenmayer.py DIR")
    #     sys.exit(1)

    # test_to_sticks()
    # draw_systems(out_dir=sys.argv[1])
    test_from_sentence()
    test_to_sentence()

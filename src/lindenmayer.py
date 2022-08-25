import turtle
import svgwrite
from random import choice, choices
from typing import Dict, List, Generator, Union, Tuple
from math import sin, cos, radians, sqrt


def setup_turtle():
    if turtle.isvisible():
        # turtle.screensize(5000, 5000)
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

    def _approx_eq(a: float, b: float, threshold=10 ** -4) -> bool:
        return abs(a - b) < threshold

    def __eq__(self, other) -> bool:
        return isinstance(other, Stick) and \
            all(Stick._approx_eq(self.__dict__[key],
                                 other.__dict__[key])
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

    def expansions(self, iters: int) -> Generator[str, None, None]:
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
        dwg.add(dwg.rect(size=('100%', '100%'), fill="white", class_='bkg'))
        lines = dwg.add(dwg.g(id='sticks', stroke='black', stroke_width=1))
        for a, b in points:
            lines.add(dwg.line(start=a, end=b))
        dwg.save()

    def render(s: str, d: float, theta: float, filename: str):
        LSystem.to_svg(
            sticks=LSystem.to_sticks(s=s, d=d, theta=theta),
            filename=filename,
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
                # turtle.forward(length)
                turtle.pencolor('gray')
                turtle.pendown()
                turtle.forward(d)
                turtle.penup()
                turtle.color('black')
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

        # check that distribution sums to 1 for any predecessor
        assert all(abs(sum(weights) - 1) < 0.01
                   for _, weights in distribution.items()), \
            "All rules with the same predecessor should have" \
            " probabilities summing to 1"
        self.distribution = distribution

    def expand(self, s: str) -> str:
        return ''.join(choices(population=self.productions.get(c, [c]),
                               weights=self.distribution.get(c, [1]))[0]
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


class CFG:
    """
    A context-free grammar.  Terminals and nonterminals are represented as
    strings. The rule A ::= B C | D | ... is represented as the mapping
    A -> [[B, C], [D]].

    Terminals and nonterminals are not input explicitly -- they are inferred
    from the given rules. If a word is a predecessor in the rule list, then
    it is a nonterminal.  Otherwise, it is a terminal.
    """

    def __init__(self, rules: Dict[str, List[List[str]]]):
        assert all(succ and all(succ) for pred, succ in rules.items()), \
            "All rule RHS should be nonempty; " \
            "each element should also be nonempty"
        self.rules = rules

    def __str__(self):
        rules = "\n  ".join(
            f"{pred} -> {succ}"
            for pred, succ in self.rules.items())
        return "Rules: {  " + rules + "\n}"

    def is_nt(self, letter: str) -> bool:
        return letter in self.rules

    def apply(self, word: List[str]) -> List[str]:
        """
        Nondeterministically apply one of the production rules to
        a letter in the word.
        """
        # Only choose nonterminals to expand
        nonterminals = [i for i, letter in enumerate(word)
                        if self.is_nt(letter)]
        if not nonterminals:
            return word
        index = choice(nonterminals)
        letter = word[index]
        repl = choice(self.rules.get(letter, [[letter]]))
        return word[:index] + repl + word[index + 1:]

    def fixpoint(self, word: List[str]) -> List[str]:
        """Keep applying rules to the word until it stops changing."""
        prev = word
        current = self.apply(word)
        while current != prev:
            prev = current
            current = self.apply(current)
        return current

    def iterate(self, word: List[str], n: int) -> List[str]:
        """Apply rules to the word `n` times."""
        s = word
        for _ in range(n):
            s = self.apply(s)
        return s

    def iterate_until(self, word: List[str], length: int) -> str:
        """Apply rules to the word until its length is >= `length`."""
        s = word
        while len(s) < length:
            cache = s
            s = self.apply(s)
            if s == cache:
                break
        return self._to_str(s)

    def _to_str(self, word: List[str]) -> str:
        """Turn the word representation into a single Turtle string."""
        filtered = [letter
                    for letter in word
                    if letter not in self.rules.keys()]
        return "".join(filtered)


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
        # 'random': meta_S0LSystem(),
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
                LSystem.render(
                    word,
                    d=5,
                    theta=angle,
                    filename=f'{out_dir}/{name}-{angle}'
                    f'[{sample}]-{level:02d}.svg'
                )
            print()


if __name__ == '__main__':
    # test_to_sticks()
    draw_systems(out_dir='../out/')

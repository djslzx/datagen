import turtle
from random import choice, choices
from typing import Dict, List, Generator, Union
import util


def setup_turtle():
    if turtle.isvisible():
        # turtle.screensize(5000, 5000)
        turtle.mode('logo')
        turtle.hideturtle()
        turtle.speed(0)
        turtle.tracer(0, 0)
        turtle.setheading(0)


class LSystem:

    def __init__(self):
        self.render_is_setup = False

    def axiom(self) -> str:
        assert False, f"Should be implemented in child {type(self).__name__}"

    def expand(self, s: str) -> str:
        assert False, f"Should be implemented in child {type(self).__name__}"

    def expansions(self, iters: int) -> Generator[str, None, None]:
        """Returns a generator over `iters` expansions."""
        word = self.axiom()
        yield word
        for _ in range(iters):
            word = self.expand(word)
            yield word

    def nth_expansion(self, n: int) -> str:
        """Returns the n-th expansion."""
        word = self.axiom()
        for _ in range(n):
            word = self.expand(word)
        return word

    def render(self, s: str, length: float, angle: float, filename: str):
        """Renders the L-System using Turtle graphics."""
        if not self.render_is_setup:
            setup_turtle()
            self.render_is_setup = True

        stack = []
        for c in s:
            if c == 'F':
                # move forward and draw a line
                turtle.pendown()
                turtle.forward(length)
                turtle.penup()
            elif c == 'f':
                # move forward without drawing
                turtle.forward(length)
            elif c == '+':
                turtle.left(angle)
            elif c == '-':
                turtle.right(angle)
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


class DOLSystem(LSystem):
    """
    A deterministic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self, axiom: str, productions: Dict[str, str]):
        super().__init__()
        self.axiom = lambda: axiom
        self.productions = productions

    def expand(self, s: str) -> str:
        # Assume identity production if predecessor is not in self.productions
        return ''.join(self.productions.get(c, c) for c in s)


class SOLSystem(LSystem):
    """
    A stochastic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self,
                 axiom: str,
                 productions: Dict[str, List[str]],
                 distribution: Union[str, Dict[str, List[float]]]):
        super().__init__()
        self.axiom = lambda: axiom
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
        return f'axiom: {self.axiom()}\n' + \
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

    def apply(self, word: List[str]) -> List[str]:
        # choose one of the productions nondeterministically
        return [letter
                for letter in word
                for letter in choice(self.rules.get(letter, [letter]))]

    def fixpoint(self, word: List[str], max_iters: int) -> List[str]:
        if max_iters == 0:
            return word
        s = word
        for i in range(max_iters):
            s = self.apply(s)
        return s

    def to_str(self, word: List[str]) -> str:
        # remove artefacts of grammar
        # filtered = [letter
        #             for letter in word
        #             if letter not in self.rules.keys()]
        return "".join(word)


if __name__ == '__main__':
    RENDER_DIR = '../imgs'
    systems: Dict[str, LSystem] = {
        'koch': DOLSystem(
            axiom='F-F-F-F',
            productions={
                'F': 'F-F+F+FF-F-F+F'
            },
        ),
        'islands': DOLSystem(
            axiom='F+F+F+F',
            productions={
                'F': 'F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF',
                'f': 'ffffff',
            },
        ),
        'branch': DOLSystem(
            axiom='F',
            productions={
                'F': 'F[+F]F[-F]F'
            },
        ),
        'wavy-branch': DOLSystem(
            axiom='F',
            productions={
                'F': 'FF-[-F+F+F]+[+F-F-F]'
            },
        ),
        'stochastic-branch': SOLSystem(
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
        'random-walk': SOLSystem(
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
        'triplet': SOLSystem(
            axiom='F',
            productions={
                'F': ['FF',
                      '[-F]F',
                      '[+F]F'],
            },
            distribution='uniform'
        ),
        # 'random': meta_SOLSystem(),
    }

    for name, angle, levels, samples in [
        # ('koch', 90, 4, 1),
        # ('islands', 90, 3, 1),
        # ('branch', 25.7, 5, 1),
        # ('branch', 73, 5, 1),
        # ('wavy-branch', 22.5, 5, 1),
        # ('stochastic-branch', 22.5, 5, 5),
        # ('random-walk', 90, 99, 1),
        # ('triplet', 35, 5, 6),
        # ('random', 45, 6, 3),
    ]:
        for sample in range(samples):
            dirpath = f'{RENDER_DIR}/{name}-{angle}'
            util.try_mkdir(dirpath)
            system = systems[name]
            print(system)
            for level, word in enumerate(system.expansions(levels)):
                print(word)
                system.render(
                    word,
                    length=5,
                    angle=angle,
                    filename=f'{dirpath}/{name}[{sample}]-{level:02d}'
                )
            print()

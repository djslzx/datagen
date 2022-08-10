import turtle
from random import choices
from typing import Dict, List, Generator

turtle.mode('logo')
turtle.hideturtle()
turtle.speed(0)
turtle.tracer(0, 0)
turtle.setheading(0)
# turtle.screensize(5000, 5000)

class LSystem:

    def axiom(self) -> str:
        assert False, f"Should be implemented in child {type(self).__name__}"

    def expand(self, s: str) -> str:
        assert False, f"Should be implemented in child {type(self).__name__}"

    def expansions(self, iters: int) -> Generator[str, None, None]:
        word = self.axiom()
        for _ in range(iters):
            yield word
            word = self.expand(word)

    def render(s: str, length: float, angle: float, filename: str):
        stack = []
        for c in s:
            assert c in ['F', 'f', '+', '-', '[', ']'], \
                f'Expected the render string to be a turtle command, but found {c}'
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

    def render_expansions(self, iters: int, length: float, angle: float, filename: str):
        for i, w in enumerate(self.expansions(iters)):
            LSystem.render(w, length=length, angle=angle, filename=f'{filename}-{i}')


class DOLSystem(LSystem):
    """
    A deterministic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self, axiom: str, productions: Dict[str, str]):
        self.axiom = lambda: axiom
        self.productions = productions
        
    def expand(self, s: str) -> str:
        # Assume identity production if predecessor is not in `self.productions`
        return ''.join(self.productions.get(c, c) for c in s)


class SOLSystem(LSystem):
    """
    A stochastic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self,
                 axiom: str,
                 productions: Dict[str, List[str]],
                 distribution: Dict[str, List[float]]):
        self.axiom = lambda: axiom
        self.productions = productions
        
        # distribution
        assert all(sum(weights) == 1 for rule, weights in distribution.items()), \
            "All rules with the same predecessor should have probabilities summing to 1"
        
        self.distribution = distribution

    def expand(self, s: str) -> str:
        return ''.join(choices(population=self.productions.get(c, [c]),
                               weights=self.distribution.get(c, [1]))[0]
                       for c in s)


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
        )
    }

    for name, id, angle, iters in [
        # ('koch', 90, 3),
        # ('islands', 90, 3),
        # ('branch', 25.7, 5),
        # ('branch', 73, 5),
        # ('wavy-branch', 22.5, 5),
        ('stochastic-branch', 1, 22.5, 5),
        ('stochastic-branch', 2, 22.5, 5),
        ('stochastic-branch', 3, 22.5, 5),
        ('stochastic-branch', 4, 22.5, 5),
        ('stochastic-branch', 5, 22.5, 5),
    ]:
        systems[name].render_expansions(
            iters=iters,
            length=5,
            angle=angle,
            filename=f'{RENDER_DIR}/{name}[{id}]-{angle}'
        )

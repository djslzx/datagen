import turtle
from random import choice
from typing import Dict, List, Generator

turtle.hideturtle()
turtle.speed(0)
turtle.tracer(0, 0)
turtle.screensize(5000, 5000)

class LSystem:

    def axiom(self) -> str:
        assert False, f"Should be implemented in child {type(self).__name__}"

    def expand(self, s: str) -> str:
        assert False, f"Should be implemented in child {type(self).__name__}"

    def expansions(self, cap: int) -> Generator[str, None, None]:
        word = self.axiom()
        for _ in range(cap + 1):
            yield word
            word = self.expand(word)

    def render(s: str, length: float, angle: float, filename: str):
        for c in s:
            assert c in ['F', 'f', '+', '-'], \
                f'Expected the render string to be a turtle command, but found {c}'
            if c == 'F':
                turtle.Turtle
                turtle.pendown()
                turtle.forward(length)
                turtle.penup()
            elif c == 'f':
                turtle.forward(length)
            elif c == '+':
                turtle.left(angle)
            elif c == '-':
                turtle.right(angle)
        turtle.update()
        turtle.getcanvas().postscript(
            file=f'{filename}.ps',
            colormode='color',
            
        )
        turtle.clear()
        turtle.setpos(0, 0)


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


class OLSystem(LSystem):
    """
    A nondeterministic context-free Lindenmayer system
    where the alphabet is the collection of ASCII characters
    """

    def __init__(self, axiom: str, productions: Dict[str, List[str]]):
        self.axiom = lambda: axiom
        self.productions = productions

    def expand(self, s: str) -> str:
        # Choose one of the predecessor's productions uniformly at random
        return ''.join(choice(self.productions.get(c, [c])) for c in s)


if __name__ == '__main__':
    RENDER_DIR = '../imgs'
    systems = {
        'filament': DOLSystem(
            axiom='b',
            productions={
                'a': 'ab',
                'b': 'a',
            },
        ),
        'koch': DOLSystem(
            axiom='F-F-F-F',
            productions={
                'F': 'F-F+F+FF-F-F+F'
            },
        ),
        'quad_koch': DOLSystem(
            axiom='F-F-F-F',
            productions={
                'F': 'F+FF-FF-F-F+F+FF-F-F+F+FF+FF-F'
            },
        ),
        'quad_snowflake': DOLSystem(
            axiom='-F',
            productions={
                'F': 'F+F-F-F+F'
            },
        ),
        'islands': DOLSystem(
            axiom='F+F+F+F',
            productions={
                'F': 'F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF',
                'f': 'ffffff',
            },
        ),
        'dragon': DOLSystem(
            axiom='F-F-F-F',
            productions={
                'F': 'F-FF--F-F'
            },
        ),
    }

    for system, n in [
        # ('koch', 4),
        # ('quad_koch', 4),
        # ('quad_snowflake', 4),
        # ('islands', 2),
        ('dragon', 5),
    ]:
        for i, w in enumerate(systems[system].expansions(n)):
            print(w)
            LSystem.render(w, length=5, angle=90, filename=f'{RENDER_DIR}/{system}-{i}')
        

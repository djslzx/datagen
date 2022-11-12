import pytest
from lindenmayer import *
import matplotlib.pyplot as plt
import util
from book_zoo import simple_zoo_systems


def test_D0L_expand():
    cases = [
        (D0LSystem("F", {"F": "FF"}), "F", "FF"),
    ]
    for sys, s_in, s_out in cases:
        out = sys.expand(s_in)
        assert s_out == out, f"Expected {s_out} but got {out}"


def test_S0L_init():
    cases = [
        (S0LSystem("F",
                   {"F": ["FF", "[+F]F"]},
                   {"F": [1, 1]}),
         np.array([0.5, 0.5])),
        (S0LSystem("F",
                   {"F": ["FF", "[+F]F"]},
                   {"F": [2, 1]}),
         np.array([2/3, 1/3])),
        (S0LSystem("F",
                   {"F": ["FF", "[+F]F"]},
                   "uniform"),
         np.array([0.5, 0.5])),
    ]
    for sys, distro in cases:
        out = sys.distribution["F"]
        assert util.vec_approx_eq(out, distro), \
            f"Expected {distro} but got {out} for system {sys}"


def test_S0L_expand():
    cases = [
        (S0LSystem("F", {"F": ["F"]}), {"F"}),
        (S0LSystem("F", {"F": ["FF", "FFF"]}), {"FF", "FFF"}),
        (S0LSystem("FF", {"F": ["F", "FF"]}), {"FF", "FFF", "FFFF"}),
    ]
    for sys, s in cases:
        out = sys.expand(sys.axiom)
        assert out in s, f"Expected elt in {s}, but got {out} for sys {sys}"


def test_S0L_eq():
    cases = [
        (S0LSystem("F", {"F": ["FF", "FFF"]}),
         S0LSystem("F", {"F": ["FF", "FFF"]}),
         True),
        # different order
        (S0LSystem("F", {"F": ["FF", "FFF"]}),
         S0LSystem("F", {"F": ["FFF", "FF"]}),
         False),
        # different elts
        (S0LSystem("F", {"F": ["F", "FFF"]}),
         S0LSystem("F", {"F": ["F", "FF"]}),
         False),
        # key subset
        (S0LSystem("F", {"F": ["FF"]}),
         S0LSystem("F", {"F": ["FF"],
                         "f": ["ff"]}),
         False),
    ]
    for sys1, sys2, ans in cases:
        assert (sys1 == sys2) == (sys2 == sys1), "Equality should be symmetric"
        out = sys1 == sys2
        assert ans == out, f"Expected {sys1} == {sys2} = {ans}, but got {out}"


def test_S0L_from_sentence():
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


def test_S0L_to_sentence():
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


def test_LSYSTEM_MG_coverage():
    """Check that LSYSTEM_MG covers the book examples"""
    for sys in simple_zoo_systems:
        ex = sys.to_sentence()
        assert LSYSTEM_MG.can_generate(ex), f"Expected {LSYSTEM_MG} to generate {ex}"


def demo_draw():
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
    }

    for name, angle, levels, samples in [
        ('koch', 90, 4, 1),
        ('islands', 90, 3, 1),
        ('branch', 25.7, 5, 1),
        ('branch', 73, 5, 1),
        ('wavy-branch', 22.5, 5, 1),
        ('stochastic-branch', 22.5, 5, 5),
    ]:
        for sample in range(samples):
            system = systems[name]
            print(system)
            for _, word in enumerate(system.expansions(levels)):
                mat = LSystem.draw(s=word, d=5, theta=angle)
                plt.imshow(mat)
                plt.show()
            print()


if __name__ == '__main__':
    demo_draw()

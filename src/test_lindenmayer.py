import pytest
import copy
from lindenmayer import *
import matplotlib.pyplot as plt
import util
from book_zoo import simple_zoo_systems


def test_D0L_expand():
    cases = [
        (D0LSystem("F", {"F": "FF"}), "F", "FF"),
        (D0LSystem("F-F", {"F": "FF"}), "F", "FF"),
        (D0LSystem("F", {"F": "FF"}), "F-F", "FF-FF"),
        (D0LSystem("F-F", {"F": "FF"}), "F-F", "FF-FF"),
    ]
    for sys, s_in, s_out in cases:
        out = sys.expand(s_in)
        assert s_out == out, f"Expected {s_out} but got {out}"


def test_D0L_expansions():
    cases = [
        (D0LSystem("F", {"F": "FF"}), ["F", "FF", "FFFF", "FFFFFFFF"]),
        (D0LSystem("F-F", {"F": "FF"}), ["F-F", "FF-FF", "FFFF-FFFF"]),
    ]
    for sys, expansions in cases:
        iters = len(expansions)

        # test D0L.expansions
        out = list(sys.expansions(iters - 1))
        assert len(out) == iters, f"Mismatched lengths: |out|={len(out)}, |ans|={iters}"
        for i, s_hat, s in zip(range(iters), out, expansions):
            assert s == s_hat, f"Expected {i}-th expansion of {sys} to be {s}, but got {s_hat}"

        # test D0L.nth_expansion
        out = [sys.nth_expansion(i) for i in range(iters)]
        assert len(out) == iters, f"Mismatched lengths: |out|={len(out)}, |ans|={iters}"
        for i, s_hat, s in zip(range(iters), out, expansions):
            assert s == s_hat, f"Expected {i}-th expansion of {sys} to be {s}, but got {s_hat}"


def test_S0L_expansions():
    cases = [
        (S0LSystem("F", {"F": ["FF"]}), ["F", "FF", "FFFF", "FFFFFFFF"]),
        (S0LSystem("F", {"F": ["F", "FF"]}), [{"F"},
                                              {"F", "FF"},
                                              {"F", "FF", "FFF", "FFFF"}]),
    ]
    for sys, expansions in cases:
        for _ in range(100):
            iters = len(expansions)

            # test S0L.expansions
            out = list(sys.expansions(iters - 1))
            assert len(out) == iters, f"Mismatched lengths: |out|={len(out)}, |ans|={iters}"
            for i, s, S in zip(range(iters), out, expansions):
                assert s in S, f"Expected {i}-th expansion of {sys} to be in {S}, but got {s}"

            # test S0L.nth-expansion
            out = [sys.nth_expansion(i) for i in range(iters)]
            assert len(out) == iters, f"Mismatched lengths: |out|={len(out)}, |ans|={iters}"
            for i, s, S in zip(range(iters), out, expansions):
                assert s in S, f"Expected {i}-th expansion of {sys} to be in {S}, but got {s}"


def test_0L_stationary_expand():
    cases: List[Tuple[LSystem, str]] = [
        (D0LSystem("F", {"F": "F"}), "F"),
        (S0LSystem("F", {"F": ["F"]}), "F"),
    ]
    n_expands = 10
    for sys, string in cases:
        # expansions
        for exp in sys.expansions(n_expands):
            assert exp == string

        # nth_expansion
        for i in range(n_expands):
            exp = sys.nth_expansion(i)
            assert exp == string

        # expand until
        for i in range(2, n_expands):
            depth, exp = sys.expand_until(i)
            assert depth == 0
            assert exp == string


def test_D0L_expand_until():
    cases = [
        (D0LSystem("F", {"F": "FF"}), 1, (0, "F")),
        (D0LSystem("F", {"F": "FF"}), 3, (2, "FFFF")),
        (D0LSystem("F", {"F": "FF"}), 5, (3, "FFFFFFFF")),
        (D0LSystem("F", {"F": "[F-F]"}), 1, (0, "F")),
        (D0LSystem("F", {"F": "[F-F]"}), 2, (1, "[F-F]")),
        (D0LSystem("F", {"F": "[F-F]"}), 5, (1, "[F-F]")),
        (D0LSystem("F", {"F": "[F-F]"}), 6, (2, "[[F-F]-[F-F]]")),
    ]
    for sys, length, ans in cases:
        depth, expansion = sys.expand_until(length)
        assert len(expansion) >= length
        # no clear upper bound: could have seed expr of length k-1,
        # but have rules that expand out the seed expr indefinitely
        assert ans == (depth, expansion), \
            f"Expected length {length} expansion to be {ans}, but got {depth, expansion}"


def test_S0L_expand_until():
    cases = [
        (S0LSystem("F", {"F": ["FF"]}), 1, {(0, "F")}),
        (S0LSystem("F", {"F": ["FF"]}), 3, {(2, "FFFF")}),
        (S0LSystem("F", {"F": ["FF"]}), 5, {(3, "FFFFFFFF")}),
        (S0LSystem("F", {"F": ["FF", "FFF"]}), 1, {(0, "F")}),
        (S0LSystem("F", {"F": ["FF", "FFF"]}), 2, {(1, "FF"), (1, "FFF")}),
        (S0LSystem("F", {"F": ["FF", "FFF"]}), 3, {(1, "FFF"),
                                                   (2, "FFFF"),
                                                   (2, "FFFFF"),
                                                   (2, "FFFFFF")}),
    ]
    for sys, length, options in cases:
        for _ in range(100):
            depth, expansion = sys.expand_until(length)
            assert len(expansion) >= length
            assert (depth, expansion) in options, \
                f"Expected expansion for {sys} to be in {options}, but got {depth, expansion}"


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


def test_S0L_to_code():
    cases = [
        (S0LSystem('F', {'F': ['fF']}, 'uniform'),
         'F;F~fF'),
        (S0LSystem('F', {'F': ['A'], 'A': ['Ab', 'bb', 'AA']}, 'uniform'),
         'F;F~A,A~Ab,A~bb,A~AA'),
        (S0LSystem('F-F-F', {'F': ['fF']}, 'uniform'),
         'F-F-F;F~fF'),
    ]
    for g, y in cases:
        out = g.to_code()
        assert out == y, f"Expected {y}, but got {out}"


def test_LSYSTEM_MG_coverage():
    """Check that LSYSTEM_MG covers the book examples"""
    for sys in simple_zoo_systems:
        ex = sys.to_sentence()
        assert MG.can_generate(ex), \
            f"Expected\n{MG}\nCNF:{MG.to_CNF()}\nto generate {ex}"


def test_parse_lsystem_str_as_tree():
    cases = [
        ("F;F~F",
         ("LSystem", 0,
          ("Axiom", 0,
           ("Nonterminal", 0),
           ("Axiom", 2)),
          ("Rules", 1,
           ("Rule", 0,
            ("Nonterminal", 0),
            ("Rhs", 1,
             ("Nonterminal", 0),
             ("Rhs", 3)))))),

        ("F+F;F~+F,F~FF",
         ("LSystem", 0,
          ("Axiom", 0,
           ("Nonterminal", 0),
           ("Axiom", 1,
            ("Terminal", 0),
            ("Axiom", 0,
             ("Nonterminal", 0),
             ("Axiom", 2)))),
          ("Rules", 0,
           ("Rule", 0,  # F~+F
            ("Nonterminal", 0),
            ("Rhs", 2,
             ("Terminal", 0),
             ("Rhs", 1,
              ("Nonterminal", 0),
              ("Rhs", 3)))),
           ("Rules", 1,  # F~FF
            ("Rule", 0,
             ("Nonterminal", 0),
             ("Rhs", 1,
              ("Nonterminal", 0),
              ("Rhs", 1,
               ("Nonterminal", 0),
               ("Rhs", 3)))))))),

        ("F;F~[F]",
         ("LSystem", 0,
          ("Axiom", 0,
           ("Nonterminal", 0),
           ("Axiom", 2)),
          ("Rules", 1,
           ("Rule", 0,
            ("Nonterminal", 0),
            ("Rhs", 0,
             ("Rhs", 1,
              ("Nonterminal", 0),
              ("Rhs", 3)),
             ("Rhs", 3)))))),
    ]
    for s, tree in cases:
        out = parse_lsystem_str_as_tree(s)
        assert tree == out, f"Expected\n{tree}\nbut got\n{out}"


def test_parse_lsystem_str_as_counts():
    cases = [
        ("F;F~F", {
            ("LSystem", 0): 1,
            ("Axiom", 0): 1,
            ("Axiom", 2): 1,
            ("Nonterminal", 0): 3,
            ("Rules", 1): 1,
            ("Rule", 0): 1,
            ("Rhs", 1): 1,
            ("Rhs", 3): 1,
        }),
        ("F;F~FF", {
            ("LSystem", 0): 1,
            ("Axiom", 0): 1,
            ("Axiom", 2): 1,
            ("Nonterminal", 0): 4,
            ("Rules", 1): 1,
            ("Rule", 0): 1,
            ("Rhs", 1): 2,
            ("Rhs", 3): 1,
        }),
        ("F+F;F~F", {
            ("LSystem", 0): 1,
            ("Axiom", 0): 2,
            ("Axiom", 1): 1,
            ("Axiom", 2): 1,
            ("Terminal", 0): 1,
            ("Nonterminal", 0): 4,
            ("Rules", 1): 1,
            ("Rule", 0): 1,
            ("Rhs", 1): 1,
            ("Rhs", 3): 1,
        }),
        ("F;F~[F]", {
            ("LSystem", 0): 1,
            ("Axiom", 0): 1,
            ("Axiom", 2): 1,
            ("Nonterminal", 0): 3,
            ("Rules", 1): 1,
            ("Rule", 0): 1,
            ("Rhs", 0): 1,
            ("Rhs", 1): 1,
            ("Rhs", 3): 2,
        }),
    ]
    for s, d in cases:
        ans = empty_mg_counts()
        for (nt, i), n in d.items():
            ans[nt][i] = n
        out = parse_lsystem_str_as_counts(s)
        assert out.keys() == ans.keys() and all(np.array_equal(out[k], ans[k]) for k in out.keys()), \
            f"Expected {ans} but got {out}"


def test_count_rules():
    cases = [
        (["F;F~F"] * 10, {
            ("LSystem", 0): 10,
            ("Axiom", 0): 10,
            ("Axiom", 2): 10,
            ("Nonterminal", 0): 30,
            ("Rules", 1): 10,
            ("Rule", 0): 10,
            ("Rhs", 1): 10,
            ("Rhs", 3): 10,
        }),
    ]
    for corpus, d in cases:
        ans = empty_mg_counts()
        for (nt, i), n in d.items():
            ans[nt][i] = n
        out = count_rules(corpus)
        assert out.keys() == ans.keys() and all(np.array_equal(out[k], ans[k]) for k in out.keys()), \
            f"Expected {ans} but got {out}"


def demo_weighted_metagrammar():  # pragma: no cover
    corpi = [
        ["F;F~F"],
        ["F;F~F"] * 3,
        ["F;F~F", "F;F~FF"],
        ["F;F~F", "F;F~FF", "F;F~FFF"],
        ["F+F;F~F[+F]F", "F-F;F~F+F", "F;F~F[+F]-FF"],
        [sys.to_code() for sys in simple_zoo_systems],
    ]
    for corpus in corpi:
        g = weighted_metagrammar(corpus)
        print(corpus)
        print(g)
    # TODO: bigram

def demo_draw():  # pragma: no cover
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


if __name__ == '__main__':  # pragma: no cover
    # demo_draw()
    demo_weighted_metagrammar()

import pytest
from cfg import *


def test_make_cfg():
    cfgs = [
        CFG("S", {"S": ["A B", "A", "B"],
                  "A": ["a1", "a2"],
                  "B": ["b1", "b2"]}),
        CFG("S", {"S": [["A", "B"], ["A"], ["B"]],
                  "A": [["a1"], ["a2"]],
                  "B": [["b1"], ["b2"]]}),
        CFG.from_rules("S", [
            ("S", "A B"),
            ("S", "A"),
            ("S", "B"),
            ("A", "a1"),
            ("A", "a2"),
            ("B", "b1"),
            ("B", "b2"),
        ]),
        CFG.from_rules("S", [
            ("S", ["A", "B"]),
            ("S", ["A"]),
            ("S", ["B"]),
            ("A", ["a1"]),
            ("A", ["a2"]),
            ("B", ["b1"]),
            ("B", ["b2"]),
        ]),
    ]
    assert all(cfgs[0] == cfg for cfg in cfgs[1:]), \
        f"Expected all CFGs to be equal, but got the following CFGs: {cfgs}"


def test_explode():
    cases = [
        (CFG.from_rules(
            start="S",
            rules=[
                ("S", ["A"]),
                ("A", ["a"]),
            ],
        ), CFG.from_rules(
            start="S",
            rules=[
                ("S", ["S_1"]),

                ("S_1", ["A_1"]),
                ("A_1", ["a"]),
            ],
        )),
        (CFG.from_rules(
            start="E",
            rules=[
                ("E", ["-", "E"]),
                ("E", ["E", "+", "E"]),
            ],
        ), CFG.from_rules(
            start="E",
            rules=[
                ("E", ["E_1"]),
                ("E", ["E_2"]),
                ("E", ["E_3"]),

                ("E_1", ["-", "E_1"]),
                ("E_1", ["-", "E_2"]),
                ("E_1", ["-", "E_3"]),
                ("E_1", ["E_1", "+", "E_1"]),
                ("E_1", ["E_1", "+", "E_2"]),
                ("E_1", ["E_1", "+", "E_3"]),
                ("E_1", ["E_2", "+", "E_1"]),
                ("E_1", ["E_2", "+", "E_2"]),
                ("E_1", ["E_2", "+", "E_3"]),
                ("E_1", ["E_3", "+", "E_1"]),
                ("E_1", ["E_3", "+", "E_2"]),
                ("E_1", ["E_3", "+", "E_3"]),

                ("E_2", ["-", "E_1"]),
                ("E_2", ["-", "E_2"]),
                ("E_2", ["-", "E_3"]),
                ("E_2", ["E_1", "+", "E_1"]),
                ("E_2", ["E_1", "+", "E_2"]),
                ("E_2", ["E_1", "+", "E_3"]),
                ("E_2", ["E_2", "+", "E_1"]),
                ("E_2", ["E_2", "+", "E_2"]),
                ("E_2", ["E_2", "+", "E_3"]),
                ("E_2", ["E_3", "+", "E_1"]),
                ("E_2", ["E_3", "+", "E_2"]),
                ("E_2", ["E_3", "+", "E_3"]),

                ("E_3", ["-", "E_1"]),
                ("E_3", ["-", "E_2"]),
                ("E_3", ["-", "E_3"]),
                ("E_3", ["E_1", "+", "E_1"]),
                ("E_3", ["E_1", "+", "E_2"]),
                ("E_3", ["E_1", "+", "E_3"]),
                ("E_3", ["E_2", "+", "E_1"]),
                ("E_3", ["E_2", "+", "E_2"]),
                ("E_3", ["E_2", "+", "E_3"]),
                ("E_3", ["E_3", "+", "E_1"]),
                ("E_3", ["E_3", "+", "E_2"]),
                ("E_3", ["E_3", "+", "E_3"]),
            ],
        )),
        (CFG.from_rules(
            start="A",
            rules=[
                ("A", ["a"]),
                ("A", ["B", "C"]),

                ("B", ["B", "B"]),
                ("B", ["C"]),

                ("C", ["A"]),
            ],
        ), CFG.from_rules(
            start="A",
            rules=[
                ("A", ["A_1"]),

                ("A_1", ["a"]),
                ("A_1", ["B_1", "C_1"]),
                ("A_1", ["B_1", "C_2"]),
                ("A_1", ["B_2", "C_1"]),
                ("A_1", ["B_2", "C_2"]),
                ("A_1", ["B_3", "C_1"]),
                ("A_1", ["B_3", "C_2"]),

                ("B_1", ["B_1", "B_1"]),
                ("B_1", ["B_1", "B_2"]),
                ("B_1", ["B_1", "B_3"]),
                ("B_1", ["B_2", "B_1"]),
                ("B_1", ["B_2", "B_2"]),
                ("B_1", ["B_2", "B_3"]),
                ("B_1", ["B_3", "B_1"]),
                ("B_1", ["B_3", "B_2"]),
                ("B_1", ["B_3", "B_3"]),
                ("B_1", ["C_1"]),
                ("B_1", ["C_2"]),

                ("B_2", ["B_1", "B_1"]),
                ("B_2", ["B_1", "B_2"]),
                ("B_2", ["B_1", "B_3"]),
                ("B_2", ["B_2", "B_1"]),
                ("B_2", ["B_2", "B_2"]),
                ("B_2", ["B_2", "B_3"]),
                ("B_2", ["B_3", "B_1"]),
                ("B_2", ["B_3", "B_2"]),
                ("B_2", ["B_3", "B_3"]),
                ("B_2", ["C_1"]),
                ("B_2", ["C_2"]),

                ("B_3", ["B_1", "B_1"]),
                ("B_3", ["B_1", "B_2"]),
                ("B_3", ["B_1", "B_3"]),
                ("B_3", ["B_2", "B_1"]),
                ("B_3", ["B_2", "B_2"]),
                ("B_3", ["B_2", "B_3"]),
                ("B_3", ["B_3", "B_1"]),
                ("B_3", ["B_3", "B_2"]),
                ("B_3", ["B_3", "B_3"]),
                ("B_3", ["C_1"]),
                ("B_3", ["C_2"]),

                ("C_1", ["A_1"]),

                ("C_2", ["A_1"]),
            ],
        )),
    ]
    for g, y in cases:
        out = g.explode()
        assert out == y, f"Expected {y}, but got {out}"


def test_to_bigram():
    # annotate repeated nonterminals with indices, then make
    # copies of the original rules with however many indices were used
    cases = [
        (CFG.from_rules(
            start="S",
            rules=[
                ("S", "a"),
            ],
        ), CFG.from_rules(
            start="S",
            rules=[
                ("S", "S_1"),
                ("S_1", "a"),
            ],
        )),
        (CFG.from_rules(
            start="S",
            rules=[
                ("S", "A"),
                ("A", "a"),
            ],
        ), CFG.from_rules(
            start="S",
            rules=[
                ("S", "S_1"),
                ("S_1", "A_1"),
                ("A_1", "a"),
            ],
        )),
        (CFG.from_rules(
            start="S",
            rules=[
                ("S", "a S"),
                ("S", "b S"),
                ("S", "c S"),
            ],
        ), CFG.from_rules(
            start="S",
            rules=[
                ("S", ["S_1"]),
                ("S", ["S_2"]),
                ("S", ["S_3"]),

                # 3 rules ^ 2
                ("S_1", ["a", "S_1"]),
                ("S_1", ["b", "S_2"]),
                ("S_1", ["c", "S_3"]),

                ("S_2", ["a", "S_1"]),
                ("S_2", ["b", "S_2"]),
                ("S_2", ["c", "S_3"]),

                ("S_3", ["a", "S_1"]),
                ("S_3", ["b", "S_2"]),
                ("S_3", ["c", "S_3"]),
            ],
        )),
        (CFG.from_rules(
            start="E",
            rules=[
                ("E", ["-", "E"]),
                ("E", ["E", "+", "E"]),
            ],
        ), CFG.from_rules(
            start="E",
            rules=[
                ("E", ["E_1"]),
                ("E", ["E_2"]),
                ("E", ["E_3"]),

                # 2 rules ^ 2
                ("E_1", ["-", "E_1"]),
                ("E_1", ["E_2", "+", "E_3"]),

                ("E_2", ["-", "E_1"]),
                ("E_2", ["E_2", "+", "E_3"]),

                ("E_3", ["-", "E_1"]),
                ("E_3", ["E_2", "+", "E_3"]),
            ],
        )),
        (CFG.from_rules(
            start="A",
            rules=[
                ("A", ["a"]),
                ("A", ["B", "C"]),
                ("A", ["B"]),
                ("A", ["C"]),
                ("B", ["B", "B"]),
                ("B", ["C"]),
                ("C", ["A"]),
            ],
        ), CFG.from_rules(
            start="A",
            rules=[
                ("A", ["A_1"]),

                ("A_1", ["a"]),
                ("A_1", ["B_1", "C_1"]),
                ("A_1", ["B_2"]),
                ("A_1", ["C_2"]),

                ("B_1", ["B_3", "B_4"]),
                ("B_1", ["C_3"]),
                ("B_2", ["B_3", "B_4"]),
                ("B_2", ["C_3"]),
                ("B_3", ["B_3", "B_4"]),
                ("B_3", ["C_3"]),
                ("B_4", ["B_3", "B_4"]),
                ("B_4", ["C_3"]),

                ("C_1", ["A_1"]),
                ("C_2", ["A_1"]),
                ("C_3", ["A_1"]),
            ],
        )),
    ]
    for g, y in cases:
        out = g.to_bigram()
        assert out == y, f"Expected {y}, but got {out}"


def test_term():
    cases = [
        (CFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B", "C", "D", "E", "F"],
                      ["D", "E", "F"],
                      [""]],
                "B": [["b"]],
                "C": [["c1", "c2", "c3"],
                      ["c4"],
                      [""]],
                "D": [["D", "d", "D"]],
                "E": [["e"]],
                "F": [["f"]],
            }),
         CFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B", "C", "D", "E", "F"],
                      ["D", "E", "F"],
                      [""]],
                "B": [["b"]],
                "C": [["_term_c1_", "_term_c2_", "_term_c3_"],
                      ["c4"],
                      [""]],
                "D": [["D", "_term_d_", "D"]],
                "E": [["e"]],
                "F": [["f"]],
                "_term_c1_": [["c1"]],
                "_term_c2_": [["c2"]],
                "_term_c3_": [["c3"]],
                "_term_d_": [["d"]],
            })),
    ]
    for x, y in cases:
        out = x._term()
        assert out == y, f"Failed test_term: Expected\n{y},\ngot\n{out}"


def test_bin():
    cases = [
        (CFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B", "C", "D", "E", "F"],
                      ["D", "E", "F"]],
                "B": [["b"]],
                "C": [["c"]],
                "D": [["d"]],
                "E": [["e"]],
                "F": [["f"]],
            },
        ), CFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B", "_bin_A_0_1_"],
                      ["D", "_bin_A_1_1_"]],
                "_bin_A_0_1_": [["C", "_bin_A_0_2_"]],
                "_bin_A_0_2_": [["D", "_bin_A_0_3_"]],
                "_bin_A_0_3_": [["E", "F"]],
                "_bin_A_1_1_": [["E", "F"]],
                "B": [["b"]],
                "C": [["c"]],
                "D": [["d"]],
                "E": [["e"]],
                "F": [["f"]],
            },
        )),
    ]
    for x, y in cases:
        out = x._bin()
        assert out == y, f"Failed test_bin: Expected\n{y},\ngot\n{out}"


def test_nullable():
    cases = [
        (CFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],
                "A": [["a"], CFG.Empty],
            },
        ), {"A"}),
        (CFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],
                "A": [["a"]],
            },
        ), set()),
        (CFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],       # nullable
                "A": [["B"], ["C", "a"]],  # nullable
                "B": [["C"]],              # nullable
                "C": [["x"], CFG.Empty],  # nullable
            },
        ), {"A", "B", "C"}),
    ]
    for x, y in cases:
        out = x.nullables()
        assert out == y, f"Expected {y}, got {out}"


def test_del():
    cases = [
        (CFG(
            start="S",
            rules={
                "S": [["A", "b", "B"], ["C"]],
                "A": [["a"], CFG.Empty],
                "B": [["A", "A"], ["A", "C"]],
                "C": [["b"], ["c"]],
            },
         ),
         CFG(
            start="S",
            rules={
                "S": [["A", "b", "B"], ["C"], ["b", "B"], ["A", "b"], ["b"]],
                "A": [["a"]],
                "B": [["A", "A"], ["A", "C"], ["A"], ["C"]],
                "C": [["b"], ["c"]],
            },
        )),
        (CFG(
            start="S",
            rules={
                "S": [["A", "s1"], ["A1"], ["A1", "s1"], ["A2", "A1", "s2"]],
                "A": [["A1"]],
                "A1": [["A2"]],
                "A2": [["a2"], CFG.Empty],
            },
         ),
         CFG(
            start="S",
            rules={
                "S": [["A", "s1"], ["s1"],
                      ["A1"],
                      ["A1", "s1"],
                      ["A2", "A1", "s2"], ["A2", "s2"], ["A1", "s2"], ["s2"]],
                "A": [["A1"]],
                "A1": [["A2"]],
                "A2": [["a2"]]
            },
        )),
    ]
    for x, y in cases:
        out = x._del()
        assert out == y, f"Failed test_del: Expected\n{y},\ngot\n{out}"


def test_unit():
    cases = [
        (CFG(
            start="S",
            rules={
                "S": [["A"], ["s", "s", "s"]],
                "A": [["a", "b", "c"], ["e", "f"]],
            },
        ), CFG(
            start="S",
            rules={
                "S": [["a", "b", "c"], ["e", "f"], ["s", "s", "s"]],
            },
        )),
        (CFG(
            start="S",
            rules={
                "S": [["A", "B"], ["C"]],
                "A": [["a"]],
                "B": [["b"]],
                "C": [["c"]],
            },
        ), CFG(
            start="S",
            rules={
                "S": [["A", "B"], ["c"]],
                "A": [["a"]],
                "B": [["b"]],
            },
        )),
        (CFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B"]],
                "B": [["C"]],
                "C": [["c"]],
            },
        ), CFG(
            start="S",
            rules={
                "S": [["c"]],
            },
        )),
        (CFG(
            start='S0',
            rules={
                'S0': [['S']],
                'S': [['A'], ['A', 'B']],
                'A': [['A', 'A']],
                'B': [['B', 'A']],
            },
         ), CFG(
             start='S0',
             rules={
                 'S0': [['A', 'A'], ['A', 'B']],
                 'A': [['A', 'A']],
                 'B': [['B', 'A']],
             },
        )),
    ]
    for x, y in cases:
        out = x._unit()
        assert out == y, f"Failed test_unit: Expected\n{y},\ngot\n{out}"


def test_to_CNF():
    test_term()
    test_bin()
    test_nullable()
    test_del()
    test_unit()


def test_is_in_CNF():
    cases = [
        # unit
        ({
            "S": [["A"]],
            "A": [["a"]],
        }, False),
        ({
            "S": [["a"]],
        }, True),
        # term/nonterm mix
        ({
            "S": [["A", "b"]],
            "A": [["a"]],
        }, False),
        ({
            "S": [["A", "B"]],
            "A": [["a"]],
            "B": [["b"]],
        }, True),
        ({
            "S": [["A", "B"], CFG.Empty],
            "A": [["a"]],
            "B": [["b"]],
        }, True),
        # three succs
        ({
            "S": [["A", "B", "C"]],
            "A": [["a"]],
            "B": [["b"]],
            "C": [["c"]],
        }, False),
        # empty successor in non-start nonterminal
        ({
            "S": [["A", "B"]],
            "A": [CFG.Empty],
            "B": [["b"]],
        }, False),

    ]
    for rules, y in cases:
        g = CFG("S", rules)
        out = g.is_in_CNF()
        assert out == y, \
            f"Failed test_is_in_CNF for {g}: Expected {y}, but got {out}"


def test_is_normalized():
    cases = [
        (PCFG("S", {"S": ["a"]}, {"S": [1]}), True),
        (PCFG("S", {"S": ["a", "b"]}, {"S": [1 / 2, 1 / 2]}), True),
        (PCFG("S", {"S": ["a", "b"]}, {"S": [1, 1]}), False),
        (PCFG("S", {"S": ["a"]}, {"S": [0]}, log_mode=True), True),
        (PCFG("S", {"S": ["a"]}, {"S": [1]}, log_mode=True), False),
        (PCFG("S", {"S": ["a", "b"]}, {"S": [math.log(1 / 2), math.log(1 / 2)]}, log_mode=True), True),
        (PCFG("S", {"S": ["a", "b"]}, {"S": [1/2, 1/2]}, log_mode=True), False),
    ]
    for g, y in cases:
        y_hat = g.is_normalized()
        assert y == y_hat, f"Expected {y}, but got {y_hat} for grammar {g}"

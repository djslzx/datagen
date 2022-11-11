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
        (PCFG.from_weighted_rules(
            start="S",
            rules=[
                ("S", ["A"], 1),
                ("A", ["a"], 1),
            ],
        ), PCFG.from_weighted_rules(
            start="S",
            rules=[
                ("S", ["S_1"], 1),

                ("S_1", ["A_1"], 1),
                ("A_1", ["a"], 1),
            ],
        )),
        (PCFG.from_weighted_rules(
            start="E",
            rules=[
                ("E", ["-", "E"], 0.5),
                ("E", ["E", "+", "E"], 0.5),
            ],
        ), PCFG.from_weighted_rules(
            start="E",
            rules=[
                ("E", ["E_1"], 0.333),
                ("E", ["E_2"], 0.333),
                ("E", ["E_3"], 0.333),

                ("E_1", ["-", "E_1"], 0.5 / 3),
                ("E_1", ["-", "E_2"], 0.5 / 3),
                ("E_1", ["-", "E_3"], 0.5 / 3),
                ("E_1", ["E_1", "+", "E_1"], 0.5 / 9),
                ("E_1", ["E_1", "+", "E_2"], 0.5 / 9),
                ("E_1", ["E_1", "+", "E_3"], 0.5 / 9),
                ("E_1", ["E_2", "+", "E_1"], 0.5 / 9),
                ("E_1", ["E_2", "+", "E_2"], 0.5 / 9),
                ("E_1", ["E_2", "+", "E_3"], 0.5 / 9),
                ("E_1", ["E_3", "+", "E_1"], 0.5 / 9),
                ("E_1", ["E_3", "+", "E_2"], 0.5 / 9),
                ("E_1", ["E_3", "+", "E_3"], 0.5 / 9),

                ("E_2", ["-", "E_1"], 0.5 / 3),
                ("E_2", ["-", "E_2"], 0.5 / 3),
                ("E_2", ["-", "E_3"], 0.5 / 3),
                ("E_2", ["E_1", "+", "E_1"], 0.5 / 9),
                ("E_2", ["E_1", "+", "E_2"], 0.5 / 9),
                ("E_2", ["E_1", "+", "E_3"], 0.5 / 9),
                ("E_2", ["E_2", "+", "E_1"], 0.5 / 9),
                ("E_2", ["E_2", "+", "E_2"], 0.5 / 9),
                ("E_2", ["E_2", "+", "E_3"], 0.5 / 9),
                ("E_2", ["E_3", "+", "E_1"], 0.5 / 9),
                ("E_2", ["E_3", "+", "E_2"], 0.5 / 9),
                ("E_2", ["E_3", "+", "E_3"], 0.5 / 9),

                ("E_3", ["-", "E_1"], 0.5 / 3),
                ("E_3", ["-", "E_2"], 0.5 / 3),
                ("E_3", ["-", "E_3"], 0.5 / 3),
                ("E_3", ["E_1", "+", "E_1"], 0.5 / 9),
                ("E_3", ["E_1", "+", "E_2"], 0.5 / 9),
                ("E_3", ["E_1", "+", "E_3"], 0.5 / 9),
                ("E_3", ["E_2", "+", "E_1"], 0.5 / 9),
                ("E_3", ["E_2", "+", "E_2"], 0.5 / 9),
                ("E_3", ["E_2", "+", "E_3"], 0.5 / 9),
                ("E_3", ["E_3", "+", "E_1"], 0.5 / 9),
                ("E_3", ["E_3", "+", "E_2"], 0.5 / 9),
                ("E_3", ["E_3", "+", "E_3"], 0.5 / 9),
            ],
        )),
        (PCFG.from_weighted_rules(
            start="A",
            rules=[
                ("A", ["a"], 0.5),
                ("A", ["B", "C"], 0.5),

                ("B", ["B", "B"], 0.5),
                ("B", ["C"], 0.5),

                ("C", ["A"], 1),
            ],
        ), PCFG.from_weighted_rules(
            start="A",
            rules=[
                ("A", ["A_1"], 1),

                ("A_1", ["a"], 0.5),
                ("A_1", ["B_1", "C_1"], 0.5 / 6),
                ("A_1", ["B_1", "C_2"], 0.5 / 6),
                ("A_1", ["B_2", "C_1"], 0.5 / 6),
                ("A_1", ["B_2", "C_2"], 0.5 / 6),
                ("A_1", ["B_3", "C_1"], 0.5 / 6),
                ("A_1", ["B_3", "C_2"], 0.5 / 6),

                ("B_1", ["B_1", "B_1"], 0.5 / 9),
                ("B_1", ["B_1", "B_2"], 0.5 / 9),
                ("B_1", ["B_1", "B_3"], 0.5 / 9),
                ("B_1", ["B_2", "B_1"], 0.5 / 9),
                ("B_1", ["B_2", "B_2"], 0.5 / 9),
                ("B_1", ["B_2", "B_3"], 0.5 / 9),
                ("B_1", ["B_3", "B_1"], 0.5 / 9),
                ("B_1", ["B_3", "B_2"], 0.5 / 9),
                ("B_1", ["B_3", "B_3"], 0.5 / 9),
                ("B_1", ["C_1"], 0.5 / 2),
                ("B_1", ["C_2"], 0.5 / 2),

                ("B_2", ["B_1", "B_1"], 0.5 / 9),
                ("B_2", ["B_1", "B_2"], 0.5 / 9),
                ("B_2", ["B_1", "B_3"], 0.5 / 9),
                ("B_2", ["B_2", "B_1"], 0.5 / 9),
                ("B_2", ["B_2", "B_2"], 0.5 / 9),
                ("B_2", ["B_2", "B_3"], 0.5 / 9),
                ("B_2", ["B_3", "B_1"], 0.5 / 9),
                ("B_2", ["B_3", "B_2"], 0.5 / 9),
                ("B_2", ["B_3", "B_3"], 0.5 / 9),
                ("B_2", ["C_1"], 0.5 / 2),
                ("B_2", ["C_2"], 0.5 / 2),

                ("B_3", ["B_1", "B_1"], 0.5 / 9),
                ("B_3", ["B_1", "B_2"], 0.5 / 9),
                ("B_3", ["B_1", "B_3"], 0.5 / 9),
                ("B_3", ["B_2", "B_1"], 0.5 / 9),
                ("B_3", ["B_2", "B_2"], 0.5 / 9),
                ("B_3", ["B_2", "B_3"], 0.5 / 9),
                ("B_3", ["B_3", "B_1"], 0.5 / 9),
                ("B_3", ["B_3", "B_2"], 0.5 / 9),
                ("B_3", ["B_3", "B_3"], 0.5 / 9),
                ("B_3", ["C_1"], 0.5 / 2),
                ("B_3", ["C_2"], 0.5 / 2),

                ("C_1", ["A_1"], 1),

                ("C_2", ["A_1"], 1),
            ],
        )),
    ]
    for g, y in cases:
        out = g.explode()
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_explode")


def test_to_bigram():
    # annotate repeated nonterminals with indices, then make
    # copies of the original rules with however many indices were used
    cases = [
        (PCFG.from_weighted_rules(
            start="S",
            rules=[
                ("S", ["a"], 1),
            ],
        ), PCFG.from_weighted_rules(
            start="S",
            rules=[
                ("S", ["S_1"], 1),
                ("S_1", ["a"], 1),
            ],
        )),
        (PCFG.from_weighted_rules(
            start="S",
            rules=[
                ("S", ["A"], 1),
                ("A", ["a"], 1),
            ],
        ), PCFG.from_weighted_rules(
            start="S",
            rules=[
                ("S", ["S_1"], 1),
                ("S_1", ["A_1"], 1),
                ("A_1", ["a"], 1),
            ],
        )),
        (PCFG.from_weighted_rules(
            start="S",
            rules=[
                ("S", ["a", "S"], 0.333),
                ("S", ["b", "S"], 0.333),
                ("S", ["c", "S"], 0.333),
            ],
        ), PCFG.from_weighted_rules(
            start="S",
            rules=[
                ("S", ["S_1"], 0.333),
                ("S", ["S_2"], 0.333),
                ("S", ["S_3"], 0.333),

                # 3 rules ^ 2
                ("S_1", ["a", "S_1"], 0.333),
                ("S_1", ["b", "S_2"], 0.333),
                ("S_1", ["c", "S_3"], 0.333),

                ("S_2", ["a", "S_1"], 0.333),
                ("S_2", ["b", "S_2"], 0.333),
                ("S_2", ["c", "S_3"], 0.333),

                ("S_3", ["a", "S_1"], 0.333),
                ("S_3", ["b", "S_2"], 0.333),
                ("S_3", ["c", "S_3"], 0.333),
            ],
        )),
        (PCFG.from_weighted_rules(
            start="E",
            rules=[
                ("E", ["-", "E"], 0.5),
                ("E", ["E", "+", "E"], 0.5),
            ],
        ), PCFG.from_weighted_rules(
            start="E",
            rules=[
                ("E", ["E_1"], 0.333),
                ("E", ["E_2"], 0.333),
                ("E", ["E_3"], 0.333),

                # 2 rules ^ 2
                ("E_1", ["-", "E_1"], 0.5),
                ("E_1", ["E_2", "+", "E_3"], 0.5),

                ("E_2", ["-", "E_1"], 0.5),
                ("E_2", ["E_2", "+", "E_3"], 0.5),

                ("E_3", ["-", "E_1"], 0.5),
                ("E_3", ["E_2", "+", "E_3"], 0.5),
            ],
        )),
        (PCFG.from_weighted_rules(
            start="A",
            rules=[
                ("A", ["a"], 0.25),
                ("A", ["B", "C"], 0.25),
                ("A", ["B"], 0.25),
                ("A", ["C"], 0.25),
                ("B", ["B", "B"], 0.5),
                ("B", ["C"], 0.5),
                ("C", ["A"], 1),
            ],
        ), PCFG.from_weighted_rules(
            start="A",
            rules=[
                ("A", ["A_1"], 1),

                ("A_1", ["a"], 0.25),
                ("A_1", ["B_1", "C_1"], 0.25),
                ("A_1", ["B_2"], 0.25),
                ("A_1", ["C_2"], 0.25),

                ("B_1", ["B_3", "B_4"], 0.5),
                ("B_1", ["C_3"], 0.5),
                ("B_2", ["B_3", "B_4"], 0.5),
                ("B_2", ["C_3"], 0.5),
                ("B_3", ["B_3", "B_4"], 0.5),
                ("B_3", ["C_3"], 0.5),
                ("B_4", ["B_3", "B_4"], 0.5),
                ("B_4", ["C_3"], 0.5),

                ("C_1", ["A_1"], 1),
                ("C_2", ["A_1"], 1),
                ("C_3", ["A_1"], 1),
            ],
        )),
    ]
    for g, y in cases:
        out = g.to_bigram()
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_to_bigram")


def test_term():
    cases = [
        (PCFG(
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
            },
            weights="uniform"),
         PCFG(
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
            },
            weights="uniform")),
    ]
    for x, y in cases:
        out = x._term()
        assert out == y, f"Failed test_term: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_term")


def test_bin():
    cases = [
        (PCFG(
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
            weights="uniform",
        ),
            PCFG(
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
            weights="uniform",
        )),
    ]
    for x, y in cases:
        out = x._bin()
        assert out.struct_eq(y), f"Failed test_bin: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_bin")


def test_nullable():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],
                "A": [["a"], PCFG.Empty],
            },
            weights="uniform",
        ), {"A"}),
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],
                "A": [["a"]],
            },
            weights="uniform",
        ), set()),
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],       # nullable
                "A": [["B"], ["C", "a"]],  # nullable
                "B": [["C"]],              # nullable
                "C": [["x"], PCFG.Empty],  # nullable
            },
            weights="uniform",
        ), {"A", "B", "C"}),
    ]
    for x, y in cases:
        out = x.nullables()
        assert out == y, f"Expected {y}, got {out}"
    print(" [+] passed test_nullable")


def test_del():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A", "b", "B"], ["C"]],
                "A": [["a"], PCFG.Empty],
                "B": [["A", "A"], ["A", "C"]],
                "C": [["b"], ["c"]],
            },
            weights="uniform",
         ),
         PCFG(
            start="S",
            rules={
                "S": [["A", "b", "B"], ["C"], ["b", "B"], ["A", "b"], ["b"]],
                "A": [["a"]],
                "B": [["A", "A"], ["A", "C"], ["A"], ["C"]],
                "C": [["b"], ["c"]],
            },
            weights={
                "S": [1, 1, 0.33, 0.33, 0.33],
                "A": [1],
                "B": [0.5, 0.33, 0.83, 0.33],
                "C": [1, 1]
            }
        )),
        (PCFG(
            start="S",
            rules={
                "S": [["A", "s1"], ["A1"], ["A1", "s1"], ["A2", "A1", "s2"]],
                "A": [["A1"]],
                "A1": [["A2"]],
                "A2": [["a2"], PCFG.Empty],
            },
            weights="uniform",
         ),
         PCFG(
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
            weights={
                "S": [0.5, 0.5,
                      1,
                      1,
                      0.25, 0.25, 0.25, 0.25],
                "A": [1],
                "A1": [1],
                "A2": [1],
            }
        )),
    ]
    # FIXME: account for weights
    for x, y in cases:
        out = x._del()
        assert out.struct_eq(y), f"Failed test_del: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_del")


def test_unit():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s", "s", "s"]],
                "A": [["a", "b", "c"], ["e", "f"]],
            },
            weights="uniform",
        ),
            PCFG(
            start="S",
            rules={
                "S": [["a", "b", "c"], ["e", "f"], ["s", "s", "s"]],
            },
            weights="uniform",
        )),
        (PCFG(
            start="S",
            rules={
                "S": [["A", "B"], ["C"]],
                "A": [["a"]],
                "B": [["b"]],
                "C": [["c"]],
            },
            weights="uniform",
        ),
            PCFG(
            start="S",
            rules={
                "S": [["A", "B"], ["c"]],
                "A": [["a"]],
                "B": [["b"]],
            },
            weights="uniform",
        )),
        (PCFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B"]],
                "B": [["C"]],
                "C": [["c"]],
            },
            weights="uniform",
        ),
            PCFG(
            start="S",
            rules={
                "S": [["c"]],
            },
            weights="uniform",
        )),
        (PCFG(
            start='S0',
            rules={
                'S0': [['S']],
                'S': [['A'], ['A', 'B']],
                'A': [['A', 'A']],
                'B': [['B', 'A']],
            },
            weights="uniform",
         ),
         PCFG(
             start='S0',
             rules={
                 'S0': [['A', 'A'], ['A', 'B']],
                 'A': [['A', 'A']],
                 'B': [['B', 'A']],
             },
             weights='uniform',
        )),
    ]
    for x, y in cases:
        out = x._unit()
        assert out.struct_eq(y), \
            f"Failed test_unit: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_unit")


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
            "S": [["A", "B"], PCFG.Empty],
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
            "A": [PCFG.Empty],
            "B": [["b"]],
        }, False),

    ]
    for rules, y in cases:
        g = PCFG("S", rules, "uniform")
        out = g.is_in_CNF()
        if out != y:
            print(f"Failed test_is_in_CNF for {g}: "
                  f"Expected {y}, but got {out}")
            pdb.set_trace()
            g.is_in_CNF()
            exit(1)
    print(" [+] passed test_is_in_CNF")


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
        if y != y_hat:
            print(f"Expected {y}, but got {y_hat} for grammar {g}")
            pdb.set_trace()
            g.is_normalized()
            exit(1)
    print(" [+] passed test_is_normalized")

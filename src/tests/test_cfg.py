import pytest
import math
from cfg import *
from typing import Callable


def test_cfg_check_rep():
    cases = [
        # unused start symbol
        lambda: CFG("S", {"A": ["a"]}),
        # empty rules
        lambda: CFG("S", {"S": []}),
        lambda: CFG("S", {"S": ["A A"], "A": []}),
        # unused nonterminals
        lambda: CFG("S", {"S": ["S"], "A": ["a"]}),
        lambda: CFG("S", {"S": ["A"],
                          "A": ["a"],
                          "B": ["b"]}),
        # duplicate rules
        lambda: CFG("S", {"S": ["A"],
                          "A": ["a", "a"]}),
        lambda: CFG("S", {"S": ["A"],
                          "A": ["a", "a", "a"]}),
        # identity mapping
        lambda: CFG("S", {"S": ["S"]}),
    ]
    for f in cases:
        with pytest.raises(ValueError):
            print(f())


def test_cfg_eq():
    cases = [
        (CFG("S", {"S": ["a", "b", "c"]}),
         CFG("S", {"S": ["c", "b", "a"]}),
         True),
        (CFG("S", {"S": ["a", "b", "c"]}),
         CFG("S", {"S": ["c", "b", "a", "z"]}),
         False),
        (CFG("S", {"S": ["A B", "c"],
                   "A": ["a1", "a2", "a3"],
                   "B": ["b1", "b2", "b3"]}),
         CFG("S", {"S": ["A B", "c"],
                   "A": ["a3", "a2", "a1"],
                   "B": ["b3", "b2", "b1"]}),
         True),
        (CFG("S", {"S": ["A B", "c"],
                   "A": ["a1", "a2", "a3"],
                   "B": ["b1", "b2", "b3"]}),
         CFG("S", {"A": ["a3", "a2", "a1"],
                   "S": ["A B", "c"],
                   "B": ["b3", "b2", "b1"]}),
         True),
    ]
    for a, b, y in cases:
        assert (a == b) == y, f"Expected ({a} == {b}) == {y}, but got {a == b}"


def test_cfg():
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


def test_cfg_iterate_fully():
    cases = [
        (CFG("S", {"S": ["a b c"]}),
         ["a", "b", "c"]),
        (CFG("S", {"S": ["A a"],
                   "A": [""]}),
         ["a"]),
        (CFG("S", {"S": ["A"],
                   "A": ["B"],
                   "B": ["C"],
                   "C": ["c"]}),
         ["c"]),
    ]
    for cfg, sentence in cases:
        out = cfg.iterate_fully()
        assert sentence == out, f"Expected {sentence} but got {out}"


def test_cfg_iterate_until():
    cases = [
        (CFG("S", {"S": ["a b c"]}), 1,
         ["S"]),
        (CFG("S", {"S": ["a b c"]}), 2,
         ["a", "b", "c"]),
        (CFG("S", {"S": ["A"],
                   "A": ["B"],
                   "B": ["C"],
                   "C": ["c"]}), 2,
         ["c"]),
        (CFG("S", {"S": ["a A"],
                   "A": ["a B"],
                   "B": ["a C"],
                   "C": ["c"]}), 2,
         ["a", "A"]),
    ]
    for cfg, k, sentence in cases:
        out = cfg.iterate_until(k)
        assert sentence == out, f"Expected {sentence} but got {out}"


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
        (CFG("E", {"E": ["x"]}),
         CFG("E", {"E": ["x"]})),
        (CFG("E", {"E": ["A", "B"],
                   "A": ["A + A"],
                   "B": ["B * B"]}),
         CFG("E", {"E": ["A_1", "B_1"],
                   "A_1": ["A_2 + A_3"],
                   "A_2": ["A_2 + A_3"],
                   "A_3": ["A_2 + A_3"],
                   "B_1": ["B_2 * B_3"],
                   "B_2": ["B_2 * B_3"],
                   "B_3": ["B_2 * B_3"]})),
        (CFG("E", {"E": ["- X", "P"],
                   "X": ["x"],
                   "P": ["P + P"]}),
         CFG("E", {"E": ["- X_1", "P_1"],
                   "X_1": ["x"],
                   "P_1": ["P_2 + P_3"],
                   "P_2": ["P_2 + P_3"],
                   "P_3": ["P_2 + P_3"]})),
        (CFG("S", {"S": ["E"],
                   "E": ["E + E", "E * E"]}),
         CFG("S", {"S": ["E_1"],
                   "E_1": ["E_2 + E_3", "E_4 * E_5"],
                   "E_2": ["E_2 + E_3", "E_4 * E_5"],
                   "E_3": ["E_2 + E_3", "E_4 * E_5"],
                   "E_4": ["E_2 + E_3", "E_4 * E_5"],
                   "E_5": ["E_2 + E_3", "E_4 * E_5"]})),
        (CFG("E", {"E": ["- E", "E + E", "E * E"]}),
         CFG("E", {"E": ["E_1", "E_2", "E_3", "E_4", "E_5"],
                   "E_1": ["- E_1", "E_2 + E_3", "E_4 * E_5"],
                   "E_2": ["- E_1", "E_2 + E_3", "E_4 * E_5"],
                   "E_3": ["- E_1", "E_2 + E_3", "E_4 * E_5"],
                   "E_4": ["- E_1", "E_2 + E_3", "E_4 * E_5"],
                   "E_5": ["- E_1", "E_2 + E_3", "E_4 * E_5"]})),
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
                "S": [["A", "A"], ["C"]],
                "A": [["B"]],
                "B": [["b"]],
                "C": [["c"]],
            },
        ), CFG(
            start="S",
            rules={
                "S": [["A", "A"], ["c"]],
                "A": [["b"]],
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


def test_to_CNF_parts():
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
        # terminal/nonterminal mix
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
             "A": [["a"], CFG.Empty],
             "B": [["b"]],
         }, False),
        # empty successor for start terminal
        ({
            "S": [CFG.Empty],
        }, True),
        # empty successor is okay if pred is start symbol
        ({
            "S": ["A A", CFG.Empty],
            "A": ["a"],
        }, True),
    ]
    for rules, y in cases:
        g = CFG("S", rules)
        out = g.is_in_CNF()
        assert out == y, \
            f"Failed test_is_in_CNF for {g}: Expected {y}, but got {out}"


def test_to_CNF():
    cases = [
        # identity
        (CFG("S", {"S": ["a"]}),
         CFG("S", {"S": ["a"]})),
        (CFG("S", {"S": [""]}),
         CFG("S", {"S": [""]})),
        (CFG("S", {"S": ["A A", CFG.Empty],
                   "A": ["a"]}),
         CFG("S", {"S": ["A A", CFG.Empty],
                   "A": ["a"]})),
        # start, term, bin
        (CFG("S", {"S": ["a b c"]}),
         CFG("_start_", {"_start_": ["_term_a_ _bin_S_0_1_"],
                         "_bin_S_0_1_": ["_term_b_ _term_c_"],
                         "_term_a_": ["a"],
                         "_term_b_": ["b"],
                         "_term_c_": ["c"],
                         })),
        # start, del
        (CFG("S", {"S": ["A B"],
                   "A": ["a", CFG.Empty],
                   "B": ["b"]}),
         CFG("_start_", {"_start_": ["b", "A B"],
                         "A": ["a"],
                         "B": ["b"]})),
        # start, del, unit
        (CFG("S", {"S": ["A"],
                   "A": ["a", CFG.Empty]}),
         CFG("_start_", {"_start_": ["a"]})),
        # start, unit
        (CFG("S", {"S": ["A"],
                   "A": ["a"]}),
         CFG("_start_", {"_start_": ["a"]})),
    ]
    for a, b in cases:
        out = a.to_CNF()
        assert b == out, f"Expected\n{b}\nbut got\n{out}"


def test_cfg_can_generate():
    cases = [
        (CFG("S", {"S": [CFG.Empty]}),
         [[]],
         [["a"]]),
        (CFG("S", {"S": ["A A", ""],
                   "A": ["a"]}),
         [["a"], ["a", "a"]],
         [["b"]]),
        (CFG("S", {"S": ["S + S", "a", ""]}),
         [["a"], ["a", "+", "a"], ["a", "+"], ["+", "a"], ["+"]],
         [["b"], ["-"]]),
        (CFG("S", {"S": ["a"]}),
         [["a"]], [["b"]]),
        (CFG("S", {"S": ["a", "b", "c"]}),
         [["a"], ["b"], ["c"]],
         [["d"], ["e"]]),
        (CFG("S", {"S": ["A", "B"],
                   "A": ["a"],
                   "B": ["b"]}),
         [["a"], ["b"]],
         [["a", "b"], ["c"]]),
        (CFG("S", {"S": ["AXIOM ; RULES"],
                   "AXIOM": ["F AXIOM", "F"],
                   "RULES": ["RULE , RULES", "RULE"],
                   "RULE": ["A -> B"]}),
         ["F F F ; A -> B".split(" "),
          "F F F ; A -> B , A -> B".split(" ")],
         ["F + F ; A -> B".split(" "),
          "F + F ; A -> BB".split(" "),
          "F + F ; C -> D".split(" ")]),
    ]
    for cfg, in_sentences, out_sentences in cases:
        for sentence in in_sentences:
            out = cfg.can_generate(sentence)
            # print(f"'{''.join(sentence)}'", out)
            assert out, f"Expected {cfg} to generate {sentence}"
        for sentence in out_sentences:
            out = cfg.can_generate(sentence)
            # print(f"'{''.join(sentence)}'", out)
            assert not out, f"Expected {cfg} not to generate {sentence}"


def test_pcfg():
    cfg = CFG("S", {"S": ["A", "B"],
                    "A": ["a"],
                    "B": ["b"]})
    cases = [
        (PCFG.from_CFG(cfg, "uniform", log_mode=False),
         PCFG.from_CFG(cfg, {"S": [0.5, 0.5],
                             "A": [1],
                             "B": [1]},
                       log_mode=False)),
        (PCFG.from_CFG(cfg, "uniform", log_mode=True),
         PCFG.from_CFG(cfg, {"S": T.tensor([0.5, 0.5]).log(),
                             "A": T.tensor([0]),
                             "B": T.tensor([0])},
                       log_mode=True)),
        (PCFG.from_CFG(cfg, "uniform", log_mode=False),
         PCFG.from_weighted_rules("S", [
             ("S", "A", 0.5),
             ("S", "B", 0.5),
             ("A", "a", 1.0),
             ("B", "b", 1.0),
         ])),
        (PCFG.from_CFG(cfg, "uniform", log_mode=False),
         PCFG.from_weighted_rules(
             "S",
             PCFG.from_CFG(cfg, "uniform", log_mode=False).as_weighted_rules()
         )),
    ]
    for a, b in cases:
        assert a == b, f"Expected {a} == {b} = True but got False"


def test_weight():
    pcfg = PCFG.from_CFG(CFG("S", {"S": ["A", "B", "C"],
                                   "A": ["a1", "a2"],
                                   "B": ["b1", "b2"],
                                   "C": ["c1", "c2"]}),
                         weights={"S": [0.1, 0.9, 0],
                                  "A": [0.2, 0.8],
                                  "B": [0.3, 0.7],
                                  "C": [0.4, 0.6]},
                         log_mode=False)
    cases = [
        ("S", ["A"], 0.1),
        ("S", ["B"], 0.9),
        ("S", ["C"], 0),
        ("A", ["a1"], 0.2),
        ("A", ["a2"], 0.8),
        ("B", ["b1"], 0.3),
        ("B", ["b2"], 0.7),
        ("C", ["c1"], 0.4),
        ("C", ["c2"], 0.6),
        # missing weight
        ("S", ["S"], 0),
    ]
    for pred, succ, w in cases:
        out = pcfg.weight(pred, succ)
        assert w == out, f"Expected {w} but got {out} as weight of\n" \
                         f"{pred} -> {succ}\nin\n{pcfg}"


def test_normalized():
    cfg = CFG("S", {"S": ["a", "b", "c"]})
    cases = [
        ([1, 1, 1], [1 / 3, 1 / 3, 1 / 3], 0),
        ([1, 1, 1], [1 / 3, 1 / 3, 1 / 3], 0.1),
        ([1, 2, 3], [1 / 6, 2 / 6, 3 / 6], 0),
        ([1, 2, 3], [1.1 / 6.3, 2.1 / 6.3, 3.1 / 6.3], 0.1),
    ]
    for w1, w2, c in cases:
        w_actual = PCFG.from_CFG(cfg, {"S": w1}).normalized(c=c).weights["S"].tolist()
        assert all(util.approx_eq(x, y) for x, y in zip(w2, w_actual)), \
            f"Expected {w2}, but got {w_actual} for initial weights {w1}"


def test_pcfg_is_normalized():
    cases = [
        (PCFG.from_CFG(CFG("S", {"S": ["a"]}), {"S": [1]}), True),
        (PCFG.from_CFG(CFG("S", {"S": ["a", "b"]}), {"S": [1 / 2, 1 / 2]}), True),
        (PCFG.from_CFG(CFG("S", {"S": ["a", "b"]}), {"S": [1, 1]}), False),
        (PCFG.from_CFG(CFG("S", {"S": ["a"]}), {"S": [0]}, log_mode=True), True),
        (PCFG.from_CFG(CFG("S", {"S": ["a"]}), {"S": [1]}, log_mode=True), False),
        (PCFG.from_CFG(CFG("S", {"S": ["a", "b"]}), {"S": [math.log(1 / 2), math.log(1 / 2)]}, log_mode=True), True),
        (PCFG.from_CFG(CFG("S", {"S": ["a", "b"]}), {"S": [1/2, 1/2]}, log_mode=True), False),
    ]
    for g, y in cases:
        y_hat = g.is_normalized()
        assert y == y_hat, f"Expected {y}, but got {y_hat} for grammar {g}"


def test_pcfg_apply_to_weights():
    cfg = CFG("S", {"S": ["A", "B"]})
    cases = [
        ((lambda x: x + 1), [1, 1], [2, 2]),
        ((lambda x: T.log(x)), [1, 1], [0, 0]),
        ((lambda x: T.log(x)), [0, 0], [-T.inf, -T.inf]),
        ((lambda x: T.exp(x)), [1, 1], [T.e, T.e]),
        ((lambda x: T.exp(x)), [-T.inf, -T.inf], [0, 0]),
    ]
    for f, w0, wf in cases:
        w = list(PCFG.from_CFG(cfg, {"S": w0}).apply_to_weights(f).weights["S"])
        assert wf == w, f"Expected {wf} but got {w}"


def test_pcfg_iterate_until():
    cases = [
        (PCFG("S", {"S": ["A"], "A": ["a"]}),
         [(1, [["S"]])]),
        (PCFG("S", {"S": ["A"], "A": ["A a", "a"]}),
         [(1, [["S"]]),
          (2, ["A a".split(), "a".split()]),
          (3, ["A a a".split(), "a a".split(), "A a".split(), "a".split()]),
          (4, ["A a a a".split(),
               "a a a".split(), "A a a".split(),
               "a a".split(), "A a".split(),
               "a".split()])]),
        (PCFG("S", {"S": ["A", "B"],
                    "A": ["A a", "a"],
                    "B": ["B B", "b"]}),
         [(1, [["S"]]),
          (2, ["A a".split(), "a".split(), "B B".split(), "b".split()]),
          (3, ["A a a".split(), "a a a".split(), "a a".split(), "a".split(),
               "B B B".split(), "B B b".split(), "b B B".split(),
               "b b b".split(), "b b".split(), "b".split()]),
          ]),
    ]
    for _ in range(100):
        for sys, xs in cases:
            for length, strings in xs:
                out = sys.iterate_until(length)
                assert out in strings, f"Expected to get string in {strings}, but got {out}" \
                                       f"for L-system {sys}"


def test_pcfg_iterate_fully():
    cases: List[Tuple[PCFG, Callable]] = [
        (PCFG("S", {"S": ["A"], "A": ["a"]}),
         lambda s: s == ["a"]),
        (PCFG("S", {"S": ["A"], "A": ["A a", "a"]}),
         lambda s: all(x == "a" for x in s[1:]) and s[0] in ["A", "a"]),
        (PCFG("S",
              {"S": ["B"],
               "B": ["B B", "b"]},
              {"S": [1],
               "B": [0.1, 0.9]}),
         lambda s: all(x in ["B", "b"] for x in s)),
        (PCFG("S",
              {"S": ["A", "B"],
               "A": ["A a", "a"],
               "B": ["B B", "b"]},
              {"S": [0.5, 0.5],
               "A": [0.5, 0.5],
               "B": [0.1, 0.9]}),
         lambda s: (all(x == "a" for x in s[1:]) and s[0] in ["A", "a"] or
                    all(x in ["B", "b"] for x in s))),
    ]
    for g, predicate in cases:
        out = g.iterate_fully()
        assert predicate(out), f"{out} did not satisfy predicate for {g}"

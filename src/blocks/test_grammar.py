from blocks.grammar import *


def test_eval():
    tests = [
        # Basic semantics
        (Nil(),
         lambda z: False),
        (Not(Nil()),
         lambda z: True),
        (Times(Z(0), Z(1)),
         lambda z: z[0] * z[1]),
        (If(Lt(Z(0), Z(1)),
            Z(0),
            Z(1)),
         lambda z: min(z[0], z[1])),
        (If(Not(Lt(Z(0),
                   Z(1))),
            Times(Z(0), Z(1)),
            Plus(Z(0), Z(1))),
         lambda z: z[0] * z[1] if not (z[0] < z[1]) else z[0] + z[1]),
        (CornerRect(Num(0), Num(0),
                    Num(1), Num(2)),
         lambda z: util.img_to_tensor(["##__",
                                       "##__",
                                       "##__",
                                       "____"], w=B_W, h=B_H)),
        (LengthLine(Num(2), Num(3), Num(-1), Num(1), Num(3)),
         lambda z: util.img_to_tensor(["_____",
                                       "_____",
                                       "_____",
                                       "__#__",
                                       "_#___",
                                       "#____"], w=B_W, h=B_H)),
        (SizeRect(Num(0), Num(1), Num(2), Num(2)),
         lambda z: util.img_to_tensor(["____",
                                       "##__",
                                       "##__"], w=B_W, h=B_H)),
        (SizeRect(Num(2), Num(2), Num(1), Num(1)),
         lambda z: util.img_to_tensor(["____",
                                       "____",
                                       "__#_",
                                       "____"], w=B_W, h=B_H)),
        (XMax(),
         lambda z: B_W - 1),
        (YMax(),
         lambda z: B_H - 1),
    ]
    for expr, correct_semantics in tests:
        for x in range(10):
            for y in range(10):
                out = expr.eval({"z": [x, y]})
                expected = correct_semantics([x, y])
                t = expr.out_type
                if t in ['int', 'bool']:
                    assert out == expected, f"failed eval test:\n" \
                                            f" expr=\n{expr}\n" \
                                            f" expected=\n{expected}\n" \
                                            f" out=\n{out}"
                elif t == 'bitmap':
                    assert T.equal(out, expected), f"failed eval test:\n" \
                                                   f" expr=\n{expr}\n" \
                                                   f" expected=\n{expected}\n" \
                                                   f" out=\n{out}"
                else:
                    assert False, "type error in eval test"

    # (0,0), (1,1)
    expr = CornerRect(Num(0), Num(0),
                      Num(1), Num(1))
    out = expr.eval({'z': []})
    expected = util.img_to_tensor(["##__",
                                   "##__",
                                   "____",
                                   "____"], w=B_W, h=B_H)
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"

    # (1,0), (3,3)
    expr = CornerRect(Z(0), Num(0),
                      Plus(Z(0), Num(2)), Num(3))
    out = expr.eval({'z': [1, 2, 3]})
    expected = util.img_to_tensor(["_###",
                                   "_###",
                                   "_###",
                                   "_###"], w=B_W, h=B_H)
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"


def test_eval_bitmap():
    tests = [
        # Line tests
        (CornerLine(Num(0), Num(0),
                    Num(1), Num(1)),
         ["#___",
          "_#__",
          "____",
          "____"]),
        (CornerLine(Num(0), Num(0),
                    Num(3), Num(3)),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
        (CornerLine(Num(1), Num(0),
                    Num(3), Num(2)),
         ["_#__",
          "__#_",
          "___#",
          "____"]),
        (CornerLine(Num(1), Num(2),
                    Num(2), Num(3)),
         ["____",
          "____",
          "_#__",
          "__#_"]),
        (CornerLine(Num(1), Num(0),
                    Num(3), Num(0)),
         ["_###",
          "____",
          "____",
          "____"]),
        (CornerLine(Num(1), Num(2),
                    Num(1), Num(3)),
         ["____",
          "____",
          "_#__",
          "_#__"]),
        (LengthLine(Num(0), Num(0), Num(1), Num(1), Num(3)),
         ["#__",
          "_#_",
          "__#"]),
        (LengthLine(Num(0), Num(0), Num(1), Num(1), Num(2)),
         ["#__",
          "_#_",
          "___"]),
        (LengthLine(Num(1), Num(0), Num(0), Num(1), Num(3)),
         ["_#_",
          "_#_",
          "_#_"]),
        (LengthLine(Num(1), Num(0), Num(0), Num(1), Num(5)),
         ["_#_",
          "_#_",
          "_#_",
          "_#_",
          "_#_"]),
        (LengthLine(Num(3), Num(2), Num(1), Num(-1), Num(2)),
         ["_______",
          "____#__",
          "___#___",
          "_______"]),
        (LengthLine(Num(3), Num(2), Num(-1), Num(1), Num(2)),
         ["_______",
          "_______",
          "___#___",
          "__#____"]),
        (LengthLine(Num(3), Num(2), Num(-1), Num(-1), Num(2)),
         ["_______",
          "__#____",
          "___#___",
          "_______"]),

        # Reflection
        (Apply(HFlip(),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["#___" + "_" * (B_W - 8) + "___#",
          "_#__" + "_" * (B_W - 8) + "__#_",
          "__#_" + "_" * (B_W - 8) + "_#__",
          "___#" + "_" * (B_W - 8) + "#___"]),
        (Join(Join(Point(Num(0), Num(0)),
                   Point(Num(1), Num(3))),
              Join(Point(Num(2), Num(0)),
                   Point(Num(3), Num(1)))),
         ["#_#_",
          "___#",
          "____",
          "_#__"]),
        (Apply(VFlip(),
               Join(Join(Point(Num(0), Num(0)),
                         Point(Num(1), Num(3))),
                    Join(Point(Num(2), Num(0)),
                         Point(Num(3), Num(1))))),
         ["#_#_",
          "___#",
          "____",
          "_#__"] +
         ["____"] * (B_H - 8) +
         ["_#__",
          "____",
          "___#",
          "#_#_"]),

        # Joining
        (Join(CornerRect(Num(0), Num(0),
                         Num(1), Num(1)),
              CornerLine(Num(2), Num(3),
                         Num(3), Num(3))),
         ["##__",
          "##__",
          "____",
          "__##"]),
        (Apply(HFlip(),
               Join(CornerRect(Num(0), Num(0),
                               Num(1), Num(1)),
                    CornerRect(Num(2), Num(2),
                               Num(3), Num(3)))),
         ["##__" + "_" * (B_W - 8) + "__##",
          "##__" + "_" * (B_W - 8) + "__##",
          "__##" + "_" * (B_W - 8) + "##__",
          "__##" + "_" * (B_W - 8) + "##__"]),

        # Translate
        (Apply(Translate(Num(0), Num(0)),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
        (Apply(Compose(Translate(Num(1), Num(0)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["12___",
          "_12__",
          "__12_",
          "___12"]),
        (Apply(Compose(Translate(Num(-1), Num(0)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1____",
          "21___",
          "_21__",
          "__21_"]),
        (Apply(Compose(Translate(Num(0), Num(1)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1___",
          "21__",
          "_21_",
          "__21",
          "___2"]),
        (Apply(Compose(Translate(Num(0), Num(-1)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["12___",
          "_12__",
          "__12_",
          "___1_"]),
        (Apply(Compose(Translate(Num(-1), Num(-1)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["2___",
          "_2__",
          "__2_",
          "___1"]),
        (Apply(Compose(Translate(Num(1), Num(1)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1____",
          "_2___",
          "__2__",
          "___2_",
          "____2"]),
        (Apply(Compose(Translate(Num(2), Num(3)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1_____",
          "_1____",
          "__1___",
          "__21__",
          "___2__",
          "____2_",
          "_____2"]),
        (Apply(Repeat(Translate(Num(1), Num(1)), Num(5)),
               Point(Num(0), Num(0))),
         ["#_____",
          "_#____",
          "__#___",
          "___#__",
          "____#_",
          "_____#"]),
        (Apply(Repeat(Translate(Num(2), Num(0)), Num(2)),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["#_#_#___",
          "_#_#_#__",
          "__#_#_#_",
          "___#_#_#"]),
        (Apply(Repeat(Compose(Translate(Num(2), Num(0)),
                              Recolor(Num(2))),
                      Num(2)),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1_2_2___",
          "_1_2_2__",
          "__1_2_2_",
          "___1_2_2"]),
        (CornerRect(Num(0), Num(0), Num(2), Num(2)),
         ["###_",
          "###_",
          "###_",
          "____"]),
        (CornerLine(Num(0), Num(0), Num(3), Num(3)),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
        # (Apply(Intersect(Rect(Num(0), Num(0), Num(2), Num(2))),
        #        Line(Num(0), Num(0), Num(3), Num(3))),
        #  ["#__",
        #   "_#_",
        #   "__#"]),
    ]
    for expr, correct_semantics in tests:
        out = expr.eval({"z": []})
        expected = util.img_to_tensor(correct_semantics, w=B_W, h=B_H)
        assert T.equal(out, expected), \
            f"failed eval test:\n" \
            f" expr=\n{expr}\n" \
            f" expected=\n{expected}\n" \
            f" out=\n{out}"


def test_eval_sprite():
    tests = [
        ([["1___",
           "1___",
           "____",
           "____"]],
         Sprite(0),
         ["1___",
          "1___",
          "____",
          "____"]),
        ([["11",
           "1_"],
          ["11",
           "_1"]],
         Sprite(0, color=Num(4)),
         ["44",
          "4_"]),
        ([["1",
           "1_"],
          ["11",
           "_1"]],
         Sprite(1),
         ["11",
          "_1"]),
        ([["1",
           "1_"],
          ["11",
           "_1"]],
         Sprite(1, x=Num(1), y=Num(2)),
         ["___",
          "___",
          "_11",
          "__1"]),
        ([["111",
           "__1",
           "_1_"]],
         Apply(Compose(HFlip(),
                       Recolor(Num(2))),
               Sprite(0)),
         ["111" + '_' * (B_W - 6) + "222",
          "__1" + '_' * (B_W - 6) + "2__",
          "_1_" + '_' * (B_W - 6) + "_2_"]),
        ([["111",
           "__1",
           "_1_"]],
         Apply(Compose(VFlip(),
                       Recolor(Num(2))),
               Sprite(0)),
         ["111",
          "__1",
          "_1_"]
         +
         ["___"] * (B_H - 6)
         +
         ["_2_",
          "__2",
          "222"]),
    ]
    for sprites, expr, correct_semantics in tests:
        env = {'z': [],
               'sprites': [util.img_to_tensor(s, w=B_W, h=B_H) for s in sprites]}
        out = expr.eval(env)
        expected = util.img_to_tensor(correct_semantics, w=B_W, h=B_H)
        assert T.equal(out, expected), \
            f"failed test:\n" \
            f" expr=\n{expr}\n" \
            f" expected=\n{expected}\n" \
            f" out=\n{out}"


def test_eval_colorsprite():
    tests = [
        ([["12_2",
           "1_35",
           "_45_"]],
         ColorSprite(0),
         ["12_2",
          "1_35",
          "_45_"]),
        ([["21",
           "1_"],
          ["12",
           "_1"]],
         ColorSprite(1),
         ["12",
          "_1"]),
        ([["1",
           "2"],
          ["12",
           "21"]],
         ColorSprite(1, x=Num(1), y=Num(2)),
         ["___",
          "___",
          "_12",
          "_21"]),
    ]
    for sprites, expr, correct_semantics in tests:
        env = {'z': [],
               'sprites': [],
               'color-sprites': [util.img_to_tensor(s, w=B_W, h=B_H) for s in sprites]}
        out = expr.eval(env)
        expected = util.img_to_tensor(correct_semantics, w=B_W, h=B_H)
        assert T.equal(out, expected), \
            f"failed test:\n" \
            f" expr=\n{expr}\n" \
            f" expected=\n{expected}\n" \
            f" out=\n{out}"


def test_eval_color():
    tests = [
        (CornerRect(Num(0), Num(0),
                    Num(1), Num(1), Num(2)),
         ["22__",
          "22__",
          "____",
          "____"]),
        (CornerLine(Num(1), Num(0),
                    Num(3), Num(2), Num(3)),
         ["_3__",
          "__3_",
          "___3",
          "____"]),
        (CornerLine(Num(1), Num(0),
                    Num(3), Num(0), Num(2)),
         ["_222",
          "____",
          "____",
          "____"]),
        (Join(Join(Point(Num(0), Num(0), Num(2)),
                   Point(Num(1), Num(3), Num(3))),
              Join(Point(Num(2), Num(0), Num(7)),
                   Point(Num(3), Num(1), Num(9)))),
         ["2_7_",
          "___9",
          "____",
          "_3__"]),
        (Join(CornerRect(Num(0), Num(0),
                         Num(1), Num(1), Num(1)),
              CornerRect(Num(2), Num(2),
                         Num(3), Num(3), Num(6))),
         ["11__",
          "11__",
          "__66",
          "__66"]),
    ]
    for expr, correct_semantics in tests:
        out = expr.eval({"z": []})
        expected = util.img_to_tensor(correct_semantics, w=B_W, h=B_H)
        assert T.equal(out, expected), \
            f"failed eval color test:\n" \
            f" expr=\n{expr}\n" \
            f" expected=\n{expected}\n" \
            f" out=\n{out}"


def test_zs():
    test_cases = [
        (CornerRect(Num(0), Num(1), Num(2), Num(3)),
         []),
        (CornerRect(Z(0), Z(1), Plus(Z(0), Num(3)), Plus(Z(1), Num(3))),
         [0, 1]),
        (CornerRect(Z(3), Z(1), Z(3), Z(0)),
         [3, 1, 0]),
        (CornerRect(Z(0), Z(1), Z(3), Z(1)),
         [0, 1, 3]),
    ]
    for expr, ans in test_cases:
        out = expr.zs()
        assert out == ans, f"test_zs failed: expected={ans}, actual={out}"


def test_sprites():
    test_cases = [
        (Num(0), []),
        (Sprite(0), [0]),
        (Seq(Sprite(0), Sprite(1), CornerRect(Num(0), Num(0), Num(2), Num(2))),
         [0, 1]),
        (Seq(Sprite(1), Sprite(0), Sprite(2), CornerRect(Num(0), Num(0), Num(2), Num(2))),
         [1, 0, 2]),
    ]
    for expr, ans in test_cases:
        out = expr.sprites()
        assert out == ans, f"test_sprites failed: expected={ans}, actual={out}"


def test_simplify_indices():
    test_cases = [
        (Seq(Z(0), Z(1), Z(3)),
         Seq(Z(0), Z(1), Z(2))),
        (Seq(Z(7), Z(9), Z(3)),
         Seq(Z(0), Z(1), Z(2))),
        (CornerRect(Z(2), Z(1), Plus(Z(0), Z(2)), Plus(Z(1), Z(3))),
         CornerRect(Z(0), Z(1), Plus(Z(2), Z(0)), Plus(Z(1), Z(3)))),
        (Seq(Sprite(1), Sprite(0), Sprite(2), Sprite(0), Sprite(1), Z(3), Z(3)),
         Seq(Sprite(0), Sprite(1), Sprite(2), Sprite(1), Sprite(0), Z(0), Z(0))),
    ]
    for expr, ans in test_cases:
        out = expr.simplify_indices()
        assert out == ans, f"test_simplify_indices failed: expected={ans}, actual={out}"


def test_serialize():
    test_cases = [
        (Nil(), [False]),
        (XMax(), ['x_max']),
        (Plus(Z(0), Z(1)), ['+', 'z_0', 'z_1']),
        (Sprite(0, color=Num(7)), ['S_0', 7, 0, 0]),
        (Plus(Times(Num(1), Num(0)), Minus(Num(3), Num(2))), ['+', '*', 1, 0, '-', 3, 2]),
        (And(Not(Nil()), Nil()), ['&', '~', False, False]),
        (Not(Lt(Num(3), Minus(Num(2), Num(7)))), ['~', '<', 3, '-', 2, 7]),
        (If(Not(Lt(Num(3), Z(0))), Num(2), Num(5)), ['?', '~', '<', 3, 'z_0', 2, 5]),
        (If(Lt(Z(0), Z(1)),
            Point(Z(0), Z(0)),
            CornerRect(Z(1), Z(1), Num(2), Num(3))),
         ['?', '<', 'z_0', 'z_1',
          'P', 1, 'z_0', 'z_0',
          'CR', 1, 'z_1', 'z_1', 2, 3]),
        (Seq(Sprite(0), Sprite(1, color=Num(6)), Sprite(2), Sprite(3)),
         ['{',
          'S_0', 1, 0, 0,
          'S_1', 6, 0, 0,
          'S_2', 1, 0, 0,
          'S_3', 1, 0, 0,
          '}']),
        (Apply(Translate(Num(1), Num(2)),
               Seq(CornerRect(Plus(Z(0), Num(1)),
                              Plus(Z(0), Num(1)),
                              Num(2),
                              Num(2)),
                   CornerRect(Z(0), Z(0), Num(2), Num(2)))),
         ['@', 'T', 1, 2, '{',
          'CR', 1, '+', 'z_0', 1, '+', 'z_0', 1, 2, 2,
          'CR', 1, 'z_0', 'z_0', 2, 2, '}']),
    ]
    for expr, ans in test_cases:
        serialized = expr.serialize()
        deserialized = deserialize(serialized)
        assert serialized == ans, \
            f'serialization failed: in={expr}:\n  expected {ans},\n   but got {serialized}'
        assert deserialized == expr, \
            f'deserialization failed: in={expr}:\n  expected {expr},\n   but got {deserialized}'


def test_deserialize_breaking():
    test_cases = [
        ([1], False),
        ([1, 2, 3], True),
        (['{'], True),
        (['P', 0, 1, 2], False),
        (['P', 'g', 1, 2], True),
        (['P', 'CR', 0, 1, 2, 3, 4, 5, 6], True),
        (['{', 'P', 0, 1, 2, '}'], False),
        (['CL', 1, 1, 3, 3, 2], False),
        (['CL', 'g', 1, 1, 3, 3], True),
        (['CR', 0, 9, 'CR', 11, 6, 8, '}', '}', 4, 2, 8, 15, 9, 9, 7, 13, 4, '}', 2, 8], True),
        (['CL', 'CR', 4, 8, 2, 4, 3, 2, '}', 9, 1, '}', 2, 6, '}', 6, 4, 8], True),
    ]
    for case, should_fail in test_cases:
        try:
            out = deserialize(case)
            failed = False
        except (AssertionError, ValueError):
            failed = True

        if should_fail and not failed:
            print(f"expected to fail but didn't: in={case}, got {out}")
            exit(1)
        elif not should_fail and failed:
            print(f"failed unexpectedly: in={case}")
            exit(1)


def test_well_formed():
    test_cases = [
        (XMax(), True),
        (Point(Num(0), Num(1)), True),
        (Point(0, 1), False),
        (CornerLine(Num(1), Num(1), Num(3), Num(3), Num(1)), True),
    ]
    for expr, ans in test_cases:
        out = expr.well_formed()
        assert out == ans, f'well_formed case failed: in={expr}, expected={ans}, got={out}'


def test_range():
    envs = [
        {'z': [1, 0]},
        {'z': [-3, 3]},
        {'z': [2, 5]},
        {'z': [8, -4]},
    ]
    test_cases = [
        (Num(0), 0, 0),
        (Num(1), 1, 1),
        (Z(0), -3, 8),
        (Z(1), -4, 5),
        (Plus(Z(0), Z(1)), -7, 13),
        (Minus(Z(0), Z(1)), -8, 12),
        (Times(Z(0), Z(1)), -32, 40),
        (Times(XMax(), Z(1)), (B_W - 1) * -4, (B_W - 1) * 5),
    ]
    for expr, lo, hi in test_cases:
        out = expr.range(envs)
        assert out == (lo, hi), f"test_range failed: in={expr}, expected={(lo, hi)}, actual={out}"


def test_leaves():
    cases = [
        (Num(0), [[Num(0)]]),
        (Plus(Num(0), Num(1)), [[Plus, Num(0)], [Plus, Num(1)]]),
        (Times(Num(1), Num(1)), [[Times, Num(1)], [Times, Num(1)]]),
        (Plus(Times(Num(3), Num(2)),
              Minus(Num(3), Num(1))),
         [[Plus, Times, Num(3)],
          [Plus, Times, Num(2)],
          [Plus, Minus, Num(3)],
          [Plus, Minus, Num(1)]]),
    ]
    for expr, ans in cases:
        leaves = expr.leaves()
        n_leaves = expr.count_leaves()
        assert n_leaves == len(ans), f"count_leaves failed: in={expr}, expected={len(ans)}, actual={n_leaves}"
        assert leaves == ans, f"leaves failed: in={expr}, expected={ans}, actual={leaves}"


def test_eval_variable_sizes():
    cases = [
        (CornerRect(Num(0), Num(0), Num(1), Num(1)), 6, 6,
         util.img_to_tensor(["##____",
                             "##____",
                             "______",
                             "______",
                             "______",
                             "______"], h=6, w=6)),
        (CornerRect(Num(0), Num(0), Num(2), Num(2)), 3, 3,
         util.img_to_tensor(["###",
                             "###",
                             "###"], h=3, w=3)),
        # should fail with assertion error:
        # (Rect(Num(0), Num(0), Num(2), Num(2)), 1, 1,
        #  util.img_to_tensor(["###",
        #                      "###",
        #                      "###", ], h=3, w=3))
    ]
    for expr, h, w, ans in cases:
        render = expr.eval(height=h, width=w)
        assert T.equal(render, ans), f"Expected={ans}, but got {render}"


def test_eval_using_xy_max():
    cases = [
        (CornerRect(Num(0), Num(0), XMax(), YMax()), 6, 6,
         util.img_to_tensor(["######",
                             "######",
                             "######",
                             "######",
                             "######",
                             "######"], h=6, w=6)),
        (CornerRect(Num(0), Num(0), XMax(), YMax()), 3, 3,
         util.img_to_tensor(["###",
                             "###",
                             "###"], h=3, w=3)),
        (CornerRect(Num(1), Num(1), XMax(), YMax()), 3, 3,
         util.img_to_tensor(["___",
                             "_##",
                             "_##"], h=3, w=3)),
        (CornerRect(Num(1), Num(1), XMax(), YMax()), 5, 5,
         util.img_to_tensor(["_____",
                             "_####",
                             "_####",
                             "_####",
                             "_####"], h=5, w=5)),
    ]
    for expr, h, w, ans in cases:
        render = expr.eval(height=h, width=w)
        assert T.equal(render, ans), f"Expected={ans}, but got {render}"


def demo_perturb_leaves():
    cases = [
        Num(0),
        Plus(Num(0), Num(1)),
        Times(Num(1), Num(1)),
        Plus(Times(Num(3), Num(2)), Minus(Num(3), Num(1))),
        CornerRect(Num(0), Num(1), Num(3), Num(3)),
    ]
    for expr in cases:
        size = expr.count_leaves()
        out = expr.perturb_leaves(1)
        print(expr, size, out)
        # assert out != expr, f"perturb_leaves failed: in={expr}, out={out}"

# import pdb
import itertools as it
import torch as T
import torch.nn.functional as F
import random
from typing import Type

import blocks.util as util
import blocks.ant as ant

# bitmap size constants
B_W = 16
B_H = 16
SPRITE_MAX_SIZE = 6

LIB_SIZE = 8  # number of z's, sprites
Z_LO = 0  # min poss value in z_n
Z_HI = max(B_W, B_H)  # max poss value in z_n
Z_IGNORE = -1  # ignore z's that have this value
IMG_IGNORE = -1  # ignore pixels that have this value
FULL_LEXICON = ([i for i in range(Z_LO, Z_HI + 1)] +
                [f'z_{i}' for i in range(LIB_SIZE)] +
                [f'S_{i}' for i in range(LIB_SIZE)] +
                [f'CS_{i}' for i in range(LIB_SIZE)] +
                ['x_max', 'y_max',
                 '~', '+', '-', '*', '<', '&', '?',
                 'P', 'L', 'CR', 'SR'
                                 'H', 'V', 'T', '#', 'o', '@', '!', '{', '}', '(', ')'])
OLD_LEXICON = (
        [i for i in range(Z_LO, Z_HI + 1)] +
        [f'z_{i}' for i in range(LIB_SIZE)] +
        [f'S_{i}' for i in range(LIB_SIZE)] +
        ['x_max', 'y_max', 'P', 'L', 'CR', 'SR', '{', '}', '(', ')']
)
SIMPLE_LEXICON = (
        [i for i in range(Z_LO, Z_HI + 1)] +
        [f'z_{i}' for i in range(LIB_SIZE)] +
        [f'S_{i}' for i in range(LIB_SIZE)] +
        [f'CS_{i}' for i in range(LIB_SIZE)] +
        ['x_max', 'y_max', 'P', 'CL', 'LL', 'CR', 'SR', '{', '}', '(', ')']
)


class Visited:
    def accept(self, visitor):
        assert False, f"`accept` not implemented for {type(self).__name__}"


class Expr(Visited):
    def accept(self, visitor):
        assert False, f"not implemented for {type(self).__name__}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __ne__(self, other):
        return str(self) != str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def eval(self, env={}, height=B_H, width=B_W):
        return self.accept(Eval(env, height, width))

    def extract_indices(self, type):
        def f_map(t, *args):
            if t == type:
                assert len(args) == 1, f"Visited {type} but got an unexpected number of args"
                i = args[0]
                return [i]
            else:
                return []

        def f_reduce(type, *children):
            return util.uniq([x for child in children for x in child])

        return self.accept(MapReduce(f_reduce, f_map))

    def zs(self):
        return self.extract_indices(Z)

    def sprites(self):
        return self.extract_indices(Sprite)

    def csprites(self):
        return self.extract_indices(ColorSprite)

    def count_leaves(self):
        def f_map(type, *args): return 1

        def f_reduce(type, *children): return sum(children)

        return self.accept(MapReduce(f_reduce, f_map))

    def leaves(self):
        def f_map(type, *args):
            return [[type(*args)]]

        def f_reduce(type, *children):
            return [[type] + path
                    for child in children
                    for path in child]

        return self.accept(MapReduce(f_reduce, f_map))

    def perturb_leaves(self, p, range=(0, 2)):
        n_perturbed = 0

        # range = self.range(envs=[])
        def perturb(expr):
            nonlocal n_perturbed
            if random.random() < p:
                n_perturbed += 1
                return expr.accept(Perturb(range))
            else:
                return expr

        def perturb_leaf(type, *args):
            return perturb(type(*args))

        def perturb_node(type, *children):
            try:
                return perturb(type(*children))
            except UnimplementedError:
                return type(*children)

        return self.accept(MapReduce(f_map=perturb_leaf, f_reduce=perturb_node))

    def lines(self):
        try:
            return self.bmps
        except AttributeError:
            return []

    def add_line(self, line):
        assert isinstance(self, Seq)
        assert type(line) in [Point, CornerLine, LengthLine, CornerRect, SizeRect, Sprite, ColorSprite]
        return Seq(*self.bmps, line)

    def simplify_indices(self):
        zs = self.zs()
        sprites = self.sprites()
        csprites = self.csprites()
        return self.accept(SimplifyIndices(zs, sprites, csprites))

    def serialize(self):
        return self.accept(Serialize())

    def well_formed(self):
        try:
            return self.accept(WellFormed())
        except (AssertionError, AttributeError):
            return False

    def range(self, envs):
        return self.accept(Range(envs))

    def size(self):
        """Counts both leaves and non-leaf nodes"""

        def f_map(type, *args): return 1 if type != Sprite else 0

        def f_reduce(type, *children): return 1 + sum(children)

        self.accept(MapReduce(f_reduce, f_map))

    def literal_str(self):
        def to_str(type: Type[Expr], *args):
            arg_str = " ".join([str(arg) for arg in args])
            return f'({type.__name__} {arg_str})'

        return self.accept(MapReduce(to_str, to_str))

    def __len__(self):
        return self.size()

    def __str__(self):
        return self.accept(Print())


class Grammar:
    def __init__(self, ops, consts):
        self.ops = ops
        self.consts = consts


def seed_zs(lo=Z_LO, hi=Z_HI, n_zs=LIB_SIZE):
    return (T.rand(n_zs) * (hi - lo) - lo).long()


def seed_sprites(n_sprites=LIB_SIZE, height=B_H, width=B_W):
    width_popn = list(range(2, min(width, SPRITE_MAX_SIZE)))
    height_popn = list(range(2, min(height, SPRITE_MAX_SIZE)))
    return T.stack([ant.make_sprite(w=random.choices(population=width_popn,
                                                     weights=[1 / (1 + w) for w in width_popn],
                                                     k=1)[0],
                                    h=random.choices(population=height_popn,
                                                     weights=[1 / (1 + h) for h in height_popn],
                                                     k=1)[0],
                                    W=width,
                                    H=height)
                    for _ in range(n_sprites)])


def seed_color_sprites(n_sprites=LIB_SIZE, height=B_H, width=B_W):
    width_popn = list(range(2, min(width, SPRITE_MAX_SIZE)))
    height_popn = list(range(2, min(height, SPRITE_MAX_SIZE)))
    return T.stack([ant.make_multicolored_sprite(
        w=random.choices(population=width_popn, weights=[1 / (1 + w) for w in width_popn], k=1)[0],
        h=random.choices(population=height_popn, weights=[1 / (1 + h) for h in height_popn], k=1)[0],
        W=width,
        H=height)
        for _ in range(n_sprites)])


def seed_envs(n_envs):
    # FIXME: add color sprites (make normal sprites then apply colors)
    return [{'z': seed_zs(),
             'sprites': seed_sprites(),
             'color-sprites': seed_color_sprites()
             }
            for _ in range(n_envs)]


# class IllFormedError(Exception): pass
# class IllFormed(Expr):
#     def __init__(self): pass
#     def accept(self, v): raise IllFormedError(f"Visitor {v} tried to visit ill-formed expression.")

class Nil(Expr):
    in_types = []
    out_type = 'bool'

    def __init__(self): pass

    def accept(self, v): return v.visit_Nil()


class Num(Expr):
    in_types = []
    out_type = 'int'

    def __init__(self, n): self.n = n

    def accept(self, v): return v.visit_Num(self.n)


class XMax(Expr):
    in_types = []
    out_type = 'int'

    def __init__(self): pass

    def accept(self, v): return v.visit_XMax()


class YMax(Expr):
    in_types = []
    out_type = 'int'

    def __init__(self): pass

    def accept(self, v): return v.visit_YMax()


class Z(Expr):
    in_types = []
    out_type = 'int'

    def __init__(self, i): self.i = i

    def accept(self, v): return v.visit_Z(self.i)


class Not(Expr):
    in_types = ['bool']
    out_type = 'bool'

    def __init__(self, b): self.b = b

    def accept(self, v): return v.visit_Not(self.b)


class Plus(Expr):
    in_types = ['int', 'int']
    out_type = 'int'

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def accept(self, v): return v.visit_Plus(self.x, self.y)


class Minus(Expr):
    in_types = ['int', 'int']
    out_type = 'int'

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def accept(self, v): return v.visit_Minus(self.x, self.y)


class Times(Expr):
    in_types = ['int', 'int']
    out_type = 'int'

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def accept(self, v): return v.visit_Times(self.x, self.y)


class Lt(Expr):
    in_types = ['bool', 'bool']
    out_type = 'bool'

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def accept(self, v): return v.visit_Lt(self.x, self.y)


class And(Expr):
    in_types = ['bool', 'bool']
    out_type = 'bool'

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def accept(self, v): return v.visit_And(self.x, self.y)


class If(Expr):
    in_types = ['bool', 'int', 'int']
    out_type = 'int'

    def __init__(self, b, x, y):
        self.b = b
        self.x = x
        self.y = y

    def accept(self, v): return v.visit_If(self.b, self.x, self.y)


class CornerLine(Expr):
    """
    A corner-to-corner representation of a line. The line is represented by two corners (x_1, y_1) and (x_2, y_2)
    and includes the corner points and the line between them.  Taken together, the two points must form a
    horizontal, vertical, or diagonal line.
    """
    in_types = ['int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x1, y1, x2, y2, color=Num(1)):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def accept(self, v): return v.visit_CornerLine(self.x1, self.y1, self.x2, self.y2, self.color)


class LengthLine(Expr):
    """
    A corner-direction-length representation of a line. The line is represented by a point (x, y),
    a direction (dx, dy), and a length l.  The resulting line must be horizontal, vertical, or diagonal.
    """
    in_types = ['int', 'int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x, y, dx, dy, length, color=Num(1)):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.length = length
        self.color = color

    def accept(self, v): return v.visit_LengthLine(self.x, self.y, self.dx, self.dy, self.length, self.color)


class Point(Expr):
    in_types = ['int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x, y, color=Num(1)):
        self.x = x
        self.y = y
        self.color = color

    def accept(self, v): return v.visit_Point(self.x, self.y, self.color)


class CornerRect(Expr):
    """
    A rectangle specified by two corners (min and max x- and y-values)
    """
    in_types = ['int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x_min, y_min, x_max, y_max, color=Num(1)):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.color = color

    def accept(self, v): return v.visit_CornerRect(self.x_min, self.y_min, self.x_max, self.y_max, self.color)


class SizeRect(Expr):
    """
    A rectangle specified by a corner, a width, and a height.
    """
    in_types = ['int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x, y, w, h, color=Num(1)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color

    def accept(self, v): return v.visit_SizeRect(self.x, self.y, self.w, self.h, self.color)


class Sprite(Expr):
    in_types = ['int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, i, x=Num(0), y=Num(0), color=Num(1)):
        self.i = i
        self.x = x
        self.y = y
        self.color = color

    def accept(self, v): return v.visit_Sprite(self.i, self.x, self.y, self.color)


class ColorSprite(Expr):
    in_types = ['int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, i, x=Num(0), y=Num(0)):
        self.i = i
        self.x = x
        self.y = y

    def accept(self, v): return v.visit_ColorSprite(self.i, self.x, self.y)


class Seq(Expr):
    in_types = ['list(bitmap)']
    out_type = 'bitmap'

    def __init__(self, *bmps):
        self.bmps = bmps

    def accept(self, v): return v.visit_Seq(self.bmps)


class Join(Expr):
    in_types = ['bitmap', 'bitmap']
    out_type = 'bitmap'

    def __init__(self, bmp1, bmp2):
        self.bmp1 = bmp1
        self.bmp2 = bmp2

    def accept(self, v): return v.visit_Join(self.bmp1, self.bmp2)


class Apply(Expr):
    """Applies a transformation to a bitmap"""
    in_types = ['transform', 'bitmap']
    out_type = 'bitmap'

    def __init__(self, f, bmp):
        self.f = f
        self.bmp = bmp

    def accept(self, v): return v.visit_Apply(self.f, self.bmp)


class Repeat(Expr):
    in_types = ['transform', 'int']
    out_type = 'transform'

    def __init__(self, f, n):
        self.f = f
        self.n = n

    def accept(self, v): return v.visit_Repeat(self.f, self.n)


# class Intersect(Expr):
#     in_types = ['bitmap']
#     out_type = 'transform'
#     def __init__(self, bmp):
#         self.bmp = bmp
#     def accept(self, v): return v.visit_Intersect(self.bmp)

class HFlip(Expr):
    in_types = []
    out_type = 'transform'

    def __init__(self): pass

    def accept(self, v): return v.visit_HFlip()


class VFlip(Expr):
    in_types = []
    out_type = 'transform'

    def __init__(self): pass

    def accept(self, v): return v.visit_VFlip()


class Translate(Expr):
    in_types = ['int', 'int']
    out_type = 'transform'

    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def accept(self, v): return v.visit_Translate(self.dx, self.dy)


class Recolor(Expr):
    in_types = ['int']
    out_type = 'transform'

    def __init__(self, c): self.c = c

    def accept(self, v): return v.visit_Recolor(self.c)


class Compose(Expr):
    in_types = ['transform', 'transform']
    out_type = 'transform'

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def accept(self, v): return v.visit_Compose(self.f, self.g)


class UnimplementedError(Exception): pass


class Visitor:
    def fail(self, s): raise UnimplementedError(f"`visit_{s}` unimplemented for `{type(self).__name__}`")

    def visit_Nil(self): self.fail('Nil')

    def visit_Num(self, n): self.fail('Num')

    def visit_XMax(self): self.fail('XMax')

    def visit_YMax(self): self.fail('YMax')

    def visit_Z(self, i): self.fail('Z')

    def visit_Not(self, b): self.fail('Not')

    def visit_Plus(self, x, y): self.fail('Plus')

    def visit_Minus(self, x, y): self.fail('Minus')

    def visit_Times(self, x, y): self.fail('Times')

    def visit_Lt(self, x, y): self.fail('Lt')

    def visit_And(self, x, y): self.fail('And')

    def visit_If(self, b, x, y): self.fail('If')

    def visit_Point(self, x, y, color): self.fail('Point')

    def visit_CornerLine(self, x1, y1, x2, y2, color): self.fail('Line')

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color): self.fail('CornerRect')

    def visit_SizeRect(self, x, y, w, h, color): self.fail('SizeRect')

    def visit_Sprite(self, i, x, y, color): self.fail('Sprite')

    def visit_ColorSprite(self, i, x, y): self.fail('ColorSprite')

    def visit_Join(self, bmp1, bmp2): self.fail('Join')

    def visit_Seq(self, bmps): self.fail('Seq')

    # def visit_Intersect(self, bmp): self.fail('Intersect')
    def visit_HFlip(self): self.fail('HFlip')

    def visit_VFlip(self): self.fail('VFlip')

    def visit_Translate(self, dx, dy): self.fail('Translate')

    def visit_Recolor(self, c): self.fail('Recolor')

    def visit_Compose(self, f, g): self.fail('Compose')

    def visit_Apply(self, f, bmp): self.fail('Apply')

    def visit_Repeat(self, f, n): self.fail('Repeat')


class EnvironmentError(Exception):
    """
    Use this exception to mark errors in Eval caused by random arguments (Z, Sprites)
    """
    pass


class Eval(Visitor):
    def __init__(self, env, height=B_H, width=B_W):
        self.env = env
        self.height = height
        self.width = width

    def make_bitmap(self, f):
        return T.tensor([[f((x, y))
                          for x in range(self.width)]
                         for y in range(self.height)]).float()

    def make_line(self, ax, ay, bx, by, c):
        if ax == bx:  # vertical
            return self.make_bitmap(lambda p: (ax == p[0] and ay <= p[1] <= by) * c)
        elif ay == by:  # horizontal
            return self.make_bitmap(lambda p: (ax <= p[0] <= bx and ay == p[1]) * c)
        elif abs(bx - ax) == abs(by - ay):  # diagonal
            min_x, max_x = (ax, bx) if ax < bx else (bx, ax)
            min_y, max_y = (ay, by) if ay < by else (by, ay)
            return self.make_bitmap(lambda p: (min_x <= p[0] <= max_x and
                                               min_y <= p[1] <= max_y and
                                               p[1] - ay == (ay - by) // (ax - bx) * (p[0] - ax)) * c)
        assert False, "Line must be vertical, horizontal, or diagonal"

    def overlay(self, *bmps):
        def overlay_pt(p):
            x, y = p
            for bmp in bmps:
                if (c := bmp[y][x]) > 0:
                    return c
            return 0

        return self.make_bitmap(overlay_pt)

    def visit_Nil(self):
        return False

    def visit_Num(self, n):
        return n

    def visit_XMax(self):
        return self.width - 1

    def visit_YMax(self):
        return self.height - 1

    def visit_Z(self, i):
        assert 'z' in self.env, "Eval env missing Z"
        z = self.env['z'][i]
        return z.item() if isinstance(z, T.Tensor) else z

    def visit_Not(self, b):
        b = b.accept(self)
        assert isinstance(b, bool)
        return not b

    def visit_Plus(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return x + y

    def visit_Minus(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return x - y

    def visit_Times(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return x * y

    def visit_Lt(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return x < y

    def visit_And(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, bool) and isinstance(y, bool)
        return x and y

    def visit_If(self, b, x, y):
        b, x, y = b.accept(self), x.accept(self), y.accept(self)
        assert isinstance(b, bool) and isinstance(x, int) and isinstance(y, int)
        return x if b else y

    def visit_Point(self, x, y, color):
        x, y, c = x.accept(self), y.accept(self), color.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return self.make_bitmap(lambda p: (p[0] == x and p[1] == y) * c)

    def visit_CornerLine(self, x1, y1, x2, y2, color):
        c = color.accept(self)
        x1, y1, x2, y2 = (x1.accept(self), y1.accept(self), x2.accept(self), y2.accept(self))
        assert all(isinstance(v, int) for v in [x1, y1, x2, y2])
        assert abs(x2 - x1) >= 1 or abs(y2 - y1) >= 1
        return self.make_line(x1, y1, x2, y2, c)

    def visit_LengthLine(self, x, y, dx, dy, length, color):
        x, y, dx, dy, l, color = (v.accept(self) for v in [x, y, dx, dy, length, color])
        assert all(isinstance(v, int) for v in [x, y, dx, dy, l])
        assert dx in [-1, 0, 1] and dy in [-1, 0, 1] and not (dx == 0 and dy == 0), \
            f'Found unexpected dx, dy=({dx}, {dy})'
        assert l > 0
        points = sorted([(x, y), (x + dx * (l - 1), y + dy * (l - 1))])
        coords = [v for x, y in points for v in [x, y]]
        return self.make_line(*coords, color)

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        c = color.accept(self)
        x_min, y_min, x_max, y_max = (x_min.accept(self), y_min.accept(self),
                                      x_max.accept(self), y_max.accept(self))
        assert all(isinstance(v, int) for v in [x_min, y_min, x_max, y_max])
        assert x_min <= x_max and y_min <= y_max
        return self.make_bitmap(lambda p: (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max) * c)

    def visit_SizeRect(self, x, y, w, h, color):
        x, y, w, h, c = (x.accept(self), y.accept(self), w.accept(self), h.accept(self), color.accept(self))
        assert all(isinstance(v, int) for v in [x, y, w, h])
        assert w > 0 and h > 0
        return self.make_bitmap(lambda p: (x <= p[0] < x + w and y <= p[1] < y + h) * c)

    def visit_Sprite(self, i, x, y, color):
        x, y, c = x.accept(self), y.accept(self), color.accept(self)
        return self.translate(self.env['sprites'][i] * c, x, y)

    def visit_ColorSprite(self, i, x, y):
        x, y = x.accept(self), y.accept(self)
        return self.translate(self.env['color-sprites'][i], x, y)

    def visit_Seq(self, bmps):
        bmps = [bmp.accept(self) for bmp in bmps]
        assert all(isinstance(bmp, T.FloatTensor) for bmp in
                   bmps), f"Seq contains unexpected type: {[type(bmp) for bmp in bmps]}"
        return self.overlay(*bmps)

    def visit_Join(self, bmp1, bmp2):
        bmp1, bmp2 = bmp1.accept(self), bmp2.accept(self)
        assert isinstance(bmp1, T.FloatTensor) and isinstance(bmp2, T.FloatTensor), \
            f"Union needs two float tensors, found bmp1={bmp1}, bmp2={bmp2}"
        return self.overlay(bmp1, bmp2)

    # def visit_Intersect(self, bmp1):
    #     bmp1 = bmp1.accept(self)
    #     assert isinstance(bmp1, T.FloatTensor), f"Intersect needs a float tensor, found bmp={bmp1}"
    #     return lambda bmp2: self.make_bitmap(lambda p: int(bmp1[p[1]][p[0]] > 0 and bmp2[p[1]][p[0]] > 0))

    def visit_HFlip(self):
        return lambda bmp: bmp.flip(1)

    def visit_VFlip(self):
        return lambda bmp: bmp.flip(0)

    @staticmethod
    def translate(bmp, dx, dy):
        assert isinstance(bmp, T.Tensor)
        assert isinstance(dx, int) and isinstance(dy, int)

        def slices(delta):
            if delta == 0:
                return None, None
            elif delta > 0:
                return None, -delta
            else:
                return -delta, None

        a, b = (dx, 0) if dx > 0 else (0, -dx)
        c, d = (dy, 0) if dy > 0 else (0, -dy)
        c_lo, c_hi = slices(dx)
        r_lo, r_hi = slices(dy)

        return F.pad(bmp[r_lo:r_hi, c_lo:c_hi], (a, b, c, d))

    def visit_Translate(self, dx, dy):
        dx, dy = dx.accept(self), dy.accept(self)
        return lambda bmp: self.translate(bmp, dx, dy)

    def visit_Recolor(self, c):
        def index(bmp, p):
            x, y = p
            return bmp[y][x]

        c = c.accept(self)
        return lambda bmp: self.make_bitmap(lambda p: c if index(bmp, p) > 0 else 0)

    def visit_Compose(self, f, g):
        f, g = f.accept(self), g.accept(self)
        return lambda bmp: f(g(bmp))

    def visit_Repeat(self, f, n):
        n, f = n.accept(self), f.accept(self)
        bmps = []

        def g(bmp):
            for i in range(n):
                bmp = f(bmp)
                bmps.append(bmp)
            return self.overlay(*bmps)

        return g

    def visit_Apply(self, f, bmp):
        # FIXME: the semantics here don't play well with intersection
        f, bmp = f.accept(self), bmp.accept(self)
        return self.overlay(f(bmp), bmp)


class Print(Visitor):
    def __init__(self): pass

    def visit_Nil(self): return 'False'

    def visit_Num(self, n): return f'{n}'

    def visit_XMax(self): return 'x_max'

    def visit_YMax(self): return 'y_max'

    def visit_Z(self, i): return f'z_{i}'

    def visit_Not(self, b): return f'(not {b.accept(self)})'

    def visit_Plus(self, x, y): return f'(+ {x.accept(self)} {y.accept(self)})'

    def visit_Minus(self, x, y): return f'(- {x.accept(self)} {y.accept(self)})'

    def visit_Times(self, x, y): return f'(* {x.accept(self)} {y.accept(self)})'

    def visit_Lt(self, x, y): return f'(< {x.accept(self)} {y.accept(self)})'

    def visit_And(self, x, y): return f'(and {x.accept(self)} {y.accept(self)})'

    def visit_If(self, b, x, y): return f'(if {b.accept(self)} {x.accept(self)} {y.accept(self)})'

    def visit_Point(self, x, y, color):
        return f'(Point[{color.accept(self)}] {x.accept(self)} {y.accept(self)})'

    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return f'(CLine[{color.accept(self)}] {x1.accept(self)} {y1.accept(self)} {x2.accept(self)} {y2.accept(self)})'

    def visit_LengthLine(self, x, y, dx, dy, length, color):
        return f'(LLine[{color.accept(self)}] {x.accept(self)} {y.accept(self)} {dx.accept(self)} {dy.accept(self)} {length.accept(self)})'

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return f'(CRect[{color.accept(self)}] {x_min.accept(self)} {y_min.accept(self)} ' \
               f'{x_max.accept(self)} {y_max.accept(self)})'

    def visit_SizeRect(self, x, y, w, h, color):
        return f'(SRect[{color.accept(self)}] {x.accept(self)} {y.accept(self)} ' \
               f'{w.accept(self)} {h.accept(self)})'

    def visit_Sprite(self, i, x, y, color):
        return f'(Sprite_{i}[{color.accept(self)}] {x.accept(self)} {y.accept(self)})'

    def visit_ColorSprite(self, i, x, y):
        return f'(CSprite_{i} {x.accept(self)} {y.accept(self)})'

    def visit_Seq(self, bmps): return '(seq ' + ' '.join([bmp.accept(self) for bmp in bmps]) + ')'

    def visit_Join(self, bmp1, bmp2): return f'(join {bmp1.accept(self)} {bmp2.accept(self)})'

    # def visit_Intersect(self, bmp): return f'(intersect {bmp.accept(self)})'
    def visit_HFlip(self): return 'h-flip'

    def visit_VFlip(self): return 'v-flip'

    def visit_Translate(self, dx, dy): return f'(translate {dx.accept(self)} {dy.accept(self)})'

    def visit_Recolor(self, c): return f'[{c.accept(self)}]'

    def visit_Compose(self, f, g): return f'(compose {f.accept(self)} {g.accept(self)})'

    def visit_Apply(self, f, bmp): return f'({f.accept(self)} {bmp.accept(self)})'

    def visit_Repeat(self, f, n): return f'(repeat {f.accept(self)} {n.accept(self)})'


def deserialize(tokens):
    """
    Deserialize a serialized seq into a program.
    """

    def D(tokens):
        if not tokens: return []
        h, t = tokens[0], D(tokens[1:])
        if isinstance(h, bool) and not h:
            return [Nil()] + t
        if isinstance(h, int):
            return [Num(h)] + t
        if isinstance(h, str):
            if h.startswith('z_'):
                return [Z(int(h[2:]))] + t
            if h.startswith('S_'):
                return [Sprite(int(h[2:]), t[1], t[2], color=t[0])] + t[3:]
            if h.startswith('CS_'):
                return [ColorSprite(int(h[3:]), t[0], t[1])] + t[2:]
            if h == 'x_max':
                return [XMax()] + t
            if h == 'y_max':
                return [YMax()] + t
        if h == '~':
            return [Not(t[0])] + t[1:]
        if h == '+':
            return [Plus(t[0], t[1])] + t[2:]
        if h == '-':
            return [Minus(t[0], t[1])] + t[2:]
        if h == '*':
            return [Times(t[0], t[1])] + t[2:]
        if h == '<':
            return [Lt(t[0], t[1])] + t[2:]
        if h == '&':
            return [And(t[0], t[1])] + t[2:]
        if h == '?':
            return [If(t[0], t[1], t[2])] + t[3:]
        if h == 'P':
            return [Point(t[1], t[2], color=t[0])] + t[3:]
        if h == 'CL':
            return [CornerLine(t[1], t[2], t[3], t[4], color=t[0])] + t[5:]
        if h == 'LL':
            return [LengthLine(t[1], t[2], t[3], t[4], t[5], color=t[0])] + t[6:]
        if h == 'CR':
            return [CornerRect(t[1], t[2], t[3], t[4], color=t[0])] + t[5:]
        if h == 'SR':
            return [SizeRect(t[1], t[2], t[3], t[4], color=t[0])] + t[5:]
        if h == 'H':
            return [HFlip()] + t
        if h == 'V':
            return [VFlip()] + t
        if h == 'T':
            return [Translate(t[0], t[1])] + t[2:]
        if h == '#':
            return [Recolor(t[0])] + t[1:]
        if h == 'o':
            return [Compose(t[0], t[1])] + t[2:]
        if h == '@':
            return [Apply(t[0], t[1])] + t[2:]
        if h == '!':
            return [Repeat(t[0], t[1])] + t[2:]
        # if h == '^':
        #     return [Intersect(t[0])] + t[1:]
        if h == '{':
            i = t.index('}')
            # assert "STOP" in t, f"A sequence must have a STOP token, but none were found: {t}"
            return [Seq(*t[:i])] + t[i + 1:]
        if h == '}':
            return tokens
        else:
            assert False, f'Failed to classify token: {h} of type {type(h)}'

    decoded = D(tokens)
    assert len(decoded) == 1, f'Parsed {len(decoded)} programs in one token sequence, expected one'
    expr = decoded[0]
    assert isinstance(expr, Expr), f'Decoded program should be of type Expr: {expr}'
    assert expr.well_formed(), f'Decoded program should be well-formed: {expr}'
    return expr


class Serialize(Visitor):
    def __init__(self, label_zs=True):
        self.label_zs = label_zs

    def visit_Nil(self): return [False]

    def visit_Num(self, n): return [n]

    def visit_XMax(self): return ['x_max']

    def visit_YMax(self): return ['y_max']

    def visit_Z(self, i): return [f'z_{i}'] if self.label_zs else ['z']

    def visit_Not(self, b): return ['~'] + b.accept(self)

    def visit_Plus(self, x, y): return ['+'] + x.accept(self) + y.accept(self)

    def visit_Minus(self, x, y): return ['-'] + x.accept(self) + y.accept(self)

    def visit_Times(self, x, y): return ['*'] + x.accept(self) + y.accept(self)

    def visit_Lt(self, x, y): return ['<'] + x.accept(self) + y.accept(self)

    def visit_And(self, x, y): return ['&'] + x.accept(self) + y.accept(self)

    def visit_If(self, b, x, y): return ['?'] + b.accept(self) + x.accept(self) + y.accept(self)

    def visit_Point(self, x, y, color): return ['P'] + color.accept(self) + x.accept(self) + y.accept(self)

    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return ['CL'] + color.accept(self) + x1.accept(self) + y1.accept(self) + x2.accept(self) + y2.accept(self)

    def visit_LengthLine(self, x, y, dx, dy, length, color):
        return ['LL'] + color.accept(self) + x.accept(self) + y.accept(self) + dx.accept(self) + dy.accept(
            self) + length.accept(self)

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return ['CR'] + color.accept(self) + x_min.accept(self) + y_min.accept(self) + x_max.accept(
            self) + y_max.accept(self)

    def visit_SizeRect(self, x, y, w, h, color):
        return ['SR'] + color.accept(self) + x.accept(self) + y.accept(self) + w.accept(self) + h.accept(self)

    def visit_Sprite(self, i, x, y, color):
        return [f'S_{i}'] + color.accept(self) + x.accept(self) + y.accept(self)

    def visit_ColorSprite(self, i, x, y):
        return [f'CS_{i}'] + x.accept(self) + y.accept(self)

    def visit_Seq(self, bmps):
        tokens = ['{']  # start
        for bmp in bmps:
            tokens.extend(bmp.accept(self))
        # tokens.append(SEQ_END)
        tokens.append('}')  # stop
        return tokens

    def visit_Join(self, bmp1, bmp2): return [';'] + bmp1.accept(self) + bmp2.accept(self)

    # def visit_Intersect(self, bmp): return ['^'] + bmp.accept(self)
    def visit_HFlip(self): return ['H']

    def visit_VFlip(self): return ['V']

    def visit_Translate(self, x, y): return ['T'] + x.accept(self) + y.accept(self)

    def visit_Recolor(self, c): return ['#'] + c.accept(self)

    def visit_Compose(self, f, g): return ['o'] + f.accept(self) + g.accept(self)

    def visit_Apply(self, f, bmp): return ['@'] + f.accept(self) + bmp.accept(self)

    def visit_Repeat(self, f, n): return ['!'] + f.accept(self) + n.accept(self)


class SimplifyIndices(Visitor):
    def __init__(self, zs, sprites, csprites):
        """
        zs: the indices of zs in the whole expression
        sprites: the indices of sprites in the whole expression
        """
        self.z_mapping = {z: i for i, z in enumerate(zs)}
        self.sprite_mapping = {sprite: i for i, sprite in enumerate(sprites)}
        self.csprite_mapping = {csprite: i for i, csprite in enumerate(csprites)}

    # Base cases
    def visit_Z(self, i):
        return Z(self.z_mapping[i])

    def visit_Sprite(self, i, x, y, color):
        return Sprite(self.sprite_mapping[i], x.accept(self), y.accept(self), color=color.accept(self))

    def visit_ColorSprite(self, i, x, y):
        return ColorSprite(self.csprite_mapping[i], x.accept(self), y.accept(self))

    # Recursive cases
    def visit_Nil(self): return Nil()

    def visit_Num(self, n): return Num(n)

    def visit_XMax(self): return XMax()

    def visit_YMax(self): return YMax()

    def visit_Not(self, b): return Not(b.accept(self))

    def visit_Plus(self, x, y): return Plus(x.accept(self), y.accept(self))

    def visit_Minus(self, x, y): return Minus(x.accept(self), y.accept(self))

    def visit_Times(self, x, y): return Times(x.accept(self), y.accept(self))

    def visit_Lt(self, x, y): return Lt(x.accept(self), y.accept(self))

    def visit_And(self, x, y): return And(x.accept(self), y.accept(self))

    def visit_If(self, b, x, y): return If(b.accept(self), x.accept(self), y.accept(self))

    def visit_Point(self, x, y, color): return Point(x.accept(self), y.accept(self), color.accept(self))

    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return CornerLine(x1.accept(self), y1.accept(self), x2.accept(self), y2.accept(self), color.accept(self))

    def visit_LengthLine(self, x, y, dx, dy, length, color):
        return LengthLine(x.accept(self), y.accept(self), dx.accept(self), dy.accept(self), length.accept(self),
                          color.accept(self))

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return CornerRect(x_min.accept(self), y_min.accept(self), x_max.accept(self), y_max.accept(self),
                          color.accept(self))

    def visit_SizeRect(self, x, y, w, h, color):
        return SizeRect(x.accept(self), y.accept(self), w.accept(self), h.accept(self), color.accept(self))

    def visit_Seq(self, bmps): return Seq(*[bmp.accept(self) for bmp in bmps])

    def visit_Join(self, bmp1, bmp2): return Join(bmp1.accept(self), bmp2.accept(self))

    # def visit_Intersect(self, bmp): return Intersect(bmp.accept(self))
    def visit_HFlip(self): return HFlip()

    def visit_VFlip(self): return VFlip()

    def visit_Translate(self, x, y): return Translate(x.accept(self), y.accept(self))

    def visit_Recolor(self, c): return Recolor(c.accept(self))

    def visit_Compose(self, f, g): return Compose(f.accept(self), g.accept(self))

    def visit_Apply(self, f, bmp): return Apply(f.accept(self), bmp.accept(self))

    def visit_Repeat(self, f, n): return Repeat(f.accept(self), n.accept(self))


class WellFormed(Visitor):
    # TODO: clean up exception handling (unsafe as is)
    def __init__(self): pass

    def visit_Nil(self): return True

    def visit_Num(self, n): return isinstance(n, int)

    def visit_XMax(self): return True

    def visit_YMax(self): return True

    def visit_Z(self, i): return isinstance(i, int)

    def visit_Not(self, b):
        return b.out_type == 'bool' and b.accept(self)

    def visit_Plus(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)

    def visit_Minus(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)

    def visit_Times(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)

    def visit_Lt(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)

    def visit_And(self, x, y):
        return x.out_type == 'bool' and y.out_type == 'bool' and x.accept(self) and y.accept(self)

    def visit_If(self, b, x, y):
        # x, y don't have fixed types, but they should have the same type
        return b.out_type == 'bool' and b.accept(self) and \
            x.out_type == y.out_type and x.accept(self) and y.accept(self)

    def visit_Point(self, x, y, color):
        return x.out_type == 'int' and y.out_type == 'int' and color.out_type == 'int' and \
            x.accept(self) and y.accept(self) and color.accept(self)

    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return all(v.out_type == 'int' and v.accept(self) for v in [x1, y1, x2, y2, color])

    def visit_LengthLine(self, x, y, dx, dy, length, color):
        return all(v.out_type == 'int' and v.accept(self) for v in [x, y, dx, dy, length, color])

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return all(v.out_type == 'int' and v.accept(self) for v in [x_min, y_min, x_max, y_max, color])

    def visit_SizeRect(self, x, y, w, h, color):
        return all(v.out_type == 'int' and v.accept(self) for v in [x, y, w, h, color])

    def visit_Sprite(self, i, x, y, color):
        return isinstance(i, int) and all(v.out_type == 'int' and v.accept(self) for v in [x, y, color])

    def visit_ColorSprite(self, i, x, y):
        return isinstance(i, int) and all(v.out_type == 'int' and v.accept(self) for v in [x, y])

    def visit_Seq(self, bmps): return all(bmp.out_type == 'bitmap' and bmp.accept(self) for bmp in bmps)

    def visit_Join(self, bmp1, bmp2): return all(bmp.out_type == 'bitmap' and bmp.accept(self) for bmp in [bmp1, bmp2])

    # def visit_Intersect(self, bmp):
    #     return bmp.out_type == 'bitmap' and bmp.accept(self)
    def visit_HFlip(self): return True

    def visit_VFlip(self): return True

    def visit_Translate(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)

    def visit_Recolor(self, c):
        return c.out_type == 'int' and c.accept(self)

    def visit_Compose(self, f, g):
        return f.out_type == 'transform' and g.out_type == 'transform' and f.accept(self) and g.accept(self)

    def visit_Apply(self, f, bmp):
        return f.out_type == 'transform' and bmp.out_type == 'bitmap' and f.accept(self) and bmp.accept(self)

    def visit_Repeat(self, f, n):
        return f.out_type == 'transform' and n.out_type == 'int' and f.accept(self) and n.accept(self)


class Perturb(Visitor):
    def __init__(self, range): self.range = range

    def visit_Nil(self): return Not(Nil())

    def visit_Num(self, n): return Num(n + random.choice([-1, 1]) * random.randint(*self.range))

    def visit_Z(self, i): return Z(random.randint(0, LIB_SIZE - 1))

    def visit_HFlip(self): return VFlip()

    def visit_VFlip(self): return HFlip()

    def visit_Plus(self, x, y):
        return Minus(x, y) if random.randint(0, 1) > 0 else Times(x, y)

    def visit_Minus(self, x, y):
        return Times(x, y) if random.randint(0, 1) > 0 else Plus(x, y)

    def visit_Times(self, x, y):
        return Plus(x, y) if random.randint(0, 1) > 0 else Minus(x, y)


class MapReduce(Visitor):
    def __init__(self, f_reduce, f_map):
        self.reduce = f_reduce
        self.f = f_map

    # Map (apply f)
    def visit_Nil(self): return self.f(Nil)

    def visit_Num(self, n): return self.f(Num, n)

    def visit_XMax(self): return self.f(XMax)

    def visit_YMax(self): return self.f(YMax)

    def visit_Z(self, i): return self.f(Z, i)

    def visit_HFlip(self): return self.f(HFlip)

    def visit_VFlip(self): return self.f(VFlip)

    # Map and reduce
    def visit_Sprite(self, i, x, y, color):
        return self.reduce(Sprite, self.f(Sprite, i), x.accept(self), y.accept(self), color.accept(self))

    def visit_ColorSprite(self, i, x, y):
        return self.reduce(ColorSprite, self.f(ColorSprite, i), x.accept(self), y.accept(self))

    # Reduce
    def visit_Not(self, b): return self.reduce(Not, b.accept(self))

    def visit_Plus(self, x, y): return self.reduce(Plus, x.accept(self), y.accept(self))

    def visit_Minus(self, x, y): return self.reduce(Minus, x.accept(self), y.accept(self))

    def visit_Times(self, x, y): return self.reduce(Times, x.accept(self), y.accept(self))

    def visit_Lt(self, x, y): return self.reduce(Lt, x.accept(self), y.accept(self))

    def visit_And(self, x, y): return self.reduce(And, x.accept(self), y.accept(self))

    def visit_If(self, b, x, y): return self.reduce(If, x.accept(self), y.accept(self))

    def visit_Point(self, x, y, color): return self.reduce(Point, x.accept(self), y.accept(self), color.accept(self))

    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return self.reduce(CornerLine, x1.accept(self), y1.accept(self), x2.accept(self), y2.accept(self),
                           color.accept(self))

    def visit_LengthLine(self, x, y, dx, dy, length, color):
        return self.reduce(LengthLine, x.accept(self), y.accept(self), dx.accept(self), dy.accept(self),
                           length.accept(self), color.accept(self))

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return self.reduce(CornerRect, x_min.accept(self), y_min.accept(self), x_max.accept(self), y_max.accept(self),
                           color.accept(self))

    def visit_SizeRect(self, x, y, w, h, color):
        return self.reduce(SizeRect, x.accept(self), y.accept(self), w.accept(self), h.accept(self))

    def visit_Join(self, bmp1, bmp2): return self.reduce(Join, bmp1.accept(self), bmp2.accept(self))

    def visit_Seq(self, bmps): return self.reduce(Seq, *[bmp.accept(self) for bmp in bmps])

    # def visit_Intersect(self, bmp): self.fail('Intersect')
    def visit_Translate(self, dx, dy): return self.reduce(Translate, dx.accept(self), dy.accept(self))

    def visit_Recolor(self, c): return self.reduce(Recolor, c.accept(self))

    def visit_Compose(self, f, g): return self.reduce(Compose, f.accept(self), g.accept(self))

    def visit_Apply(self, f, bmp): return self.reduce(Apply, f.accept(self), bmp.accept(self))

    def visit_Repeat(self, f, n): return self.reduce(Repeat, f.accept(self), n.accept(self))


class Range(Visitor):
    def __init__(self, envs, height=B_H, width=B_W):
        self.envs = envs
        self.height = height
        self.width = width

    def visit_Num(self, n):
        return n, n

    def visit_XMax(self):
        return self.width - 1, self.width - 1

    def visit_YMax(self):
        return self.height - 1, self.height - 1

    def visit_Z(self, i):
        return (min(env['z'][i] for env in self.envs),
                max(env['z'][i] for env in self.envs))

    def visit_Plus(self, x, y):
        x_min, x_max = x.accept(self)
        y_min, y_max = y.accept(self)
        return x_min + y_min, x_max + y_max

    def visit_Minus(self, x, y):
        x_min, x_max = x.accept(self)
        y_min, y_max = y.accept(self)
        return x_min - y_max, x_max - y_min

    def visit_Times(self, x, y):
        x_min, x_max = x.accept(self)
        y_min, y_max = y.accept(self)
        products = [x * y for x in [x_min, x_max] for y in [y_min, y_max]]
        return min(products), max(products)

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        vals = list(it.chain.from_iterable(v.accept(self) for v in [x_min, y_min, x_max, y_max, color]))
        return min(vals), max(vals)

    def visit_SizeRect(self, x, y, w, h, color):
        vals = list(it.chain.from_iterable(v.accept(self) for v in [x, y, w, h, color]))
        return min(vals), max(vals)

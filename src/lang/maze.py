from __future__ import annotations

import pdb
import sys
import math
from typing import List, Optional, Union, Tuple
import numpy as np
import shapely as shp
import shapely.affinity as aff
import einops as ein
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import wandb


def bmp_from_str(str_map: List[str]) -> List[List[Union[int, str]]]:
    """
    Convert a list of strings into a binary float array,
    where any '#' chars are interpreted as 1, any other char
    is interpreted as 0.
    """

    def convert(s: str) -> Union[str, int]:
        if s == "#":
            return 1
        elif s in " _.":
            return 0
        elif s in "grc":
            return s
        else:
            raise ValueError(f"Unexpected cell value: '{s}'")

    return [
        [convert(c) for c in line]
        for line in str_map
    ]


SAVED_MAZES = {
    "cross": bmp_from_str([
        "#####",
        "##g##",
        "#g g#",
        "##r##",
        "#####",
    ]),
    "lehman-ecj-11-hard": bmp_from_str([
        "###################",
        "#g                #",
        "####   ######     #",
        "#  ## ##          #",
        "#   ###           #",
        "###   #  ##       #",
        "# ##  ##  ###     #",
        "#  ## ###   ##    #",
        "#     # ###  ##   #",
        "#     #   ##  ##  #",
        "#     #    ##  ## #",
        "#     #     ##  ###",
        "#                 #",
        "#          ##     #",
        "#         ##      #",
        "#r       ##       #",
        "###################",
    ]),
    "users-guide": bmp_from_str([
        "##########",
        "#r       #",
        "# ###### #",
        "# #      #",
        "# #    # #",
        "# # #### #",
        "# #    # #",
        "# ###  # #",
        "#   #  # #",
        "#  ##g # #",
        "##########",
    ]),
    "big-symm": bmp_from_str([
        "#####################################",
        "# #       #       #     #         #g#",
        "# # ##### # ### ##### ### ### ### # #",
        "#       #   # #     #     # # #   # #",
        "##### # ##### ##### ### # # # ##### #",
        "#   # #       #     # # # # #     # #",
        "# # ####### # # ##### ### # ##### # #",
        "# #       # # #   #     #     #   # #",
        "# ####### ### ### # ### ##### # ### #",
        "#     #   # #   # #   #     # #     #",
        "# ### ### # ### # ##### # # # #######",
        "#   #   # # #   #   #   # # #   #   #",
        "####### # # # ##### # ### # ### ### #",
        "#     # #     #   # #   # #   #     #",
        "# ### # ##### ### # ### ### ####### #",
        "# #   #     #     #   # # #       # #",
        "# # ##### # ### ##### # # ####### # #",
        "# #     # # # # #     #       # #   #",
        "# ##### # # # ### ##### ##### # #####",
        "# #   # # #     #     # #   #       #",
        "# # ### ### ### ##### ### # ##### # #",
        "#r#         #     #       #       # #",
        "#####################################",
    ]),
    "unbounded-10x10": bmp_from_str([
        "          ",
        "          ",
        "          ",
        "          ",
        "    r     ",
        "          ",
        "          ",
        "          ",
        "          ",
        "          ",
    ]),
    "empty-10x10": bmp_from_str([
        "###########",
        "#         #",
        "#         #",
        "#         #",
        "#         #",
        "#    r    #",
        "#         #",
        "#         #",
        "#         #",
        "#         #",
        "#         #",
        "###########",
    ]),
    "empty-20x20": bmp_from_str([
        "####################",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#        r         #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "####################",
    ]),
}


class Maze:

    def __init__(
            self,
            maze_map: List[List[Union[str, int]]],
            scaling: float
    ):
        walls = np.array([
            [
                int(x == 1)
                for x in row
            ]
            for row in maze_map
        ])
        self.scaling = scaling
        self.height = walls.shape[0]
        self.width = walls.shape[1]
        self.scaled_height = self.height * self.scaling
        self.scaled_width = self.width * self.scaling

        self.walls = polygon_from_bitmap(walls, scaling=scaling)
        self.str_map = maze_map

        self.max_dist = (self.height + self.width) * self.scaling

        self.x_center = self.width * self.scaling / 2.
        self.y_center = self.height * self.scaling / 2.

    def start_states_xy(self) -> np.ndarray:
        """Returns the set of possible start positions"""
        starts = []
        for r in range(self.height):
            for c in range(self.width):
                if self.str_map[r][c] == "r":
                    x, y = self.rc_to_xy(r, c)
                    starts.append(np.array([x, y]))
        return np.array(starts)

    @staticmethod
    def from_saved(name: str, scaling=4.) -> "Maze":
        if name not in SAVED_MAZES:
            raise ValueError(
                f"Uknown maze name {name}; must be from "
                f"{list(SAVED_MAZES.keys())}"
            )
        else:
            return Maze(maze_map=SAVED_MAZES[name], scaling=scaling)

    def cardinal_rangefinders(self, p: shp.Point) -> List[shp.LineString]:
        """Construct rangefinders in N, E, S, W directions"""
        r = self.scaled_height + self.scaled_width
        return [
            shp.LineString([(p.x, p.y), (p.x, p.y + r)]),  # up
            shp.LineString([(p.x, p.y), (p.x - r, p.y)]),  # left
            shp.LineString([(p.x, p.y), (p.x, p.y - r)]),  # down
            shp.LineString([(p.x, p.y), (p.x + r, p.y)]),  # right
        ]

    def ordinal_rangefinders(self, p: shp.Point) -> List[shp.LineString]:
        """Construct rangefinders in NE, NW, SW, SE directions"""
        h = self.scaled_height
        w = self.scaled_width
        return [
            shp.LineString([(p.x, p.y), (p.x + w, p.y + h)]),  # top right
            shp.LineString([(p.x, p.y), (p.x - w, p.y + h)]),  # top left
            shp.LineString([(p.x, p.y), (p.x - w, p.y - h)]),  # bottom left
            shp.LineString([(p.x, p.y), (p.x + w, p.y - h)]),  # bottom right
        ]

    def to_oriented(self, rfs: List[shp.LineString], p: shp.Point, theta: float) -> List[shp.LineString]:
        """Rotate and translate rangefinders to be relative to point p with angle theta."""
        return [
            aff.rotate(rf, theta, use_radians=True, origin=p)
            for rf in rfs
        ]

    def rangefinder_dists(self, p: shp.Point, rfs: List[shp.LineString]) -> np.ndarray:
        return np.array(
            intersection_distances(p, self.walls, rfs, max_dist=self.max_dist)
        )

    def xy_to_rc(self, x: float, y: float) -> Tuple[int, int]:
        r = math.floor((self.y_center - y) / self.scaling)
        c = math.floor((x + self.x_center) / self.scaling)
        return r, c

    def rc_to_xy(self, r: float, c: float) -> Tuple[float, float]:
        x = (c + 0.5) * self.scaling - self.x_center
        y = -((r + 0.5) * self.scaling - self.y_center)
        return x, y

    def limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        xmin, ymin = self.rc_to_xy(-0.5, -0.5)
        xmax, ymax = self.rc_to_xy(self.height - 0.5, self.width - 0.5)
        ymin, ymax = ymax, ymin
        return (xmin, xmax), (ymin, ymax)

    def plot_endpoints(self, coords: np.ndarray) -> plt.Figure:
        assert coords.ndim == 2, f"Expected vector of 2D points, got {coords.shape}"

        fig, ax = plt.subplots()
        ax.scatter(coords[:, 0], coords[:, 1])

        # add maze bitmap
        plot_shapes(ax, [self.walls])

        # set limits
        xlim, ylim = self.limits()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.tight_layout()

        return fig

    def plot_trails(self, trails: np.ndarray) -> plt.Figure:
        assert trails.ndim == 3, f"Expected vector of 2D trails, got {trails.shape}"
        assert trails.shape[-1] == 2, f"Expected trail of 2D points, got {trails.shape}"
        b, t, _ = trails.shape
        fig, ax = plt.subplots()

        for trail in trails:
            ax.plot(trail[:, 0], trail[:, 1])

        # add maze bitmap
        plot_shapes(ax, [self.walls])

        # set limits
        xlim, ylim = self.limits()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.tight_layout()

        return fig


def make_square(x: float, y: float, s: float) -> shp.Polygon:
    """Make a square centered at (x, y) with side length s"""
    d = 0.5 * s
    return shp.Polygon(shell=[
        (x - d, y - d),
        (x - d, y + d),
        (x + d, y + d),
        (x + d, y - d),
    ])


def polygon_from_bitmap(bmp: np.ndarray, scaling: float = 1.0) -> shp.Polygon:
    assert bmp.ndim == 2, f"Expected 2D bitmap, but got {bmp.shape}"
    assert bmp.max() in {0, 1}, f"Expected bitmap, but got values in range {bmp.min(), bmp.max()}"
    assert bmp.min() in {0, 1}, f"Expected bitmap, but got values in range {bmp.min(), bmp.max()}"

    height, width = bmp.shape
    x_center = width / 2. * scaling
    y_center = height / 2. * scaling
    squares = []
    for i in range(height):
        for j in range(width):
            if bmp[i][j]:
                x = (j + 0.5) * scaling - x_center
                y = -((i + 0.5) * scaling - y_center)
                squares.append(make_square(x, y, s=scaling))
    return shp.unary_union(squares)


def intersection_distances(
        p: shp.Point,
        maze: shp.Polygon,
        rangefinders: List[shp.LineString],
        max_dist: float,
) -> List[float]:
    dists = []
    for rf in rangefinders:
        xs = rf.intersection(maze)
        if xs.is_empty:
            d = max_dist
        elif (xs.geom_type == "Point" or
              xs.geom_type == "LineString"):
            d = p.distance(xs)
            # plot_shapes(plt.gca(), [xs])
        elif (xs.geom_type == "MultiLineString" or
              xs.geom_type == "GeometryCollection"):
            d = p.distance(xs.geoms[0])
            # plot_shapes(plt.gca(), [xs.geoms[0]])
        else:
            raise ValueError(f"Unexpected geom type {xs.geom_type}")
        dists.append(d)
    return dists


def plot_shapes(ax, shapes: shp.geometry, **kwargs):
    for shape in shapes:
        if isinstance(shape, shp.Polygon):
            plot_polygon(ax, shape, color="gray")
        elif isinstance(shape, shp.LineString):
            ax.plot(*shape.xy)
        elif isinstance(shape, shp.Point):
            ax.plot(shape.x, shape.y, "or")


def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def demo_maze_rangefinders():
    # maze = Maze.from_saved("unbounded-10x10", scaling=4.)
    maze = Maze.from_saved("lehman-ecj-11-hard", scaling=4.)
    # hull = maze.walls.convex_hull
    # interior = hull.difference(maze.walls)
    # p = interior.representative_point()

    for _ in range(5):
        # choose a random point
        x, y = np.random.rand(2) * 60 - 30
        p = shp.Point(x, y)

        # choose a random orientation
        angle = np.random.rand() * 2 * np.pi

        # make oriented ant
        ant = shp.Polygon([(0, 1), (-1, -1), (1, -1)])
        ant = aff.rotate(ant, angle, use_radians=True)
        ant = aff.translate(ant, x, y)

        fig, ax = plt.subplots()
        # make fig have 1:1 aspect ratio
        ax.set_aspect("equal")

        rfs = maze.to_oriented(maze.cardinal_rangefinders(p) + maze.ordinal_rangefinders(p), p, angle)
        # rfs = maze.cardinal_rangefinders(p) + maze.ordinal_rangefinders(p)
        dists = maze.rangefinder_dists(p, rfs)

        # set title to dists with 2 decimal places
        title = "[" + ",".join([f"{d:.0f}" for d in dists]) + "]"
        ax.set_title(title)

        plot_shapes(ax, [maze.walls, ant, *rfs])
        xlim, ylim = maze.limits()
        print(f"xlim={xlim}, ylim={ylim}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.show()
        plt.close()


if __name__ == "__main__":
    demo_maze_rangefinders()

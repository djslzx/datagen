from __future__ import annotations

import pdb
import sys
import math
from typing import List, Optional, Union, Tuple
import numpy as np
import shapely as shp
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
}


class WallCollisionError(Exception):
    pass


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
        self.walls = polygon_from_bitmap(walls, scaling=scaling)
        self.str_map = maze_map

        self.x_center = self.width * self.scaling / 2.
        self.y_center = self.height * self.scaling / 2.

    @staticmethod
    def from_saved(name: str, scaling=4.) -> "Maze":
        if name not in SAVED_MAZES:
            raise ValueError(
                f"Uknown maze name {name}; must be from "
                f"{list(SAVED_MAZES.keys())}"
            )
        else:
            return Maze(maze_map=SAVED_MAZES[name], scaling=scaling)

    def cardinal_rangefinder_lines(self, p: shp.Point) -> List[shp.LineString]:
        h = self.height * self.scaling
        w = self.width * self.scaling
        return [
            shp.LineString([(p.x, p.y), (p.x, p.y + h)]),
            shp.LineString([(p.x, p.y), (p.x, p.y - h)]),
            shp.LineString([(p.x, p.y), (p.x - w, p.y)]),
            shp.LineString([(p.x, p.y), (p.x + w, p.y)]),
        ]

    def cardinal_wall_distances(self, x: float, y: float) -> np.ndarray:
        p = shp.Point(x, y)
        # if not p.within(self.walls):
        #     raise WallCollisionError(
        #         f"Point {p.x, p.y} is within the walls of the maze with distance {p.distance(self.walls)}"
        #     )
        rfs = self.cardinal_rangefinder_lines(p)
        dists = intersection_distances(p, self.walls, rfs)
        return np.array(dists)

    def xy_to_rc(self, x: float, y: float) -> Tuple[int, int]:
        r = math.floor((self.y_center - y) / self.scaling)
        c = math.floor((x + self.x_center) / self.scaling)
        return r, c

    def rc_to_xy(self, r: int, c: int) -> Tuple[float, float]:
        x = (c + 0.5) * self.scaling - self.x_center
        y = -((r + 0.5) * self.scaling - self.y_center)
        return x, y

    def limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        xmin, ymin = self.rc_to_xy(-0.5, -0.5)
        xmax, ymax = self.rc_to_xy(self.height - 0.5, self.width - 0.5)
        ymin, ymax = ymax, ymin
        return (xmin, xmax), (ymin, ymax)

    def plot_endpoints(self, coords: np.ndarray) -> wandb.Image:
        assert coords.ndim == 2, f"Expected vector of 2D points, got {coords.shape}"

        fig, ax = plt.subplots()
        plt.scatter(coords[:, 0], coords[:, 1])

        # add maze bitmap
        plot_shapes(ax, [self.walls])

        # set limits
        xlim, ylim = self.limits()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.tight_layout()

        img = wandb.Image(fig)
        plt.close()
        return img

    def plot_trails(self, trails: np.ndarray) -> wandb.Image:
        assert trails.ndim == 3, f"Expected vector of 2D trails, got {trails.shape}"
        assert trails.shape[-1] == 2, f"Expected trail of 2D points, got {trails.shape}"
        b, t, _ = trails.shape
        
        trails = ein.rearrange(trails, "b t xy -> (b t) xy")
        colors = np.repeat(np.arange(b), t)

        fig, ax = plt.subplots()
        plt.scatter(trails[:, 0], trails[:, 1], s=2, c=colors)

        # add maze bitmap
        plot_shapes(ax, [self.walls])

        # set limits
        xlim, ylim = self.limits()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # plot setup
        plt.tight_layout()

        img = wandb.Image(fig)
        plt.close()
        return img


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
        inf_dist=np.inf,
) -> List[float]:
    dists = []
    for rf in rangefinders:
        overlaps = rf.intersection(maze)
        closest_point = first_point(overlaps)
        if closest_point is None:
            print("WARNING: no overlaps found; ensure that maze has boundaries to avoid this warning.",
                  file=sys.stderr)
            dists.append(inf_dist)
        else:
            dist = p.distance(closest_point)
            dists.append(dist)
    return dists


def first_point(overlaps) -> Optional[shp.Point]:
    if overlaps is None:
        return None
    if isinstance(overlaps, shp.LineString):
        if len(overlaps.coords) > 0:
            return shp.Point(overlaps.coords[0])
        else:
            return None
    return shp.Point(overlaps.geoms[0].coords[0])


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
    maze = Maze.from_saved("lehman-ecj-11-hard", scaling=4.)
    # hull = maze.walls.convex_hull
    # non_walls = hull.difference(maze.walls)
    # ant = non_walls.representative_point()
    # ant = shp.Point(0.8669061676890559, -3.7016983612545915)
    ant = shp.Point(-34.077333476682725, -22.972487448294483)
    # ant = shp.Point(-31.836598332945606, -30.043303065771358)

    fig, ax = plt.subplots()
    plot_shapes(ax, [maze.walls, ant])
    xlim, ylim = maze.limits()
    print(f"xlim={xlim}, ylim={ylim}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()
    plt.close()
    print(maze.cardinal_wall_distances(ant.x, ant.y))


if __name__ == "__main__":
    demo_maze_rangefinders()

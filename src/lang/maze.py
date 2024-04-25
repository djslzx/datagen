from __future__ import annotations

import pdb
import sys
from typing import List, Optional, Union
import numpy as np
import shapely as shp
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


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
        self.height = walls.shape[0] * scaling
        self.width = walls.shape[1] * scaling
        self.walls = polygon_from_bitmap(walls, scaling=scaling)
        self.str_map = maze_map

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
        return [
            shp.LineString([(p.x, p.y), (p.x, p.y + self.height)]),
            shp.LineString([(p.x, p.y), (p.x, p.y - self.height)]),
            shp.LineString([(p.x, p.y), (p.x - self.width, p.y)]),
            shp.LineString([(p.x, p.y), (p.x + self.width, p.y)]),
        ]

    def cardinal_wall_distances(self, x: float, y: float) -> np.ndarray:
        p = shp.Point(x, y)
        assert not p.within(self.walls), \
            f"Point {p.x, p.y} is within the walls of the maze"
        rfs = self.cardinal_rangefinder_lines(p)
        dists = intersection_distances(p, self.walls, rfs)
        return np.array(dists)


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


def plot_shapes(shapes: shp.geometry, **kwargs):
    fig, ax = plt.subplots()
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
    ant = shp.Point(-0.6673714628512808, -3.758217991828773)

    plot_shapes([maze.walls, ant])
    plt.show()
    plt.close()
    print(maze.cardinal_wall_distances(ant.x, ant.y))


if __name__ == "__main__":
    demo_maze_rangefinders()

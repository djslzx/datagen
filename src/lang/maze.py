from __future__ import annotations

import pdb
import sys
from typing import List
import numpy as np
import shapely as shp
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


class Maze:
    def __init__(self, maze_map: List[List[int]]):
        walls = np.array([
            [
                int(x == 1)
                for x in row
            ]
            for row in maze_map
        ])
        self.height, self.width = walls.shape
        self.walls = polygon_from_bitmap(walls)

    @staticmethod
    def bmp_from_str(str_mask: List[str]) -> np.ndarray:
        """
        Convert a list of strings into a binary float array,
        where any '#' chars are interpreted as 1, any other char
        is interpreted as 0.
        """
        return np.array([
            [
                float(c == "#")
                for c in line
            ]
            for line in str_mask
        ])

    def cardinal_wall_distances(self, p: shp.Point) -> np.ndarray:
        assert not p.within(self.walls), \
            f"Point {p.x, p.y} is within the walls of the maze"
        rfs = cardinal_rangefinder_lines(p, width=self.width, height=self.height)
        dists = intersection_distances(p, self.walls, rfs)
        return np.array(dists)


def unit_square(x: int, y: int) -> shp.Polygon:
    return shp.Polygon(shell=[
        (x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y),
    ])


def polygon_from_bitmap(bmp: np.ndarray) -> shp.Polygon:
    assert bmp.ndim == 2, f"Expected 2D bitmap, but got {bmp.shape}"
    assert bmp.max() in {0, 1}, f"Expected bitmap, but got values in range {bmp.min(), bmp.max()}"
    assert bmp.min() in {0, 1}, f"Expected bitmap, but got values in range {bmp.min(), bmp.max()}"

    height, width = bmp.shape
    squares = []
    for i in range(height):
        for j in range(width):
            if bmp[i][j]:
                x = j + 0.5 - width / 2
                y = height / 2 - i + 0.5
                squares.append(unit_square(x, y))
    return shp.unary_union(squares)


def cardinal_rangefinder_lines(p: shp.Point, width: int, height: int) -> List[shp.LineString]:
    return [
        shp.LineString([(p.x, p.y), (p.x, p.y + height)]),
        shp.LineString([(p.x, p.y), (p.x, p.y - height)]),
        shp.LineString([(p.x, p.y), (p.x - width, p.y)]),
        shp.LineString([(p.x, p.y), (p.x + width, p.y)]),
    ]


def intersection_distances(
        p: shp.Point,
        maze: shp.Polygon,
        rangefinders: List[shp.LineString],
        inf_dist=np.inf,
) -> List[float]:
    dists = []
    for rf in rangefinders:
        overlaps = rf.intersection(maze)

        if overlaps is None:
            print("WARNING: no overlaps found; ensure that maze has boundaries to avoid this warning.",
                  file=sys.stderr)
            dists.append(inf_dist)
        else:
            if isinstance(overlaps, shp.LineString):
                if len(overlaps.coords) > 0:
                    closest_point = overlaps.coords[0]
                else:
                    print("WARNING: no overlaps found; ensure that maze has boundaries to avoid this warning.",
                          file=sys.stderr)
                    dists.append(inf_dist)
            else:
                closest_point = overlaps.geoms[0].coords[0]
            dist = p.distance(shp.Point(closest_point))
            dists.append(dist)
    return dists


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
    str_mask = [
        "###################",
        "#                 #",
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
        "#        ##       #",
        "###################",
    ]
    bmp = Maze.bmp_from_str(str_mask)
    # bmp = np.array([
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 0, 0, 0, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 1, 1, 1, 1],
    # ])
    maze = Maze(bmp)
    hull = maze.walls.convex_hull
    non_walls = hull.difference(maze.walls)
    ant = non_walls.representative_point()

    plot_shapes([maze.walls, ant])
    print(maze.cardinal_wall_distances(ant))
    plt.show()
    plt.close()


if __name__ == "__main__":
    demo_maze_rangefinders()

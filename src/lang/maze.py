from __future__ import annotations

import sys
from typing import List
import numpy as np
import shapely as shp
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


def str_to_float_mask(str_mask: List[str]) -> np.ndarray:
    """
    Convert a list of strings into a binary float array,
    where any '#' chars are interpreted as 1, any other char
    is interpreted as 0.
    """
    return np.rot90(np.array([
        [
            float(c == "#")
            for c in line
        ]
        for line in str_mask
    ]), axes=(1, 0))


def unit_square(x: int, y: int) -> shp.Polygon:
    return shp.Polygon(shell=[
        (x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y),
    ])


def polygon_from_bitmap(bmp: np.ndarray) -> shp.Polygon:
    assert bmp.ndim == 2, f"Expected 2D bitmap, but got {bmp.shape}"
    assert bmp.max() in {0, 1}, f"Expected bitmap, but got values in range {bmp.min(), bmp.max()}"
    assert bmp.min() in {0, 1}, f"Expected bitmap, but got values in range {bmp.min(), bmp.max()}"

    squares = [
        unit_square(i, j)
        for i in range(bmp.shape[0])
        for j in range(bmp.shape[1])
        if bmp[i][j]
    ]
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
        print(overlaps)

        if overlaps is None:
            print("WARNING: no overlaps found; ensure that maze has boundaries to avoid this warning.", 
                  file=sys.stderr)
            dists.append(inf_dist)
        else:
            if isinstance(overlaps, shp.LineString):
                closest_point = overlaps.coords[0]
            else:
                closest_point = overlaps.geoms[0].coords[0]
            dist = p.distance(shp.Point(closest_point))
            dists.append(dist)
    return dists


def wall_cardinal_distances(
        x: float, 
        y: float, 
        maze: shp.Polygon, 
        width: int, 
        height: int
) -> np.ndarray:
    p = shp.Point(x, y)
    assert not p.within(maze), f"Point {x, y} is within the walls of the maze"
    rangefinders = cardinal_rangefinder_lines(p, width=width, height=height)
    dists = intersection_distances(p, maze, rangefinders)
    return np.array(dists)


def plot_shapes(shapes: shp.geometry, width: int, height: int, **kwargs):
    fig, ax = plt.subplots()
    for shape in shapes:
        if isinstance(shape, shp.Polygon):
            plot_polygon(ax, shape, color="gray")
        elif isinstance(shape, shp.LineString):
            ax.plot(*shape.xy)
        elif isinstance(shape, shp.Point):
            ax.plot(shape.x, shape.y, "or")
    plt.xlim((0, width))
    plt.ylim((0, height))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


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
    bmp = str_to_float_mask(str_mask)
    # bmp = np.array([
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 0, 0, 0, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 1, 1, 1, 1],
    # ])
    W, H = bmp.shape
    valid_locs = np.array(list(zip(*np.nonzero(1 - bmp)))) + 0.5

    for _ in range(10):
        i = np.random.choice(len(valid_locs))
        ant = shp.Point(*valid_locs[i])
        maze = polygon_from_bitmap(bmp)

        plot_shapes([maze, ant], width=W, height=H)
        print(wall_cardinal_distances(ant.x, ant.y, maze, width=W, height=H))
        plt.show()
        plt.close()


if __name__ == "__main__":
    demo_maze_rangefinders()
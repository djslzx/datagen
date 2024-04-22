from __future__ import annotations

import sys
from typing import List
import numpy as np
import shapely as shp
import matplotlib.pyplot as plt
from util import plot_polygon


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


def cardinal_rangefinder_lines(p: shp.Point) -> List[shp.LineString]:
    return [
        shp.LineString([(p.x, p.y), (p.x, p.y + H)]),
        shp.LineString([(p.x, p.y), (p.x, p.y - H)]),
        shp.LineString([(p.x, p.y), (p.x - W, p.y)]),
        shp.LineString([(p.x, p.y), (p.x + W, p.y)]),
    ]


def intersection_distances(
        p: shp.Point,
        lines: List[shp.LineString],
        maze: shp.Polygon,
        inf_dist=np.inf
) -> List[float]:
    dists = []
    for line in lines:
        overlaps = line.intersection(maze)
        if overlaps is None:
            print("WARNING: no overlaps found; ensure that maze has boundaries to avoid this warning.", file=sys.stderr)
            dists.append(inf_dist)
        else:
            if isinstance(overlaps, shp.LineString):
                closest_point = overlaps.coords[0]
            else:
                closest_point = overlaps.geoms[0].coords[0]
            dist = p.distance(shp.Point(closest_point))
            dists.append(dist)
    return dists


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

i = np.random.choice(len(valid_locs))
ant_loc = valid_locs[i]
ant_orientation = np.pi
ant = shp.Point(*ant_loc)
maze = polygon_from_bitmap(bmp)

# ant rangefinders
rangefinders = cardinal_rangefinder_lines(ant)

dists = intersection_distances(ant, rangefinders, maze)
print(dists)

fig, ax = plt.subplots()
plot_polygon(ax, maze, color="gray")
ax.plot(ant.x, ant.y, "or")
for rf in rangefinders:
    ax.plot(*rf.xy)
plt.xlim((0, W))
plt.ylim((0, H))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

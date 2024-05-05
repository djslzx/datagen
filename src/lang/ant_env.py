from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import shapely as shp

import lang.maze as maze

Observation = namedtuple(
    typename="Observation",
    field_names=[
        "observation",
        "state",
        "ended",
    ]
)


class Environment:
    def reset(self) -> Observation:
        raise NotImplementedError

    def step(self, action_weights: np.ndarray) -> Observation:
        raise NotImplementedError


class AntMaze2D(Environment):
    def __init__(
            self,
            maze_map: maze.Maze,
            step_length: float = 0.1,
    ):
        self.maze_map = maze_map
        self.starts = maze_map.start_states_xy()
        self.pos = AntMaze2D.reset_position(self.starts)
        self.step_length = step_length
        self.d_step = np.array([
            [0., 1.],  # up
            [0., -1.],  # down
            [-1., 0.],  # left
            [1., 0.],  # right
        ]) * step_length

    @staticmethod
    def reset_position(starts: np.ndarray) -> np.ndarray:
        return starts[np.random.randint(len(starts)), :]

    def observe(self, ended: bool):
        return Observation(
            observation=self.maze_map.cardinal_wall_distances(*self.pos),
            state=self.pos,
            ended=ended,
        )

    def reset(self) -> Observation:
        self.pos = self.reset_position(self.starts)
        return self.observe(ended=False)

    def step(self, action_weights: np.ndarray) -> Observation:
        assert action_weights.shape == (4,), f"Expected action vector shape (4,), got {action_weights.shape}"
        d_pos = action_weights @ self.d_step
        new_pos = self.pos + d_pos

        # exit early if we walk into a wall
        if shp.Point(*new_pos).within(self.maze_map.walls):
            return self.observe(ended=True)
        else:
            self.pos = new_pos
            return self.observe(ended=False)

        # # add some noise to the step
        # rand_weights = np.random.rand(4) * self.step_length / 3
        # rand_step = rand_weights @ self.d_step
        # if not shp.Point(*(self.pos + rand_step)).within(self.maze_map.walls):
        #     self.pos += rand_step

    def viz(self):
        fig, ax = plt.subplots()
        maze.plot_shapes(ax, [self.maze_map.walls, shp.Point(*self.pos)])
        plt.show()
        plt.close()


if __name__ == "__main__":
    maze_map = maze.Maze.from_saved("cross")
    env = AntMaze2D(
        maze_map=maze_map,
        step_length=1,
    )

    trails = []
    for _ in range(10):
        trail = []
        o = env.reset()
        for _ in range(100):
            o = env.step(np.random.rand(4))
            trail.append(o.state)
        trails.append(trail)

    trails = np.array(trails)
    maze_map.plot_trails(trails)
    maze_map.plot_endpoints(trails[:, -1, :])
    plt.show()

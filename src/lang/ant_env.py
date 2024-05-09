from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import shapely as shp
import einops as ein

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
        """Reset state to an initial state"""
        raise NotImplementedError

    def step(self, action_weights: np.ndarray) -> Observation:
        """Take a step in the environment using policy defined by `action_weights`"""
        raise NotImplementedError

    def observe(self, ended: bool) -> Observation:
        """Return the externally visible current state"""
        raise NotImplementedError

    @staticmethod
    def reset_position(starts: np.ndarray) -> np.ndarray:
        return starts[np.random.randint(len(starts)), :]

    @property
    def observation_dim(self) -> int:
        raise NotImplementedError(f"Environment {self.__class__.__name__} must implement `observation_dim`")


class AntMaze(Environment):
    def __init__(
            self,
            maze_map: maze.Maze,
            step_length: float = 0.1,
    ):
        self.maze_map = maze_map
        self.starts = maze_map.start_states_xy()
        self.pos = super().reset_position(self.starts)
        self.step_length = step_length
        self.d_step = np.array([
            [0., 1.],  # up
            [0., -1.],  # down
            [-1., 0.],  # left
            [1., 0.],  # right
        ]) * step_length
        self.collision_distance = 0.5
        self.dilated_walls = maze_map.walls.buffer(self.collision_distance).exterior
        self.time_step = 0

    @property
    def observation_dim(self) -> int:
        return 9  # 8 rangefinders, 1 time dimension

    def observe(self, ended: bool) -> Observation:
        p = shp.Point(*self.pos)
        rfs = self.maze_map.cardinal_rangefinders(p) + self.maze_map.ordinal_rangefinders(p)
        rf_dists = self.maze_map.rangefinder_dists(p, rfs)
        obs = np.append(rf_dists, self.time_step)

        return Observation(
            observation=obs,
            state=self.pos,
            ended=ended,
        )

    def reset(self) -> Observation:
        self.pos = self.reset_position(self.starts)
        return self.observe(ended=False)

    def step(self, action_weights: np.ndarray) -> Observation:
        assert action_weights.shape == (4,), f"Expected action vector shape (4,), got {action_weights.shape}"

        d_pos = action_weights @ self.d_step

        # # add some noise to the step
        # rand_weights = np.random.rand(4) * self.step_length / 3
        # rand_step = rand_weights @ self.d_step
        # d_pos += rand_step

        new_pos = self.pos + d_pos

        # exit early if we walk into a wall
        if shp.Point(*new_pos).within(self.maze_map.walls):
            obs = self.observe(ended=True)
            self.time_step += 1
        else:
            self.pos = new_pos
            obs = self.observe(ended=False)
            self.time_step += 1
        return obs

    def viz(self):
        fig, ax = plt.subplots()
        maze.plot_shapes(ax, [self.maze_map.walls, shp.Point(*self.pos)])
        plt.show()
        plt.close()


if __name__ == "__main__":
    maze_map = maze.Maze.from_saved("cross")
    env = AntMaze(
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

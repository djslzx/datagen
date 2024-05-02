from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import shapely as shp
# import gymnasium as gym

import maze

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

    def step(self, action: np.ndarray) -> Observation:
        raise NotImplementedError


# class AntMaze(Environment):
#     def __init__(
#             self,
#             maze_map: maze.Maze,
#             camera_mode: str,
#             save_video=False,
#     ):
#         self.env = gym.make(
#             "AntMaze_UMaze-v4",
#             maze_map=maze_map.str_map,
#             render_mode="rgb_array_list",
#             camera_name="free" if camera_mode == "fixed" else None,
#             use_contact_forces=True,  # required to match ICLR'22 paper
#         )
#         self.save_video = save_video
#         self.maze_map = maze_map
#
#         # edit camera settings
#         if save_video and camera_mode == "fixed":
#             ant_env = self.env.unwrapped.ant_env
#             ant_env.mujoco_renderer.default_cam_config = {
#                 "trackbodyid": 0,
#                 "elevation": -60,
#                 "lookat": np.array([0, 0.0, 0.0]),
#                 "distance": ant_env.model.stat.extent * 1.5,
#                 "azimuth": 0,
#             }
#
#     def reset(self, seed: int):
#         self.env.reset(seed=seed)
#
#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
#         obs, _, terminated, truncated, _ = self.env.step(action)
#         return obs, terminated or truncated


class Ant2DMaze(Environment):
    def __init__(
            self,
            maze_map: maze.Maze,
            step_length: float = 0.1,
    ):
        self.maze_map = maze_map
        self.starts = maze_map.start_states_xy()
        self.pos = Ant2DMaze.reset_position(self.starts)
        self.d_step = np.array([
            [0., 1.],  # up
            [0., -1.],  # down
            [-1., 0.],  # left
            [1., 0.],  # right
        ]) * step_length

    @staticmethod
    def reset_position(starts: np.ndarray) -> np.ndarray:
        return starts[np.random.randint(len(starts)), :]

    def observe(self):
        return Observation(
            observation=self.maze_map.cardinal_wall_distances(*self.pos),
            state=self.pos,
            ended=False,
        )

    def reset(self) -> Observation:
        self.pos = self.reset_position(self.starts)
        return self.observe()

    def step(self, action: np.ndarray) -> Observation:
        assert action.shape == (4,), f"Expected action vector shape (4,), got {action.shape}"
        d_pos = action @ self.d_step
        new_pos = self.pos + d_pos

        # only step if it doesn't make us go into a maze wall
        if not shp.Point(*new_pos).within(self.maze_map.walls):
            self.pos = new_pos

        return self.observe()

    def viz(self):
        fig, ax = plt.subplots()
        maze.plot_shapes(ax, [self.maze_map.walls, shp.Point(*self.pos)])
        plt.show()
        plt.close()


if __name__ == "__main__":
    maze_map = maze.Maze.from_saved("cross")
    env = Ant2DMaze(
        maze_map=maze_map,
        step_length=0.5
    )

    trails = []
    for _ in range(10):
        trail = []
        obs = env.reset()
        for _ in range(100):
            obs = env.step(np.random.rand(4))
            trail.append(obs.state)
        trails.append(trail)

    trails = np.array(trails)
    maze_map.plot_trails(trails)
    maze_map.plot_endpoints(trails[:, -1, :])
    plt.show()

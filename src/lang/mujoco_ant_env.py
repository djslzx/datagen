import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import einops as ein

from lang.maze import Maze
from lang.ant_env import Environment, Observation


class MujocoAntMaze(Environment):
    def __init__(
            self,
            maze_map: Maze,
            camera_mode: str,
            include_orientation=False,
            video_dir="videos",
            primitives_dir="/home/djl328/prob-repl/src/lang/primitives/ant",
            seed=0,
    ):
        assert maze_map.scaling == 4, \
            f"gymnasium AntMaze assumes scale of 4, but got maze with scale={maze_map.scaling}"
        assert camera_mode in {"fixed", "follow"}, f"Camera setting must be either 'fixed' or 'follow'"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.env = gym.make(
            "AntMaze_UMaze-v4",
            maze_map=maze_map.str_map,
            render_mode="rgb_array_list",
            camera_name="free" if camera_mode == "fixed" else None,
            use_contact_forces=True,  # required to match ICLR'22 paper
        )
        self.video_dir = video_dir
        self.maze_map = maze_map
        self.seed = seed
        self.include_orientation = include_orientation

        # load primitives
        self.primitives = [
            torch.load(f"{primitives_dir}/{direction}.pt").pi.to(self.device)
            for direction in ["up", "down", "left", "right"]
        ]

        # store low state observations
        self.low_obs = None

        # edit camera settings
        if camera_mode == "fixed":
            ant_env = self.env.unwrapped.ant_env
            ant_env.mujoco_renderer.default_cam_config = {
                "trackbodyid": 0,
                "elevation": -60,
                "lookat": np.array([0, 0.0, 0.0]),
                "distance": ant_env.model.stat.extent * 1.5,
                "azimuth": 0,
            }

    def reset(self) -> Observation:
        obs, _ = self.env.reset(seed=self.seed)
        return self.observe(obs, ended=False)

    def observe(self, obs: dict, ended: bool) -> Observation:
        self.low_obs = obs["observation"][None, :]
        x, y = obs["achieved_goal"]
        rf_dists = self.maze_map.cardinal_wall_distances(x, y)
        if self.include_orientation:
            orientation = obs["observation"][:5]
            high_obs, _ = ein.pack([rf_dists, orientation], "*")
        else:
            high_obs = rf_dists
        return Observation(
            observation=high_obs,
            state=np.array([x, y]),
            ended=ended,
        )

    def get_action(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, f"Model expects 2D array x, got {x.shape}"
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).to(self.device)
            action, _ = model(x, deterministic=True)
        return action.cpu().numpy()

    def step(self, action_weights: np.ndarray) -> Observation:
        assert self.low_obs is not None, "Must initialize environment using reset() before running step()"

        # compile primitives
        primitive_actions = []
        for pi in self.primitives:
            pi_action = self.get_action(pi, self.low_obs)
            primitive_actions.append(pi_action)
        primitive_actions = ein.rearrange(primitive_actions, "n 1 d -> (n 1) d")

        # take weighted action
        action = action_weights @ primitive_actions
        obs, _, terminated, truncated, _ = self.env.step(action)

        return self.observe(obs, ended=terminated or truncated)

    def render_video(self):
        save_video(
            self.env.render(),
            self.video_dir,
            fps=self.env.metadata["render_fps"],
            step_starting_index=0,
            episode_index=0,
        )

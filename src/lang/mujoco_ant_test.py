from __future__ import annotations
import pdb
from typing import List, Dict, Any, Tuple, Iterable
import numpy as np
import einops as ein
from tqdm import tqdm
import matplotlib.pyplot as plt

import util
from lang.maze import Maze
from lang.ant import FixedDepthAnt, EndFeaturizer
from lang.ant_env import AntMaze
from lang.mujoco_ant_env import MujocoAntMaze


def save_trails(maze: Maze, trails: List[np.ndarray], save_dir: str, name="trail"):
    trails = np.array(trails)
    np.save(f"{save_dir}/{name}.npy", trails, allow_pickle=True)
    maze.plot_trails(trails)
    plt.savefig(f"{save_dir}/{name}.png")
    plt.close()


def setup_save_dir(parent_dir: str) -> str:
    ts = util.timestamp()
    save_dir = f"{parent_dir}/{ts}"
    util.try_mkdir(parent_dir)    
    util.try_mkdir(save_dir)    
    return save_dir


def mujoco_ant_test():
    save_dir = setup_save_dir("../../out/mujoco_tests")
    maze = Maze.from_saved("lehman-ecj-11-hard")
    featurizer = EndFeaturizer()
    mujoco_env = MujocoAntMaze(
        maze_map=maze,
        camera_mode="fixed",
        include_orientation=False,
    )
    mujoco_ant = FixedDepthAnt(
        env=mujoco_env,
        program_depth=6,
        steps=1000,
        featurizer=featurizer,
    )

    simple_env = AntMaze(
        maze_map=maze,
        step_length=0.05,
    )
    simple_ant = FixedDepthAnt(
        env=simple_env,
        program_depth=6,
        steps=1000,
        featurizer=featurizer,
    )

    trees = mujoco_ant.samples(n_samples=2, length_cap=10_000)
    
    with open(f"{save_dir}/programs.txt", "w") as f:
        for tree in trees:
            f.write(mujoco_ant.to_str(tree) + "\n")

    mujoco_trails = []
    simple_trails = []
    for tree in trees:
        mujoco_trail = mujoco_ant.eval(tree, env={'load_bar': True})[:, :2]
        mujoco_env.render_video()
        mujoco_trails.append(mujoco_trail)

        simple_trail = simple_ant.eval(tree, env={'load_bar': True})[:, :2]
        simple_trails.append(simple_trail)

    save_trails(maze, mujoco_trails, save_dir, "mujoco_trails")
    save_trails(maze, simple_trails, save_dir, "simple_trails")


def mujoco_walking_test():
    save_dir = setup_save_dir("../../out/mujoco_walking_test")
    maze = Maze.from_saved("empty-20x20")
    env = MujocoAntMaze(
        maze_map=maze,
        camera_mode="fixed",
        include_orientation=False,
    )
    route = [
        # ([1, 0, 0, 0], 50),
        # ([0, 0, 1, 0], 50),
        # ([0, 1, 0, 0], 50),
        # ([0, 0, 0, 1], 50),
        ([1, 0, 0, 0], 100),
        ([0, 0, 1, 0], 100),
        ([1, 0, 0, 0], 100),
        ([0, 0, 1, 0], 100),
        ([1, 0, 0, 0], 100),
        ([0, 0, 1, 0], 100),
    ]
    obs = env.reset()
    trail = [obs.state]
    for weights, n_steps in route:
        for i in tqdm(range(n_steps), f"route piece {weights, n_steps}"):
            obs = env.step(weights)
            trail.append(obs.state)
            if obs.ended:
                break
        if obs.ended:
            break

    trail = np.array(trail)[None, :]
    save_trails(maze, trail, save_dir, "route")


def mujoco_straight_line_test():
    """
    See how straight of a line each primitive policy walks
    """
    # simple ant policies
    ants = {
        "up":    """(root (conds [0 0 0 0 1])
                          (stmts [1 0 0 0]
                                 [0 0 0 0]))""",
        "down":  """(root (conds [0 0 0 0 1])
                          (stmts [0 -1 0 0]
                                 [0 0 0 0]))""",
        "left":  """(root (conds [0 0 0 0 1])
                          (stmts [0 0 -1 0]
                                 [0 0 0 0]))""",
        "right": """(root (conds [0 0 0 0 1])
                          (stmts [0 0 0 1]
                                 [0 0 0 0]))""",
    }

    save_dir = setup_save_dir("../../out/mujoco_line_tests")
    maze = Maze.from_saved("lehman-ecj-11-hard")
    featurizer = EndFeaturizer()
    env = MujocoAntMaze(
        maze_map=maze,
        camera_mode="fixed",
        include_orientation=False,
    )
    lang = FixedDepthAnt(
        env=env,
        program_depth=2,
        steps=1000,
        featurizer=featurizer,
    )

    for name, ant in ants.items():
        p = lang.parse(ant)
        trail = lang.eval(p, env={'load_bar': True})[None, :, :2]
        save_trails(maze, trail, save_dir, f"{name}_trail")


if __name__ == "__main__":
    # mujoco_ant_test()
    # mujoco_straight_line_test()
    mujoco_walking_test()

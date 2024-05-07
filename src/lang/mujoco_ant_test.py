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
from lang.ant_env import AntMaze2D
from lang.mujoco_ant_env import MujocoAntMaze


def save_trails(maze: Maze, trails: List[np.ndarray], save_dir: str, name="trail"):
    trails = np.array(trails)
    np.save(f"{save_dir}/{name}.npy", trails, allow_pickle=True)
    maze.plot_trails(trails)
    plt.savefig(f"{save_dir}/{name}.png")
    plt.close()


def mujoco_ant_test():
    ts = util.timestamp()
    save_dir = f"../../out/mujoco_tests/{ts}"
    util.try_mkdir(save_dir)

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

    simple_env = AntMaze2D(
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
        mujoco_trail = mujoco_ant.eval(tree, env={'load_bar': True})
        mujoco_env.render_video()
        mujoco_trails.append(mujoco_trail)

        simple_trail = simple_ant.eval(tree, env={'load_bar': True})
        simple_trails.append(simple_trail)

    save_trails(maze, mujoco_trails, save_dir, "mujoco_trails")
    save_trails(maze, simple_trails, save_dir, "simple_trails")


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
    
    # setup
    ts = util.timestamp()
    save_dir = f"../../out/mujoco_line_tests/{ts}"
    util.try_mkdir(save_dir)

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
        trail = lang.eval(p, env={'load_bar': True})[None, :]
        save_trails(maze, trail, save_dir, f"{name}_trail")


if __name__ == "__main__":
    # mujoco_ant_test()
    mujoco_straight_line_test()

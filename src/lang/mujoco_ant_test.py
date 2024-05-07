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

    mujoco_trails = []
    simple_trails = []
    for tree in trees:
        mujoco_trail = mujoco_ant.eval(tree, env={'load_bar': True})
        mujoco_env.render_video()
        mujoco_trails.append(mujoco_trail)

        simple_trail = simple_ant.eval(tree, env={'load_bar': True})
        simple_trails.append(simple_trail)

    mujoco_trails = np.array(mujoco_trails)
    maze.plot_trails(mujoco_trails)
    plt.savefig("mujoco_trails.png")
    plt.close()
    np.save(f"{save_dir}/mujoco_trails.npy", mujoco_trails, allow_pickle=True)

    simple_trails = np.array(simple_trails)
    maze.plot_trails(simple_trails)
    plt.savefig("simple_trails.png")
    plt.close()
    np.save(f"{save_dir}/simple_trails.npy", simple_trails, allow_pickle=True)


def mujoco_straight_line_test():
    """
    See how straight of a line each primitive policy walks
    """
    ants = {
        "north": """(root (conds [1 0 0 0 -2])
                          (stmts [1 0 0 0]
                                 [0 1 0 0]))""",
        "south": """""",
        "east": """""",
        "west": """""",
    }
    raise NotImplementedError


if __name__ == "__main__":
    mujoco_ant_test()


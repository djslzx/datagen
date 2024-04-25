from __future__ import annotations
import pdb
from typing import List, Dict, Any, Tuple
import numpy as np
import einops as ein
from scipy.special import softmax
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from tqdm import tqdm
import matplotlib.pyplot as plt

from lang.tree import Language, Tree, Grammar, ParseError, Featurizer
from lang.maze import Maze
from spinup.algos.pytorch.sac.core import SquashedGaussianMLPActor


class FixedDepthAnt(Language):
    """
    Mujoco ant domain from Programmatic RL without Oracles, with fixed tree depth.
    """
    # grammar with list structure
    grammar = r"""
        root: "(" "ant" "(" "conds" conds ")" \
                        "(" "stmts" stmts ")" ")" -> root
        conds: vec+                               -> conds
        stmts: vec+                               -> stmts
        vec: "[" (NUMBER ","?)* "]"               -> vec

        %import common.NUMBER
        %import common.WS
        %ignore WS
    """

    # grammar with if/else tree structure
    # grammar = r"""
    #     e: "if" "(" b ")" "then" vec "else" "("? e ")"? -> if
    #      | vec                                          -> c
    #     b: NUMBER "+" vec "* X >= 0"                    -> b
    #     vec: "[" (NUMBER ","?)* "]"                     -> vec
    #     %import common.NUMBER
    #     %import common.WS
    #     %ignore WS
    # """

    def __init__(
            self, 
            high_state_dim: int, 
            low_state_dim: int, 
            program_depth: int, 
            maze: Maze,
            steps: int,
            primitives: List[nn.Module],
            featurizer: Featurizer,
            temperature=0.5,
            structure_params=False,
    ):
        assert program_depth > 1
        assert high_state_dim > 0
        assert low_state_dim > 0
        assert 0 <= temperature <= 1
        assert steps > 0

        self.n_conds = program_depth - 1
        self.n_stmts = program_depth
        self.temperature = temperature

        super().__init__(
            parser_grammar=FixedDepthAnt.grammar,
            parser_start="root",
            root_type=None,
            model=None,
            featurizer=featurizer,
        )

        self.high_state_dim = high_state_dim
        self.low_state_dim = low_state_dim
        self.action_dim = len(primitives)
        self.primitives = primitives

        if structure_params:
            raise NotImplementedError("Structured params are not yet implemented")
        self.structure_params = structure_params

        # mujoco env
        self.steps = steps
        assert maze.scaling == 4, \
            f"gymnasium AntMaze assumes scale of 4, but got maze with scale={maze.scaling}"
        self.maze = maze
        self.gym_env = gym.make(
            "AntMaze_UMaze-v4", 
            maze_map=maze.str_map, 
            render_mode="rgb_array_list",
            camera_name="free",
            use_contact_forces=True,  # required to match ICLR'22 paper
        )

        # camera settings
        ant_env = self.gym_env.unwrapped.ant_env
        ant_env.mujoco_renderer.default_cam_config = {
            "trackbodyid": 0,
            "elevation": -60,
            "lookat": np.array([0, 0.0, 0.0]),
            "distance": ant_env.model.stat.extent * 1.5,
            "azimuth": 0,
        }

    def _structured_params(self, t: Tree) -> Tuple[np.ndarray, np.ndarray]:
        assert t.value == "root"
        assert len(t.children) == 2
        conds = t.children[0]
        stmts = t.children[1]

        cond_vec = np.stack([
            np.array([float(gc.value) for gc in c.children])
            for c in conds.children
        ])
        expected_cond_vec_shape = (self.n_conds, self.high_state_dim + 1)
        assert cond_vec.shape == expected_cond_vec_shape, \
            (f"Expected condition params of shape {expected_cond_vec_shape}, "
             f"but got {cond_vec.shape}")

        stmt_vec = np.stack([
            np.array([float(gc.value) for gc in c.children])
            for c in stmts.children
        ])
        expected_stmt_vec_shape = (self.n_stmts, self.action_dim) 
        assert stmt_vec.shape == expected_stmt_vec_shape, \
            (f"Expected statement params of shape {expected_stmt_vec_shape}, "
             f"but got {stmt_vec.shape}")

        return cond_vec, stmt_vec

    def _flat_params(self, t: Tree) -> np.ndarray:
        cond_vec, stmt_vec = FixedDepthAnt._structured_params(self, t)
        return ein.pack([cond_vec, stmt_vec], "*")[0]

    def sample(self) -> Tree:
        raise NotImplementedError

    def fit(self, corpus: List[Tree], alpha):
        raise NotImplementedError

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> Iterable[np.ndarray]:
        assert t.value == "root"

        # extract parameters from program
        cond_params, stmt_params = self._structured_params(t)
        
        # run sim loop
        obs, info = self.gym_env.reset()

        # construct high-level observations for symbolic program
        x, y = obs['achieved_goal']
        yield np.array([x, y])
        dists = self.maze.cardinal_wall_distances(x, y)
       # todo: add more high-level features?
        
        for step in range(self.steps):
            action = self.act_from_params(
                cond_params, 
                stmt_params, 
                high_obs=dists,
                low_obs=obs['observation'],
            )
            obs, _, terminated, truncated, info = self.gym_env.step(action)

            x, y = obs['achieved_goal']
            yield np.array([x, y])
            
            if terminated or truncated:
                break

        save_video(
            self.gym_env.render(),
            "videos",
            fps=self.gym_env.metadata["render_fps"],
            step_starting_index=0,
            episode_index = 0,
        )
        self.gym_env.close()

    @staticmethod
    def get_action(model: nn.Module, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, f"Model expects 2D array x, got {x.shape}"
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action, _ = model(x, deterministic=True)
        return action.numpy()

    def act_from_params(
            self, 
            cond_params: np.ndarray,
            stmt_params: np.ndarray,
            high_obs: np.ndarray,
            low_obs: np.ndarray,
    ) -> np.ndarray:
        assert high_obs.ndim == 1, f"Expected 1D high-level obs array, got {high_obs.shape}"
        assert low_obs.ndim == 1, f"Expected 1D low-level obs array, got {low_obs.shape}"
        assert high_obs.shape[0] == self.high_state_dim, f"Expected high obs dim {self.high_state_dim}, got {high_obs.shape}"
        assert low_obs.shape[0] == self.low_state_dim, f"Expected low obs dim {self.low_state_dim}, got {low_obs.shape}"

        # simulate program to get weights over primitive policies
        action_weights = self.fold_eval_E(cond_params, stmt_params, high_obs)

        # compile primitives
        primitive_actions = []
        for pi in self.primitives:
            pi_action = self.get_action(pi, low_obs[None, :])
            primitive_actions.append(pi_action)
        primitive_actions = ein.rearrange(primitive_actions, "n 1 d -> (n 1) d")

        # action = weighted sum of primitives
        action = action_weights @ primitive_actions
        
        return action

    def extract_features(self, trees: Collection[Tree], n_samples=1, batch_size=4, load_bar=False) -> np.ndarray:
        raise NotImplementedError("Ant should handle feature extraction differently than exec-once langs")

    def fold_eval_E(
            self, 
            cond_params: np.ndarray,
            stmt_params: np.ndarray,
            state: np.ndarray,
    ) -> np.ndarray:
        """Evaluate E by folding, no recursion"""
        e = stmt_params[-1]
        for b, c in zip(cond_params[::-1], stmt_params[:-1][::-1]):
            w = softmax(b[-1] + np.dot(b[:-1], state))
            e = w * c + (1 - w) * e
        return e

    @property
    def str_semantics(self) -> Dict:
        return {
            "root": lambda conds, stmts: f"(root (conds {conds}) (stmts {stmts}))",
            "conds": lambda *conds: " ".join(conds),
            "stmts": lambda *stmts: " ".join(stmts),
            "vec": lambda *xs: "[" + (" ".join(xs)) + "]",
        }


class MujocoAntFeaturizer(Featurizer):
    """
    Ant features: center of gravity along trajectory of ant, i.e. body pos over time?
    """
    def __init__(self):
        pass

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        if isinstance(batch, list):
            batch = np.stack(batch)
        assert batch.ndim == 2, f"Expected 2D batch, got {batch.shape}"
        return batch


if __name__ == "__main__":
    np.random.seed(0)
    STEPS = 1000
    primitives_dir = "/home/djl328/prob-repl/src/lang/primitives/ant/"
    primitives = [
        torch.load(f"{primitives_dir}/{direction}.pt").pi
        for direction in ["up", "down", "left", "right"]
    ]
    featurizer = MujocoAntFeaturizer()
    maze = Maze.from_saved("lehman-ecj-11-hard")
    lang = FixedDepthAnt(
        maze=maze,
        primitives=primitives,
        high_state_dim=4,
        low_state_dim=111,
        program_depth=2,
        steps=STEPS,
        featurizer=featurizer,
    )
    s = f"""
        (ant (conds [{' '.join([str(i) for i in range(4 + 1)])}]) 
             (stmts [1. 0. 0. 0.] 
                    [0. 1. 0. 0.]))
    """
    tree = lang.parse(s)
    print(tree, lang.to_str(tree), sep='\n')
    print(lang._structured_params(tree))
    print(lang._flat_params(tree))
    actions = lang.eval(tree)
    pos = []
    for action in tqdm(actions, total=lang.steps):
        feat_vec = featurizer.apply([action])
        pos.append(feat_vec)
    print(pos)


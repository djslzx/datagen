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
            env_dim: int, 
            depth: int, 
            maze_map: List[List[int]],
            steps: int,
            primitives: List[nn.Module],
            featurizer: Featurizer,
            temperature=0.5,
            structure_params=False,
    ):
        assert depth > 1
        assert env_dim > 0
        assert 0 <= temperature <= 1
        assert steps > 0
        assert all(cell == 0 or cell == 1
                   for row in maze_map 
                   for cell in row), \
            f"Expected binary map but got {maze_map}"

        self.n_conds = depth - 1
        self.n_stmts = depth
        self.temperature = temperature

        super().__init__(
            parser_grammar=FixedDepthAnt.grammar,
            parser_start="root",
            root_type=None,
            model=None,
            featurizer=featurizer,
        )

        self.env_dim = env_dim
        self.n_primitives = len(primitives)
        self.primitives = primitives

        if structure_params:
            raise NotImplementedError("Structured params are not yet implemented")
        self.structure_params = structure_params

        # mujoco env
        self.maze_map = maze_map
        self.steps = steps
        self.gym_env = gym.make(
            "AntMaze_UMaze-v4", 
            maze_map=maze_map, 
            render_mode="rgb_array_list",
            use_contact_forces=True,  # required to get enough observations to match ICLR'22 paper
        )

    def _structured_params(self, t: Tree) -> Tuple[np.ndarray, np.ndarray]:
        assert t.value == "root"
        assert len(t.children) == 2
        conds = t.children[0]
        stmts = t.children[1]

        cond_vec = np.stack([
            np.array([float(gc.value) for gc in c.children])
            for c in conds.children
        ])
        assert cond_vec.shape == (self.n_conds, self.env_dim + 1), \
            (f"Expected condition params of shape {(self.n_conds, self.env_dim + 1)}, "
             f"but got {cond_vec.shape}")

        stmt_vec = np.stack([
            np.array([float(gc.value) for gc in c.children])
            for c in stmts.children
        ])
        assert stmt_vec.shape == (self.n_stmts, self.n_primitives), \
            (f"Expected statement params of shape {(self.n_stmts, self.n_primitives)}, "
             f"but got {stmt_vec.shape}")

        return cond_vec, stmt_vec

    def _flat_params(self, t: Tree) -> np.ndarray:
        cond_vec, stmt_vec = FixedDepthAnt._structured_params(self, t)
        return ein.pack([cond_vec, stmt_vec], "*")[0]

    def sample(self) -> Tree:
        raise NotImplementedError

    def fit(self, corpus: List[Tree], alpha):
        raise NotImplementedError

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> np.ndarray:
        assert t.value == "root"
        assert not self.structure_params
        assert "state" in env, "Ant must evaluate on an env state"

        obs = env["state"]
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        assert obs.shape[0] == self.env_dim

        # extract parameters and act
        cond_params, stmt_params = self._structured_params(t)
        
        obs, info = self.gym_env.reset()
        step_start = 0
        episode = 0
        for step in tqdm(range(self.steps)):
            action = self.act_from_params(cond_params, stmt_params, obs['observation'])
            obs, reward, terminated, truncated, info = self.gym_env.step(action)
            
            if terminated or truncated:
                save_video(
                    self.gym_env.render(),
                    "videos",
                    fps=self.gym_env.metadata["render_fps"],
                    step_starting_index=step_start,
                    episode_index = episode,
                )
                step_start = step + 1
                episode += 1
                observation, info = self.gym_env.reset()
        
        save_video(
            self.gym_env.render(),
            "videos",
            fps=self.gym_env.metadata["render_fps"],
            step_starting_index=step_start,
            episode_index = episode,
        )
        self.gym_env.close()

        return action

    @staticmethod
    def get_action(model: nn.Module, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, f"Model expects 2D array x, got {x.shape}"
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action, _ = model(x, deterministic=True)
        return action.numpy()

    def act_from_params(self, cond_params: np.ndarray, stmt_params: np.ndarray, obs: np.ndarray) -> np.ndarray:
        assert not self.structure_params
        assert obs.ndim == 1, f"Expected 1D obs array, got {obs.shape}"
        assert obs.shape[0] == self.env_dim

        # simulate program to get weights over primitive policies
        action_weights = self.fold_eval_E(cond_params, stmt_params, obs)

        # compile primitives
        primitive_actions = []
        for pi in self.primitives:
            pi_action = self.get_action(pi, obs[None, :])
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
            env_state: np.ndarray,
    ) -> np.ndarray:
        """Evaluate E by folding, no recursion"""
        e = stmt_params[-1]
        for b, c in zip(cond_params[::-1], stmt_params[:-1][::-1]):
            w = softmax(b[-1] + np.dot(b[:-1], env_state))
            # todo: gumbel softmax for c
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
        # # mujoco env
        # self.maze_map = maze_map
        # self.steps = steps
        # self.gym_env = gym.make(
        #     "AntMaze_UMaze-v4", 
        #     maze_map=maze_map, 
        #     render_mode="rgb_array_list",
        #     use_contact_forces=True,  # required to get enough observations to match ICLR'22 paper
        # )
        pass

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        return batch


if __name__ == "__main__":
    np.random.seed(0)
    primitives_dir = "/home/djl328/prob-repl/src/lang/primitives/ant/"
    primitives = [
        torch.load(f"{primitives_dir}/{direction}.pt").pi
        for direction in ["up", "down", "left", "right"]
    ]
    # pdb.set_trace()
    featurizer = MujocoAntFeaturizer()
    lang = FixedDepthAnt(
        maze_map=[
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        primitives=primitives,
        env_dim=111,
        depth=2,
        steps=1000,
        featurizer=featurizer,
    )
    # s = "if (1.0 + [0 1 2] * X >= 0) then [1 2 3] else [4, 5, 6]"
    s = f"""
        (ant (conds [{' '.join([str(i) for i in range(112)])}]) 
             (stmts [1. 0. 0. 0.] 
                    [0. 1. 0. 0.]))
    """
    tree = lang.parse(s)
    print(tree, lang.to_str(tree), sep='\n')
    print(lang._structured_params(tree))
    print(lang._flat_params(tree))
    weights = lang.eval(tree, {"state": np.random.rand(111)})
    print(weights)
    featurizer.apply([weights])


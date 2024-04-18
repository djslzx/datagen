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
            primitives: List[nn.Module],
            maze_map: List[List[int]],
            steps: int,
            featurizer: Featurizer,
            structure_params=False
    ):
        assert depth > 1
        assert steps > 0
        assert env_dim > 0
        assert all(cell == 0 or cell == 1
                   for row in maze_map for cell in row), \
                           f"Expected binary map but got {maze_map}"

        self.n_conds = depth - 1
        self.n_stmts = depth

        super().__init__(
            parser_grammar=FixedDepthAnt.grammar,
            parser_start="root",
            root_type=None,
            model=None,
            featurizer=featurizer,
        )

        self.env_dim = env_dim
        self.n_primitives = len(primitives)

        if structure_params:
            raise NotImplementedError("Structured params are not yet implemented")
        self.structure_params = structure_params

        self.primitives = primitives

        # mujoco env
        self.maze_map = maze_map
        self.env = gym.make(
            "AntMaze_UMaze-v4", 
            maze_map=self.maze_map, 
            render_mode="rgb_array_list",
        )
        self.steps = steps

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
        return FixedDepthAnt._flatten(cond_vec, stmt_vec)

    @staticmethod
    def _flatten(cond_vec: np.ndarray, stmt_vec: np.ndarray) -> np.ndarray:
        return ein.pack([cond_vec, stmt_vec], "*")[0]

    def sample(self) -> Tree:
        raise NotImplementedError

    def fit(self, corpus: List[Tree], alpha):
        raise NotImplementedError

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> np.ndarray:
        assert t.value == "root"
        assert not self.structure_params
        assert "state" in env, "Ant must evaluate on an env state"
        env_state = env["state"]
        assert isinstance(env_state, np.ndarray)
        assert env_state.ndim == 1
        assert env_state.shape[0] == self.env_dim

        # extract all parameters from program
        cond_params, stmt_params = self._structured_params(t)

        # simulate program to get weights over primitive policies
        action_weights = self.fold_eval_E(cond_params, stmt_params, env_state)

        # run mujoco env for `self.steps` steps and collect vector of positions of
        # ant center of mass
        obs, info = self.env.reset()
        step_start = 0
        episode = 0
        for step in tqdm(range(100)):
            pdb.set_trace()
            primitive_actions = np.stack([
                pi.act(obs, deterministic=True) for pi in self.primitives
            ])

            # weighted sum of primitive policies
            # todo: gumbel softmax
            action = np.matmul(action_weights, primitive_actions)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                save_video(
                    env.render(),
                    "videos",
                    fps=env.metadata["render_fps"],
                    step_starting_index=step_start,
                    episode_index = episode,
                )
                step_start = step + 1
                episode += 1
                observation, info = env.reset()
        
        save_video(
            env.render(),
            "videos",
            fps=env.metadata["render_fps"],
            step_starting_index=step_start,
            episode_index = episode,
        )
        env.close()

        # todo: return some feature of the animation
        return action


    def rec_eval_E(
            self,
            cond_params: np.ndarray,
            stmt_params: np.ndarray,
            env_state: np.ndarray,
            i=0,
    ) -> np.ndarray:
        """Evaluate E recursively"""
        assert i <= len(cond_params)
        if i == len(cond_params):
            return stmt_params[i]

        b = cond_params[i]
        c = stmt_params[i]
        w = softmax(b[-1] + np.dot(b[:-1], env_state))
        return w * c + (1 - w) * self.rec_eval_E(cond_params, stmt_params, env_state, i=i + 1)

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
        """
        Translates a series of weights over primitive functions into an ant path
        """
        return batch


if __name__ == "__main__":
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
        steps=100,
        featurizer=featurizer,
    )
    # s = "if (1.0 + [0 1 2] * X >= 0) then [1 2 3] else [4, 5, 6]"
    s = """
        (ant (conds [0 1]) 
             (stmts [0.25 0.25 0.25 0.25] 
                    [0.5 0.5 0. 0.]))
    """
    tree = lang.parse(s)
    print(tree, lang.to_str(tree), sep='\n')
    print(lang._structured_params(tree))
    print(lang._flat_params(tree))
    weights = lang.eval(tree, {"state": np.ones(1)})
    print(weights)
    featurizer.apply([weights])


from __future__ import annotations
import pdb
from typing import List, Dict, Any, Tuple
import numpy as np
import einops as ein
from scipy.special import softmax
from scipy import stats
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from tqdm import tqdm
import wandb

from lang.tree import Language, Tree, Grammar, ParseError, Featurizer
from lang.maze import Maze
from spinup.algos.pytorch.sac.core import SquashedGaussianMLPActor
import util


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class FixedDepthAnt(Language):
    """
    Mujoco ant domain from Programmatic RL without Oracles, with fixed tree depth.
    """
    # grammar with list structure
    grammar = r"""
        root: "(" "root" "(" "conds" conds ")" \
                         "(" "stmts" stmts ")" ")" -> root
        conds: vec+                                -> conds
        stmts: vec+                                -> stmts
        vec: "[" (float ","?)* "]"                 -> vec
        float: NUMBER                              -> pos
             | "-" NUMBER                          -> neg

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
            program_depth: int,
            maze: Maze,
            steps: int,
            featurizer: Featurizer,
            high_state_dim=9,   # state space dimensions in symbolic program (rangefinders)
            low_state_dim=111,  # state space dim for primitive policies
            primitives_dir="/home/djl328/prob-repl/src/lang/primitives/ant",
            camera="fixed",
            save_video=True,
            video_dir="videos",
            seed=0,
    ):
        assert program_depth > 1
        assert high_state_dim > 0
        assert low_state_dim > 0
        assert steps > 0
        assert camera in {"fixed", "follow"}, f"Camera setting must be either 'fixed' or 'follow'"
        assert maze.scaling == 4, \
            f"gymnasium AntMaze assumes scale of 4, but got maze with scale={maze.scaling}"

        self.n_conds = program_depth - 1
        self.n_stmts = program_depth
        self.save_video = save_video
        self.video_dir = video_dir
        self.camera_mode = camera
        self.seed = seed

        super().__init__(
            parser_grammar=FixedDepthAnt.grammar,
            parser_start="root",
            root_type=None,
            model=None,
            featurizer=featurizer,
        )

        # load primitives
        self.primitives = [
            torch.load(f"{primitives_dir}/{direction}.pt").pi.to(device)
            for direction in ["up", "down", "left", "right"]
        ]

        self.high_state_dim = high_state_dim
        self.low_state_dim = low_state_dim
        self.action_dim = len(self.primitives)

        self.n_cond_params = self.n_conds * (self.high_state_dim + 1)
        self.n_stmt_params = self.n_stmts * self.action_dim
        self.n_params = self.n_cond_params + self.n_stmt_params

        self.cond_shape = (self.n_conds, self.high_state_dim + 1)
        self.stmt_shape = (self.n_stmts, self.action_dim)

        # mujoco env
        self.steps = steps
        self.maze = maze
        self.gym_env = gym.make(
            "AntMaze_UMaze-v4", 
            maze_map=maze.str_map, 
            render_mode="rgb_array_list",
            camera_name="free" if self.camera_mode == "fixed" else None,
            use_contact_forces=True,  # required to match ICLR'22 paper
        )

        # initial sampling distribution: assume uniform weights
        null_data = np.zeros(self.n_cond_params + self.n_stmt_params)
        self.distribution = MultivariateGaussianSampler(null_data)

        # camera settings
        if self.save_video and self.camera_mode == "fixed":
            ant_env = self.gym_env.unwrapped.ant_env
            ant_env.mujoco_renderer.default_cam_config = {
                "trackbodyid": 0,
                "elevation": -60,
                "lookat": np.array([0, 0.0, 0.0]),
                "distance": ant_env.model.stat.extent * 1.5,
                "azimuth": 0,
            }

    def _parse_leaf(self, t: Tree) -> float:
        if t.value == "pos":
            return float(t.children[0].value)
        elif t.value == "neg":
            return -float(t.children[0].value)

    def _extract_params(self, t: Tree) -> Tuple[np.ndarray, np.ndarray]:
        assert t.value == "root"
        assert len(t.children) == 2
        conds = t.children[0]
        stmts = t.children[1]

        t_conds = np.stack([
            np.array([self._parse_leaf(gc) for gc in c.children])
            for c in conds.children
        ])
        assert t_conds.shape == self.cond_shape, \
            (f"Expected condition params of shape {self.cond_shape}, "
             f"but got {t_conds.shape}")

        t_stmts = np.stack([
            np.array([self._parse_leaf(gc) for gc in c.children])
            for c in stmts.children
        ])
        assert t_stmts.shape == self.stmt_shape, \
            (f"Expected statement params of shape {self.stmt_shape}, "
             f"but got {t_stmts.shape}")

        return t_conds, t_stmts

    def unflatten_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert len(params) == self.n_params
        conds = ein.rearrange(params[:self.n_cond_params], "(n d) -> n d", n=self.n_conds)
        stmts = ein.rearrange(params[self.n_cond_params:], "(n d) -> n d", n=self.n_stmts)
        return conds, stmts

    @staticmethod
    def _array_to_str(arr: np.ndarray) -> str:
        return "[" + " ".join(
            str(v) for v in arr
        ) + "]"

    def make_program(self, params: np.ndarray) -> Tree:
        assert params.ndim == 1
        assert params.shape[0] == self.n_params, f"Expected {self.n_params}, got {params.shape}"

        conds, stmts = self.unflatten_params(params)
        conds_str = " ".join(self._array_to_str(cond) for cond in conds)
        stmts_str = " ".join(self._array_to_str(stmt) for stmt in stmts)
        return self.parse(f"""
            (root (conds {conds_str}) 
                  (stmts {stmts_str}))
        """)

    def sample(self) -> Tree:
        params = self.distribution.sample(k=1)
        return self.make_program(params)

    def fit(self, corpus: List[Tree], alpha):
        all_params = []
        for tree in corpus:
            c, s = self._extract_params(tree)
            params, _ = ein.pack([c, s], "*")
            all_params.append(params)

        self.distribution = MultivariateGaussianSampler(data)

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> Iterable[np.ndarray]:
        assert t.value == "root"

        # extract parameters from program
        conds, stmts = self._extract_params(t)
        
        # run sim loop
        obs, info = self.gym_env.reset(seed=self.seed)

        # output ant trail
        outputs = []

        for step in tqdm(range(self.steps), desc="Evaluating ant"):
            x, y = obs['achieved_goal']
            outputs.append([x, y])

            # construct high-level observations for symbolic program
            orientation = obs['observation'][:5]
            dists = self.maze.cardinal_wall_distances(x, y)
            high_obs, _ = ein.pack([orientation, dists], "*")

            # get low-level observations for policy
            low_obs = obs['observation']
        
            action = self.act_from_params(conds, stmts, high_obs=high_obs, low_obs=low_obs)
            obs, _, terminated, truncated, info = self.gym_env.step(action)

            if terminated or truncated:
                break

        if self.save_video:
            save_video(
                self.gym_env.render(),
                self.video_dir,
                fps=self.gym_env.metadata["render_fps"],
                step_starting_index=0,
                episode_index = 0,
            )
        # self.gym_env.close()  # don't close for now b/c we want to eval multiple times per language
        return np.array(outputs)

    @staticmethod
    def get_action(model: nn.Module, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, f"Model expects 2D array x, got {x.shape}"
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).to(device)
            action, _ = model(x, deterministic=True)
        return action.cpu().numpy()

    def act_from_params(
            self, 
            conds: np.ndarray,
            stmts: np.ndarray,
            high_obs: np.ndarray,
            low_obs: np.ndarray,
    ) -> np.ndarray:
        assert high_obs.ndim == 1, f"Expected 1D high-level obs array, got {high_obs.shape}"
        assert low_obs.ndim == 1, f"Expected 1D low-level obs array, got {low_obs.shape}"
        assert high_obs.shape[0] == self.high_state_dim, \
            f"Expected high obs dim {self.high_state_dim}, got {high_obs.shape}"
        assert low_obs.shape[0] == self.low_state_dim, \
            f"Expected low obs dim {self.low_state_dim}, got {low_obs.shape}"

        # simulate program to get weights over primitive policies
        action_weights = self.fold_eval_E(conds, stmts, high_obs)

        # compile primitives
        primitive_actions = []
        for pi in self.primitives:
            pi_action = self.get_action(pi, low_obs[None, :])
            primitive_actions.append(pi_action)
        primitive_actions = ein.rearrange(primitive_actions, "n 1 d -> (n 1) d")

        # action = weighted sum of primitives
        action = action_weights @ primitive_actions
        
        return action

    def fold_eval_E(
            self, 
            conds: np.ndarray,
            stmts: np.ndarray,
            state: np.ndarray,
    ) -> np.ndarray:
        """Evaluate E by folding, no recursion"""
        e = stmts[-1]
        for b, c in zip(conds[::-1], stmts[:-1][::-1]):
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
            "pos": lambda n: f"{n}",
            "neg": lambda n: f"-{n}",
        }


class TrailFeaturizer(Featurizer):
    """
    Take the full trail as features, using stride to cut if desired
    """
    
    def __init__(self, stride: int = 1):
        self.stride = stride

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        if isinstance(batch, list):
            batch = np.stack(batch)
        assert batch.ndim == 2, f"Expected 2D batch, got {batch.shape}"
        assert batch.shape[1] == 2, f"Expected 2D points, got {batch.shape}"
        return batch[::self.stride]


class EndFeaturizer(Featurizer):
    def __init__(self):
        pass
    
    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        if isinstance(batch, list):
            batch = np.stack(batch)
        assert batch.ndim == 2, f"Expected 2D batch, got {batch.shape}"
        assert batch.shape[1] == 2, f"Expected 2D points, got {batch.shape}"
        return batch[-1]


class HeatMapFeaturizer(Featurizer):
    """
    Generate a heatmap over maze cells; an 'unordered', discrete representation
    """

    def __init__(self, maze: Maze):
        self.width = maze.width
        self.height = maze.height
        self.scaling = maze.scaling
        self.xy_to_rc = maze.xy_to_rc
        
    def apply(self, batch: List[np,ndarray]) -> np.ndarray:
        if isinstance(batch, list):
            batch = np.stack(batch)
        assert batch.ndim == 2, f"Expected 2D batch, got {batch.shape}"
        assert batch.shape[1] == 2, f"Expected 2D points, got {batch.shape}"

        heatmap = np.zeros((self.width, self.height))  # treat i,j as equal to x,y
        for x, y in batch:
            r, c = self.xy_to_rc(x, y)
            print(f"{x, y} => {r, c}")
            heatmap[r, c] += 1

        # normalize
        heatmap = heatmap / heatmap.sum()

        return heatmap


class MultivariateGaussianSampler:
    def __init__(self, data: np.ndarray):
        assert data.ndim in {1, 2}, f"Expected 1D or 2D data, but got {data.shape}"

        # if data consists of a single example in 2D, flatten to 1D
        if data.shape[0] == 1:
            data = np.ravel()

        # use different distributions depending on whether we get
        #  a single data point or many
        if data.ndim == 1:
            self.rv_kind = "single"
            self.rv = stats.multivariate_normal(mean=data)
        else:
            n, d = data.shape
            assert n >= d, "Number of samples must be greater than number of dimensions"

            self.rv_kind = "multiple"
            self.rv = stats.gaussian_kde(data.T)

    def sample(self, k: int) -> np.ndarray:
        if self.rv_kind == "single":
            return self.rv.rvs(k)
        else:
            samples = self.rv.resample(k)
            return ein.rearrange(samples, "d k -> k d", k=k)

    def logpdf(self, x) -> float:
        return self.rv.logpdf(x).item()


if __name__ == "__main__":
    np.random.seed(0)

    ts = util.timestamp()
    video_dir = f"videos/{ts}"
    util.try_mkdir(video_dir)

    maze = Maze.from_saved(
        "lehman-ecj-11-hard"
        # "cross"
    )
    featurizer = HeatMapFeaturizer(maze)
    lang = FixedDepthAnt(
        maze=maze,
        program_depth=6,
        steps=1000,
        featurizer=featurizer,
        save_video=True,
        video_dir=video_dir,
        camera="follow",
    )
    params = np.random.rand(lang.n_params) * 2 - 1
    p = lang.make_program(params)
    s = lang.to_str(p)

    tree = lang.parse(s)
    print(tree, lang.to_str(tree), sep='\n')
    print(lang._extract_params(tree))
    coords = lang.eval(tree)
    feat_vec = featurizer.apply(coords)
    print(feat_vec)
    print(maze.trail_img(coords))

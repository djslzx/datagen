from __future__ import annotations
import pdb
from typing import List, Dict, Any, Tuple, Iterable
import numpy as np
import einops as ein
from scipy import stats
from scipy.special import expit
from tqdm import tqdm
import matplotlib.pyplot as plt

from lang.tree import Language, Tree, Featurizer
from lang.maze import Maze
from lang.ant_env import Environment, AntMaze2D
import util


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
            steps: int,
            env: Environment,
            featurizer: Featurizer,
            include_orientation=False,
    ):
        assert program_depth > 1
        assert steps > 0

        self.env = env
        self.steps = steps
        self.n_conds = program_depth - 1
        self.n_stmts = program_depth

        super().__init__(
            parser_grammar=FixedDepthAnt.grammar,
            parser_start="root",
            root_type=None,
            model=None,
            featurizer=featurizer,
        )

        self.action_dim = 4
        self.high_state_dim = 4  # rangefinders, cardinal directions
        if include_orientation:
            self.high_state_dim += 5

        # parameter counts and shapes
        self.n_conds = program_depth - 1
        self.n_stmts = program_depth
        self.n_cond_params = self.n_conds * (self.high_state_dim + 1)
        self.n_stmt_params = self.n_stmts * self.action_dim
        self.n_params = self.n_cond_params + self.n_stmt_params
        self.cond_shape = (self.n_conds, self.high_state_dim + 1)
        self.stmt_shape = (self.n_stmts, self.action_dim)

        # initial sampling distribution: assume uniform weights
        null_data = np.zeros(self.n_params)
        self.distribution = MultivariateGaussianSampler(null_data)

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
        assert params.ndim == 1, f"Programs must be constructed from 1D parameter vectors, got {params.shape}"
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
        return self.make_program(params[0])

    def fit(self, corpus: List[Tree], alpha):
        all_params = []
        for tree in corpus:
            c, s = self._extract_params(tree)
            params, _ = ein.pack([c, s], "*")
            all_params.append(params)

        all_params = np.array(all_params)
        self.distribution = MultivariateGaussianSampler(all_params)

    def log_probability(self, t: Tree) -> float:
        c, s = self._extract_params(t)
        params, _ = ein.pack([c, s], "*")
        return self.distribution.logpdf(params)

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> np.ndarray:
        assert t.value == "root"

        conds, stmts = self._extract_params(t)
        obs = self.env.reset()
        outputs = []

        steps = range(self.steps)
        if env is not None and "load_bar" in env:
            steps = tqdm(steps, desc="Evaluating program")
        for _ in steps:
            x, y = obs.state
            assert obs.observation.shape == (self.high_state_dim,), \
                f"Expected high obs dim {self.high_state_dim}, got {obs.observation.shape}"

            action_weights, stmt_weights = self.fold_eval_E(conds, stmts, obs.observation)
            outputs.append([x, y, *stmt_weights])
            obs = self.env.step(action_weights)

            if obs.ended:
                break

        outputs = np.array(outputs)

        if len(outputs) < self.steps:
            # repeat last position
            padding = ein.repeat(outputs[-1], "d -> n d", n=self.steps - len(outputs))
            outputs = np.concatenate([outputs, padding], axis=0)

        return outputs

    @staticmethod
    def fold_eval_E(
            conds: np.ndarray,
            stmts: np.ndarray,
            state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate E by folding, no recursion"""
        e = stmts[-1]
        weights = np.zeros(len(stmts))
        weights[0] = 1.0
        for i, (b, c) in enumerate(zip(conds[::-1], stmts[:-1][::-1]), 1):
            w = expit(b[-1] + np.dot(b[:-1], state))
            e = w * c + (1 - w) * e
            weights *= 1 - w
            weights[i] = w
        return e, weights[::-1]

    @staticmethod
    def arr_eval_E(
            conds: np.ndarray,
            stmts: np.ndarray,
            state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cond_weights = expit(conds[:, -1] + conds[:, :-1] @ state)
        stmt_weights = np.ones(len(stmts))
        for i, c in enumerate(cond_weights, 1):
            stmt_weights[-(i + 1)] = c
            stmt_weights[-i:] *= 1 - c
        policy = np.dot(stmt_weights, stmts)
        return policy, stmt_weights

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


def test_eval_E():
    cases = [
        ([[0, 0]], [[1], [2]], [1],
         [0.5 * 1 + 0.5 * 2], [0.5, 0.5]),
        ([[0, 0], [0, 0], [0, 0]], [[1], [2], [3], [4]], [1],
         [0.5 * 1 + 0.25 * 2 + 0.125 * 3 + 0.125 * 4], [0.5, 0.25, 0.125, 0.125]),
        ([[0, 100], [0, 100], [0, 100]], [[1], [2], [3], [4]], [1],
         [1 * 1], [1, 0, 0, 0]),
    ]
    for conds, stmts, state, policy, weights in cases:
        inputs = (
            np.array(conds),
            np.array(stmts),
            np.array(state),
        )
        arr_policy, arr_weights = FixedDepthAnt.arr_eval_E(*inputs)
        assert np.array_equal(arr_policy, policy), f"Expected {policy}, got {arr_policy}"
        assert np.array_equal(arr_weights, weights), f"Expected {weights}, got {arr_weights}"

        fold_policy, fold_weights = FixedDepthAnt.fold_eval_E(*inputs)
        assert np.array_equal(fold_policy, policy), f"Expected {policy}, got {fold_policy}"
        assert np.array_equal(fold_weights, weights), f"Expected {weights}, got {fold_weights}"


class TrailFeaturizer(Featurizer):
    """
    Take the full trail as features, using stride to cut if desired
    """

    def __init__(self, stride: int = 1):
        self.stride = stride

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        if isinstance(batch, list):
            batch = np.stack(batch)
        assert batch.ndim == 3, f"Expected 3D batch, got {batch.shape}"
        assert batch.shape[-1] == 2, f"Expected 2D points, got {batch.shape}"

        # stride through t dim, then flatten
        return ein.rearrange(batch[:, ::self.stride, :2], "b t d -> b (t d)")


class EndFeaturizer(Featurizer):
    def __init__(self):
        pass

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        if isinstance(batch, list):
            batch = np.stack(batch)
        assert batch.ndim == 3, f"Expected 3D batch, got {batch.shape}"
        assert batch.shape[-1] == 2, f"Expected 2D points, got {batch.shape}"
        return batch[:, -1, :2]


class HeatMapFeaturizer(Featurizer):
    """
    Generate a heatmap over maze cells; an 'unordered', discrete representation
    """

    def __init__(self, maze: Maze):
        self.width = maze.width
        self.height = maze.height
        self.scaling = maze.scaling
        self.xy_to_rc = maze.xy_to_rc

    def apply(self, batch: List[np.ndarray]) -> np.ndarray:
        if isinstance(batch, list):
            batch = np.stack(batch)
        assert batch.ndim == 3, f"Expected 3D batch, got {batch.shape}"
        assert batch.shape[-1] == 2, f"Expected 2D points, got {batch.shape}"

        batch_size = batch.shape[0]
        heatmaps = np.zeros((batch_size, self.width, self.height))  # treat i,j as equal to x,y
        for i, coords in enumerate(batch[:, :, :2]):
            for x, y in coords:
                r, c = self.xy_to_rc(x, y)
                heatmaps[i, r, c] += 1

        heatmaps /= ein.reduce(heatmaps, "b h w -> b () ()", "sum")  # normalize
        heatmaps = ein.rearrange(heatmaps, "b h w -> b (h w)")  # flatten
        return heatmaps


class ProgramWeightFeaturizer(Featurizer):
    """Characterize programs by how much weight they place on statement nodes"""

    def __init__(self):
        pass

    def apply(self, batch: Any) -> np.ndarray:
        if isinstance(batch, list):
            batch = np.stack(batch)
        assert batch.ndim == 3, f"Expected 3D batch, got {batch.shape}"
        assert batch.shape[-1] == 2, f"Expected 2D points, got {batch.shape}"

        return ein.rearrange(batch[:, :, 2:], "b n d -> b (n d)")


class MultivariateGaussianSampler:
    def __init__(self, data: np.ndarray):
        assert data.ndim in {1, 2}, f"Expected 1D or 2D data, but got {data.shape}"

        # if data consists of a single example in 2D, flatten to 1D
        if data.shape[0] == 1:
            data = np.ravel(data)

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
            samples = self.rv.rvs(k)
            if k == 1:
                return samples[None, :]
            else:
                return samples
        else:
            samples = self.rv.resample(k)
            return ein.rearrange(samples, "d k -> k d", k=k)

    def logpdf(self, x) -> float:
        return self.rv.logpdf(x).item()


def simple_ant_test():
    maze = Maze.from_saved("lehman-ecj-11-hard")
    environment = AntMaze2D(
        maze_map=maze,
        step_length=0.5,
    )
    featurizer = HeatMapFeaturizer(maze)
    lang = FixedDepthAnt(
        env=environment,
        program_depth=6,
        steps=1000,
        featurizer=featurizer,
    )
    trees = lang.samples(n_samples=50, length_cap=10_000)
    # trees.append(
    #     lang.parse("""
    #         (root (conds [1 0 0 0 -2])
    #               (stmts [1 0 0 0]
    #                      [0 1 0 0]))
    #     """))

    trails = []
    for tree in trees:
        output = lang.eval(tree)
        trail = output[:, :2]
        weights = output[:, 2:]
        print(np.round(weights, 2))
        trails.append(trail)

    maze.plot_trails(np.array(trails))
    plt.savefig("trails.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # np.random.seed(0)
    # ts = util.timestamp()
    # video_dir = f"videos/{ts}"
    # util.try_mkdir(video_dir)

    simple_ant_test()

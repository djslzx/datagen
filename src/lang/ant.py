from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import einops as ein
from scipy.special import softmax

from lang.tree import Language, Tree, Grammar, ParseError, Featurizer


class Ant(Language):
    """
    Mujoco ant domain from Programmatic RL without Oracles.
    """
    grammar = r"""
        e: "if" "(" b ")" "then" vec "else" "("? e ")"? -> if
         | vec                                          -> c
        b: NUMBER "+" vec "* X >= 0"                    -> b
        vec: "[" (NUMBER ","?)* "]"                     -> vec
        
        %import common.NUMBER
        %import common.WS
        %ignore WS
    """

    def __init__(self, env_dim: int, primitive_policies: np.ndarray):
        super().__init__(
            parser_grammar=Ant.grammar,
            parser_start="e",
            root_type=None,
            model=None,
            featurizer=AntFeaturizer(),
        )
        self.env_dim = env_dim
        assert primitive_policies.ndim == 2
        # print(f"Loaded primitive policies with shape {primitive_policies.shape}")
        self.primitive_policies = primitive_policies

    def _extract_weight_map(self, t: Tree) -> np.ndarray:
        raise NotImplementedError

    def sample(self) -> Tree:
        raise NotImplementedError

    def fit(self, corpus: List[Tree], alpha):
        raise NotImplementedError

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> Any:
        assert "state" in env, "Ant must evaluate on an env state"
        env_state = env["state"]
        assert isinstance(env_state, np.ndarray)
        assert env_state.ndim == 1
        assert env_state.shape[0] == self.env_dim

        # 1. extract all weights and organize into structure:
        #   - single vector of all weights (conds and returns)
        #   - structured vector of weights by node
        weights = self._extract_weight_map(t)

        # 2. simulate on input env
        pass

        # 3. simulate ant in mujoco using combination of primitive policies pulled
        #    from file
        pass

        return t

    @property
    def str_semantics(self) -> Dict:
        return {
            "if": lambda b, c, e: f"if ({b}) then {c} else ({e})",
            "c": lambda v: f"{v}",
            "b": lambda n, v: f"{n} + {v} * X >= 0",
            "vec": lambda *xs: "[ " + " ".join(xs) + " ]",
        }


class FixedDepthAnt(Language):
    """
    Mujoco ant domain from Programmatic RL without Oracles, with fixed tree depth.
    """
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

    def __init__(self, env_dim: int, primitive_policies: np.ndarray, depth: int, structure_params=False):
        assert depth > 1
        self.n_conds = depth - 1
        self.n_stmts = depth

        super().__init__(
            parser_grammar=FixedDepthAnt.grammar,
            parser_start="root",
            root_type=None,
            model=None,
            featurizer=AntFeaturizer(),
        )

        self.env_dim = env_dim
        assert primitive_policies.ndim == 2

        self.primitive_policies = primitive_policies

        if structure_params:
            raise NotImplementedError("Structured params not implemented")
        self.structure_params = structure_params

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
        assert stmt_vec.shape == (self.n_stmts, self.primitive_policies.shape[0]), \
            (f"Expected statement params of shape {(self.n_stmts, self.primitive_policies.shape[0])}, "
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

    def eval(self, t: Tree, env: Dict[str, Any] = None) -> Any:
        assert t.value == "root"
        assert not self.structure_params
        assert "state" in env, "Ant must evaluate on an env state"
        env_state = env["state"]
        assert isinstance(env_state, np.ndarray)
        assert env_state.ndim == 1
        assert env_state.shape[0] == self.env_dim

        # extract all parameters from program
        cond_params, stmt_params = self._structured_params(t)

        # simulate program to get choice of primitives
        action = self.fold_eval_E(cond_params, stmt_params, env_state, i=0)

        # simulate ant in mujoco using combination of primitive policies
        pass

        return action

    def rec_eval_E(
            self,
            cond_params: np.ndarray,
            stmt_params: np.ndarray,
            env_state: np.ndarray,
            i: int,
    ) -> np.ndarray:
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


class AntFeaturizer(Featurizer):
    """
    Ant features: points along trajectory of ant?
    """
    pass


if __name__ == "__main__":
    # primitives_paths = [
    #     "pi-PRL/primitives/ant/up.pt",
    #     "pi-PRL/primitives/ant/down.pt",
    #     "pi-PRL/primitives/ant/left.pt",
    #     "pi-PRL/primitives/ant/right.pt",
    # ]
    # primitives = np.stack([
    #     np.load(path) for path in primitives_paths
    # ])
    primitives = np.random.rand(3, 10)
    lang = FixedDepthAnt(
        env_dim=1,
        primitive_policies=primitives,
        depth=2,
    )
    # s = "if (1.0 + [0 1 2] * X >= 0) then [1 2 3] else [4, 5, 6]"
    s = """
        (ant (conds [0 1]) 
             (stmts [0.3 0.3 0.3] [1 0 0]))
    """
    tree = lang.parse(s)
    print(tree, lang.to_str(tree), sep='\n')
    print(lang._structured_params(tree))
    print(lang._flat_params(tree))
    out = lang.eval(tree, {"state": np.ones(1)})
    print(out)

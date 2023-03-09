"""
Test out evolutionary search algorithms for data augmentation.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import *
from pprint import pp
import time
import Levenshtein
from einops import rearrange, reduce
from sys import stderr

from lang import Language, Tree, ParseError
from lindenmayer import LSys
from regexpr import Regex
from zoo import zoo
from util import Timing, ParamTester, try_mkdir

# Hyper-parameters
OUTDIR = "../out/evo"


def next_gen(lang: Language, n: int, p_arkv: float, simplify: bool) -> Tuple[np.ndarray, np.ndarray]:
    popn = np.empty(n, dtype=object)
    arkv = set()
    n_simplified = 0
    for i in range(n):  # parallel
        t = lang.sample()
        if simplify:
            while True:  # retry until we get non-nil
                try:
                    n_old = len(t)
                    t = lang.simplify(t)
                    n_simplified += n_old - len(t)
                    break
                except ParseError:  # retry
                    t = lang.sample()
        popn[i] = t
        if np.random.random() < p_arkv:
            arkv.add(t)
    print(f"Simplified {n_simplified / n:.3e} tokens on avg")
    return popn, np.array(list(arkv), dtype=object)


def pad_array(arr: np.ndarray, batch_size: int) -> np.ndarray:
    if (r := len(arr) % batch_size) != 0:
        return np.concatenate((arr, np.empty(batch_size - r, dtype=object)))
    else:
        return arr


def semantic_score(lang: Language, cur_gen: np.ndarray, new_gen: np.ndarray,
                   n_samples: int, n_neighbors: int, batch_size: int = 7) -> np.ndarray:
    n_new = len(new_gen)
    n_cur = len(cur_gen)
    # batch cur_/next_gen
    # 1. add padding
    cur_gen = pad_array(cur_gen, batch_size)
    new_gen = pad_array(new_gen, batch_size)
    # 2. reshape
    cur_gen = rearrange(cur_gen, "(b x) -> b x", x=batch_size)
    new_gen = rearrange(new_gen, "(b x) -> b x", x=batch_size)

    with Timing("Computing features"):
        popn_features = []
        for batch in cur_gen:
            outputs = [lang.eval(t, env={})
                       for t in batch if t is not None
                       for _ in range(n_samples)]
            features = lang.featurizer.apply(outputs)
            popn_features.append(features)
        popn_features = np.concatenate(popn_features)

    if n_neighbors > n_cur:
        print(f"WARNING: number of neighbors ({n_neighbors}) is greater than #individuals in current generation ({n_cur})."
              f"Lowering n_neighbors to {n_cur}.", file=stderr)
        n_neighbors = n_cur
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(popn_features)

    with Timing("Scoring instances"):
        scores = []
        for batch in new_gen:
            outputs = [lang.eval(t, env={})
                       for t in batch if t is not None
                       for _ in range(n_samples)]
            features = lang.featurizer.apply(outputs)
            distances, _ = knn.kneighbors(features)
            batch_scores = reduce(distances, "(b s) n -> b", 'mean', s=n_samples, n=n_neighbors)
            scores.append(batch_scores)
        scores = np.concatenate(scores)

    assert len(scores) == n_new
    return scores


def syntactic_semantic_score(popn: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
    n_popn = len(popn)
    scores = np.empty(n_popn)
    for i, s in enumerate(popn):
        syntactic_score = sum(Levenshtein.distance(s, x) for x in popn) / n_popn
        scores[i] = semantic_scores[i] * syntactic_score
    return scores


def novelty_search(lang: Language,
                   init_popn: List[Tree],
                   max_popn_size: int,
                   iters: int,
                   n_samples: int,
                   arkv_growth_rate: float,
                   n_neighbors: int,
                   next_gen_ratio: int,
                   simplify: bool,       # use egg to simplify expressions between generations
                   ingen_novelty: bool,  # measure novelty wrt current gen, not archive/past gens
                   out_dir: str) -> np.ndarray:  # store outputs here
    log_params = locals()  # Pull local variables so that we can log the args that were passed to this function
    p_arkv = arkv_growth_rate / max_popn_size

    arkv = None
    popn = np.array(init_popn, dtype=object)
    n_next_gen = max_popn_size * next_gen_ratio

    for iter in range(iters):
        print(f"[Novelty search: iter {iter}]")
        t_start = time.time()

        # generate next gen
        with Timing("Fitting metagrammar"):
            lang.fit(popn, alpha=1)

        with Timing("Generating next gen"):
            new_gen, new_arkv = next_gen(lang=lang, n=n_next_gen, p_arkv=p_arkv, simplify=simplify)
            if arkv is not None:
                arkv = np.concatenate((arkv, new_arkv), axis=0)
            else:
                arkv = new_arkv

        # compute popn features, build knn data structure, and score new_gen
        with Timing("Scoring population"):
            if ingen_novelty:
                scores = semantic_score(lang=lang, cur_gen=new_gen, new_gen=new_gen,
                                        n_neighbors=n_neighbors, n_samples=n_samples)
            else:
                scores = semantic_score(lang=lang, cur_gen=np.concatenate((arkv, popn), axis=0), new_gen=new_gen,
                                        n_neighbors=n_neighbors, n_samples=n_samples)

        # cull popn
        with Timing("Culling popn"):
            indices = np.argsort(-scores)  # sort descending: higher mean distances first
            new_gen = new_gen[indices]
            popn = new_gen[:max_popn_size]  # take indices of top `max_popn_size` agents

        # make labels
        with Timing("Logging"):
            scores = scores[indices]
            min_score = scores[max_popn_size - 1]
            labels = [f"{score:.2e}" + ("*" if score >= min_score else "")
                      for score in scores]

        # printing/saving
        print(f"[Completed iteration {iter} in {time.time() - t_start:.2f}s.]")

        # save gen
        with open(f"{out_dir}/gen-{iter}.txt", "w") as f:
            f.write(f"# {log_params}\n")
            for t, label in zip(new_gen, labels):
                f.write("".join(lang.to_str(t)) + f" : {label}\n")

        # save arkv
        with open(f"{out_dir}/arkv.txt", "a") as f:
            for t in new_arkv:
                f.write(''.join(lang.to_str(t)) + "\n")

    return arkv


def main(name: str, lang: Language, init_popns: List[List]):
    t = int(time.time())
    p = ParamTester({
        'lang': lang,
        'init_popn': init_popns,
        'simplify': [False],
        'max_popn_size': [25],
        'n_neighbors': [5],
        'arkv_growth_rate': [1],
        'iters': 10,
        'next_gen_ratio': 5,
        'ingen_novelty': False,
        'n_samples': 4,
    })
    for i, params in enumerate(p):
        out_dir = f"{OUTDIR}/{t}-{name}-{i}"
        try_mkdir(out_dir)
        params.update({
            "out_dir": out_dir,
        })
        print("****************")
        print(f"Running {i}-th novelty search with parameters:")
        pp(params)
        novelty_search(**params)


if __name__ == '__main__':
    # lsys = LSys(theta=45, step_length=3, render_depth=3, n_rows=128, n_cols=128)
    # lsys_seeds = {
    #     "random": [lsys.sample() for _ in range(len(zoo))],  # lsys starts out as uniform
    #     "zoo": [lsys.parse(x.to_str()) for x in zoo],
    #     "simple": [
    #         "F;F~F",
    #         "F;F~[+F][-F]F,F~F-F",
    #         "F;F~FF",
    #         "F+F;F~F[-F],F~F[+F]",
    #     ],
    # }
    # main('intrasimpl', lang=lsys, init_popns=list(lsys_seeds.values()))

    reg = Regex()
    reg_seeds = [reg.sample() for _ in range(10)]
    main('regex', lang=reg, init_popns=[reg_seeds])

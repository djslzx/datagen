"""
Test out evolutionary search algorithms for data augmentation.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import *
from pprint import pp
import time
import Levenshtein

from lang import Language, ParseError
from lindenmayer import LSys
from regexpr import Regex
from zoo import zoo
from featurizers import ResnetFeaturizer, Featurizer
from util import Timing, ParamTester, try_mkdir

# Hyper-parameters
OUTDIR = "../out/evo"


def next_gen(lang: Language, n: int, p_arkv: float, simplify: bool) -> Tuple[np.ndarray, Set]:
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
        s = lang.to_str(t)
        popn[i] = s
        if np.random.random() < p_arkv:
            arkv.add(s)
    print(f"Simplified {n_simplified / n:.3e} tokens on avg")
    return popn, arkv


def semantic_score(lang: Language, cur_gen: np.ndarray, new_gen: np.ndarray, n_samples: int, n_neighbors: int) -> np.ndarray:
    n_features = lang.featurizer.n_features

    # build knn data structure
    with Timing("Computing features"):
        popn_features = np.empty((len(cur_gen), n_features * n_samples))
        for i, s in enumerate(cur_gen):  # parallel
            t = lang.parse(s)
            for j in range(n_samples):
                bmp = lang.eval(t, env={})
                # TODO: handle non-bitmaps too -- this should work for text outputs as well
                popn_features[i, j * n_features: (j + 1) * n_features] = lang.featurizer.apply(bmp)

    with Timing("Building knn data structure"):
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(cur_gen))).fit(popn_features)

    # compute scores of next generation
    with Timing("Scoring instances"):
        scores = np.empty(len(new_gen))
        for i, s in enumerate(new_gen):  # parallel
            features = np.empty((1, n_features * n_samples))
            t = lang.parse(s)
            for j in range(n_samples):
                bmp = lang.eval(t, env={})
                features[0, j * n_features: (j + 1) * n_features] = lang.featurizer.apply(bmp)
            distances, indices = knn.kneighbors(features)
            scores[i] = distances.sum(axis=1).item()
            # scores[i] = distances.mean(axis=1).item() ** 2 / len(s)  # prioritize shorter agents

    return scores


def syntactic_semantic_score(popn: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
    n_popn = len(popn)
    scores = np.empty(n_popn)
    for i, s in enumerate(popn):
        syntactic_score = sum(Levenshtein.distance(s, x) for x in popn) / n_popn
        scores[i] = semantic_scores[i] * syntactic_score
    return scores


def novelty_search(lang: Language,
                   init_popn: List[str],
                   max_popn_size: int,
                   iters: int,
                   n_samples: int,
                   arkv_growth_rate: float,
                   n_neighbors: int,
                   next_gen_ratio: int,
                   simplify: bool,       # use egg to simplify expressions between generations
                   ingen_novelty: bool,  # measure novelty wrt current gen, not archive/past gens
                   out_dir: str) -> Set[Iterable[str]]:  # store outputs here
    log_params = locals()  # Pull local variables so that we can log the args that were passed to this function
    p_arkv = arkv_growth_rate / max_popn_size

    arkv = set()
    popn = np.array(init_popn, dtype=object)
    n_next_gen = max_popn_size * next_gen_ratio

    for iter in range(iters):
        print(f"[Novelty search: iter {iter}]")
        t_start = time.time()

        # generate next gen
        with Timing("Fitting metagrammar"):
            corpus = [lang.parse(x) for x in popn]
            lang.fit(corpus, alpha=1)

        with Timing("Generating next gen"):
            new_gen, new_arkv = next_gen(lang=lang,
                                         n=n_next_gen,
                                         p_arkv=p_arkv,
                                         simplify=simplify)
            arkv |= new_arkv

        # compute popn features, build knn data structure, and score new_gen
        with Timing("Scoring population"):
            if ingen_novelty:
                scores = semantic_score(lang=lang,
                                        cur_gen=new_gen,
                                        new_gen=new_gen,
                                        n_neighbors=n_neighbors,
                                        n_samples=n_samples)
            else:
                scores = semantic_score(lang=lang,
                                        cur_gen=np.concatenate((np.array(popn), np.array(list(arkv), dtype=object)), axis=0),
                                        new_gen=new_gen,
                                        n_neighbors=n_neighbors,
                                        n_samples=n_samples)

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
            for agent, label in zip(new_gen, labels):
                f.write("".join(agent) + f" : {label}\n")

        # save arkv
        with open(f"{out_dir}/arkv.txt", "a") as f:
            for agent in new_arkv:
                f.write(''.join(agent) + "\n")

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
        'iters': 1,
        'next_gen_ratio': 5,
        'ingen_novelty': False,
        'n_samples': 3,
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
    lsys = LSys(theta=45, step_length=3, render_depth=3, n_rows=128, n_cols=128)
    lsys_seeds = {
        "random": [lsys.to_str(lsys.sample()) for _ in range(len(zoo))],  # lsys starts out as uniform
        "zoo": [x.to_str() for x in zoo],
        "simple": [
            "F;F~F",
            "F;F~[+F][-F]F,F~F-F",
            "F;F~FF",
            "F+F;F~F[-F],F~F[+F]",
        ],
    }
    # main('intrasimpl', lang=lsys, init_popns=list(lsys_seeds.values()))

    reg = Regex()
    reg_seeds = [reg.to_str(reg.sample()) for _ in range(10)]
    main('regex', lang=reg, init_popns=[reg_seeds])

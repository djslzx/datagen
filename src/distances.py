"""
Experiments with different distance metrics for probabilistic programs
"""
from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools as it
from sklearn.manifold import MDS
from Levenshtein import distance as levenshtein
from scipy.spatial.distance import minkowski, directed_hausdorff
from einops import rearrange

from featurizers import TextClassifier
from lang.regexpr import Regex

def embed(xs: List[str]) -> np.ndarray:
    C = TextClassifier()
    return C.apply(xs)

def sort_by_instance_distances(corpus: List[str]):
    """Sanity-check distances for individuals using order-1 metrics"""
    embeddings = embed(corpus)
    for a, u in zip(corpus, embeddings):
        # a, b: strings
        # u, v: embeddings
        print(f"Minkowski distance from {a}: ")
        for d, b in sorted([(minkowski(u, v), b) for b, v in zip(corpus, embeddings)]):
            print(f"  {b}: {d}")

        print(f"Levenshtein distance from {a}: ")
        for d, b in sorted([(levenshtein(a, b), b) for b in corpus]):
            print(f"  {b}: {d}")

def viz_labeled_points(points: np.ndarray, labels: List[str]):
    assert points.ndim == 2, f"Expected 2d points but received points.shape={points.shape}"
    n = points.shape[0]
    assert len(labels) == n, \
        f"Points should have same cardinality as labels, but got |points|={n}, |labels|={len(labels)}"
    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "labels": labels})
    sns.relplot(df, x="x", y="y")
    for i, label in enumerate(labels):
        plt.text(points[i, 0] + 0.01, points[i, 1] + 0.01, label)

def viz_grouped_points(points: np.ndarray, groups: List[str]):
    assert len(points.shape) == 2, f"Expected 2d points but got points.shape={points.shape}"
    df = pd.DataFrame({"x": points[:, 0],
                       "y": points[:, 1],
                       "group": groups})
    sns.relplot(df, x="x", y="y", hue="group", style="group")

def viz_instance_distance_by_MDS(corpus: List[str]):
    embeddings = embed(corpus)
    mds = MDS(
        n_components=2,
        random_state=0,
    )
    points = mds.fit_transform(embeddings)
    viz_labeled_points(points, labels=corpus)

def chamfer(X, Y):
    return (sum(min(np.dot(x - y, x - y) for y in Y) for x in X) +
            sum(min(np.dot(x - y, x - y) for x in X) for y in Y))

def sqrt_chamfer(X, Y):
    return (sum(min(np.linalg.norm(x - y) for y in Y) for x in X) +
            sum(min(np.linalg.norm(x - y) for x in X) for y in Y))

def hausdorff(X, Y):
    return directed_hausdorff(X, Y)[0]

def viz_set_instances(labels: List[str], samples: List[List[str]]):
    embeddings = np.array([embed(grp) for grp in samples])  # [n, m]
    mds = MDS(n_components=2, random_state=0)
    points = mds.fit_transform(rearrange(embeddings, "n m k -> (n m) k"))
    groups = list(it.chain.from_iterable([label] * m for label in labels))
    viz_grouped_points(points, groups=groups)

def viz_set_distance_by_MDS(labels: List[str], samples: List[List[str]], dist: str):
    assert len(labels) == len(samples), f"|labels|={len(labels)}, |samples|={len(samples)}"

    def unimplemented(X, Y):
        raise NotImplementedError

    fn_map = {
        "hausdorff": hausdorff,
        "wasserstein": unimplemented,
        "KL_div": unimplemented,
        "chamfer": chamfer,
        "sqrt_chamfer": sqrt_chamfer,
    }
    assert dist in fn_map.keys(), f"Expected distance function in {list(fn_map.keys())}, but got {dist}"

    # embed each sample
    embeddings = np.array([embed(grp) for grp in samples])  # [n, m]

    # use distance metric on sets (or distros) to get MDS embedding
    dist_mat = dissimilarity(embeddings, dist=fn_map[dist])

    # viz MDS points
    mds = MDS(
        n_components=2,
        random_state=0,
        metric=True,
        dissimilarity='precomputed',
    )
    points = mds.fit_transform(dist_mat)
    viz_labeled_points(points, labels)
    plt.title(f"{dist} on set")

def dissimilarity(point_sets: np.ndarray, dist: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    n = len(point_sets)
    mat = np.zeros((n, n))
    for i, a in enumerate(point_sets):
        for j, b in enumerate(point_sets[i + 1:]):
            d = dist(a, b)
            mat[i, j] = d
            mat[j, i] = d
    return mat

if __name__ == "__main__":
    str_examples = [
        "$100",
        "$10,000",
        "$10b",
        "$10 billion",
        "$10,000,000,000",
        "10 TB",
        "71 MB",
        "100000000000 KB",
        "0.001 MB",
        "johndoe@mail.org",
        "ellisk@cs.cornell.edu",
        "djsl@cs.cornell.edu",
        "djl328@cornell.edu",
        "djl5@williams.edu",
        "ab1@williams.edu",
        "zz11@williams.edu",
        "iluvny@gmail.com"
        "2023/10/01",
        "2023/03/08",
        "1970/01/01",
        "hello",
        "goodbye",
        "sayonara",
        "welcome",
        "money",
        "billionaire",
    ]
    R = Regex()
    regex_examples = [R.parse(s) for s in [
        r".",
        r"\w+@gmail\.com",
        r"\w+@hotmail\.com",
        r"\w+@aol\.com",
        r"\w+@cornell\.edu",
        r"\w+@\w+.com",
        r"\w+@\w+\.\w\w\w?",
        r"$\d\d\d,\d\d\d\.\d\d",
        r"$\d,\d\d\d\.\d\d",
        r"$\d+,\d\d\d",
        r"\d,\d\d\d KRW",
        r"\d,\d\d\d dollars",
        r"\d,\d\d\d USD",
        r"\d,\d\d\d EURO",
        r"\d,\d\d\d PESO",
        r"\d,\d\d\d CAD",
    ]]
    # sort_by_instance_distances()
    # viz_instance_distance_by_MDS(str_examples)

    # take samples from each regex
    m = 10
    regex_labels = [R.to_str(x) for x in regex_examples]  # [n]
    regex_samples = [[R.eval(x, env={}) for _ in range(m)] for x in regex_examples]  # [n, m]
    viz_set_instances(labels=regex_labels, samples=regex_samples)
    plt.show()
    for dist in ["chamfer", "sqrt_chamfer", "hausdorff"]:
        viz_set_distance_by_MDS(regex_labels, regex_samples, dist=dist)
        plt.show()

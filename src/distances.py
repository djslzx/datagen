"""
Experiments with different distance metrics for probabilistic programs
"""
from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from Levenshtein import distance as levenshtein
from scipy.spatial.distance import minkowski, directed_hausdorff
from scipy.stats import wasserstein_distance as wasserstein
from einops import rearrange

from featurizers import TextClassifier
from lang import Regex, Tree
from regexpr import Regex

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

def viz_labeled_points(x: np.ndarray, y: np.ndarray, labels: List[str]):
    assert len(x.shape) == 1 and len(y.shape) == 1, \
        f"Expected 2d points but received |x|={x.shape}, |y|={y.shape}"
    n = len(x)
    assert len(labels) == n, \
        f"Points should have same cardinality as labels, but got |points|={n}, |labels|={len(labels)}"
    plt.scatter(x, y)
    for i, label in enumerate(labels):
        plt.text(x[i] + 0.01, y[i] + 0.01, label)

def viz_instance_distance_by_MDS(corpus: List[str]):
    embeddings = embed(corpus)
    mds = MDS(
        n_components=2,
        random_state=0,
    )
    points = mds.fit_transform(embeddings)
    x = points[:, 0]
    y = points[:, 1]
    viz_labeled_points(x, y, labels=corpus)

def viz_set_distance_by_MDS(labels: List[str],
                            samples: List[List[str]]):
    assert len(labels) == len(samples), f"|labels|={len(labels)}, |samples|={len(samples)}"
    # embed each sample
    embeddings = np.array([embed(grp) for grp in samples])  # [n, m]
    # n = len(labels)
    # mds = MDS(random_state=0)
    # embeddings = mds.fit_transform(rearrange(embeddings, "n m f -> (n m) f"))
    # embeddings = rearrange(embeddings, "(n m) f -> n m f", n=n)

    # use distance metric on sets (or distros) to get MDS embedding
    # 1. hausdorff
    hausdorff_mat = dissimilarity(embeddings, dist=lambda x,y: directed_hausdorff(x, y)[0])
    # 2. wasserstein
    # 3. KL div

    # viz MDS points
    mds = MDS(
        n_components=2,
        random_state=0,
        metric=False,
        dissimilarity='precomputed',
    )
    points = mds.fit_transform(hausdorff_mat)
    viz_labeled_points(points[:,0], points[:, 1], labels)

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
        r"\w+@\w+.com",
        r"\w+@\w+\.\w\w\w?",
        r"$\d\d\d,\d\d\d\.\d\d",
        r"$\d,\d\d\d\.\d\d",
        r"$\d+,\d\d\d",
        r"\d,\d\d\d KRW",
        r"\d,\d\d\d USD",
    ]]
    # sort_by_instance_distances()
    # viz_instance_distance_by_MDS(str_examples)

    # take samples from each regex
    m = 3
    regex_labels = [R.to_str(x) for x in regex_examples]  # [n]
    regex_samples = [[R.eval(x, env={}) for _ in range(m)] for x in regex_examples]  # [n, m]
    viz_instance_distance_by_MDS([x for grp in regex_samples for x in grp])
    plt.show()
    viz_set_distance_by_MDS(regex_labels, regex_samples)
    plt.show()
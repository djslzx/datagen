"""
Distances on line images produced by turtle interpretation of l-systems
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.neighbors import NearestNeighbors

import featurizers as feat
import util
from lang.lindenmayer import LSys
from ns import extract_features

Image = np.ndarray
Vec = np.ndarray

def plot_nearest_neighbors(targets: np.ndarray, guesses: np.ndarray,
                           e_targets: np.ndarray, e_guesses: np.ndarray,
                           k: int):
    """
    Plot guess images by distance from target:
    t_1  g_1^1 g_1^2 ... g_1^k
    t_2  g_2^1 g_2^2 ... g_2^k
    ...
    t_n  g_n^1 g_n^2 ... g_n^k
    """
    assert len(e_targets.shape) == 2, f"Target embeddings should be 2D, but got {e_targets.shape}"
    assert len(e_guesses.shape) == 2, f"Guess embeddings should be 2D target embeddings, but got {e_guesses.shape}"
    assert e_targets.shape[1] == e_guesses.shape[1], f"Targets and guesses must have the same embedding size"
    assert targets.shape[0] == e_targets.shape[0]
    assert guesses.shape[0] == e_guesses.shape[0]

    # find closest neighbors
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(e_guesses)  # we want the closest guesses to each target
    _, indices = knn.kneighbors(e_targets)

    # construct image list
    n_targets = len(targets)
    images = []
    for i in range(n_targets):
        target = targets[i]
        neighbors = guesses[indices[i]]
        images.append(target)
        images.extend(neighbors)

    util.plot(images, shape=(n_targets, 1 + k))


def eval_and_embed(xs: List[str]) -> Tuple[Image, Vec]:
    trees = [lang.parse(s) for s in xs]
    imgs = np.array([lang.eval(t) for t in trees])
    features = extract_features(lang, trees)
    return imgs, features


if __name__ == "__main__":
    lang = LSys(kind="deterministic", featurizer=feat.ResnetFeaturizer(), step_length=3, render_depth=3)
    target_strs = [
        "90;F;F~[+F][-F]F",
        "90;F;F~+F+F+F",
    ]
    guess_strs = [
        "15;F;F~[+F][-F]F",
        "30;F;F~[+F][-F]F",
        "45;F;F~[+F][-F]F",
        "90;F;F~[+F][-F]F",
        "15;F;F~+F+F+F",
        "30;F;F~+F+F+F",
        "45;F;F~+F+F+F",
        "90;F;F~+F+F+F"
    ]
    target_imgs, target_embeddings = eval_and_embed(target_strs)
    guess_imgs, guess_embeddings = eval_and_embed(guess_strs)

    plot_nearest_neighbors(target_imgs, guess_imgs, target_embeddings, guess_embeddings, k=len(guess_imgs))

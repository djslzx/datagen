"""
Distances on line images produced by turtle interpretation of l-systems
"""
import itertools
import os
from typing import List, Tuple
import numpy as np
import cv2 as cv
import sklearn.manifold as manifold
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from glob import glob
import Levenshtein as leven

import examples
import featurizers as feat
import util
from lang.lindenmayer import LSys
from ns import extract_features


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

    util.plot_image_grid(images, shape=(n_targets, 1 + k))


def eval_and_embed(lang: LSys, xs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    trees = [lang.parse(s) for s in xs]
    imgs = np.array([lang.eval(t) for t in trees])
    features = extract_features(lang, trees)
    return imgs, features


def rank_lsys(lsys: LSys, systems: List[str]):
    images, embeddings = eval_and_embed(lsys, systems)
    plot_nearest_neighbors(images, images, embeddings, embeddings, k=len(systems))


def generate_lsystem_pics(lsys: LSys, systems: List[str], path: str):
    """
    Generate n images from the l-system and save them to path
    """
    os.makedirs(path, exist_ok=True)
    for i, x in enumerate(systems):
        t = lsys.parse(x)
        img = lsys.eval(t)
        Image.fromarray(img).save(f"{path}/system-{i:02d}.png")


def read_pics(path: str, n_files=None) -> List[np.ndarray]:
    imgs = []
    filenames = sorted(glob(path)[:n_files])
    assert filenames, f"Got empty glob for {path}"
    for filename in filenames:
        with Image.open(filename) as im:
            img = np.array(im.resize((256, 256)))[..., :3]
            imgs.append(img)
    return imgs


def embed_pics(featurizer: feat.Featurizer, images: List[np.ndarray]) -> np.ndarray:
    embeddings = []
    for batch in util.batched(images, batch_size=16):
        if len(batch) == 1:
            batch = [batch]
        e = featurizer.apply(batch)
        embeddings.extend(e)
    return np.stack(embeddings)


def rank_pics(featurizer: feat.Featurizer, path: str, n_files=None):
    imgs = read_pics(path, n_files=n_files)
    embeddings = embed_pics(featurizer, imgs)
    imgs = np.stack(imgs)
    plot_nearest_neighbors(targets=imgs, guesses=imgs,
                           e_targets=embeddings, e_guesses=embeddings,
                           k=len(imgs))


def cluster_pics(featurizer: feat.Featurizer, path: str, n_files=None):
    imgs = read_pics(path, n_files=n_files)
    img_size = max(max(img.shape) for img in imgs)
    embeddings = embed_pics(featurizer, imgs)
    mds = manifold.MDS(n_components=2, random_state=0)
    points = mds.fit_transform(embeddings) * img_size / 2
    imgs = np.stack([util.add_border(img, thickness=1) for img in imgs])
    util.plot_images_at_positions(imgs, points)


if __name__ == "__main__":
    dir = "/Users/djsl/Documents/research/prob-repl/out/test/images"
    featurizer = feat.ResnetFeaturizer(
        disable_last_layer=True,
        softmax_outputs=False,
        sigma=0,
    )
    lsys = LSys(kind="deterministic",
                featurizer=featurizer,
                step_length=3,
                render_depth=3,
                n_rows=256,
                n_cols=256,
                vary_color=True)

    # generate_lsystem_pics(lsys,
    #                       examples.lsystem_book_det_examples,
    #                       f"{dir}/lsystems/color-256")
    # rank_lsys(lsys, examples.lsystem_book_det_examples)
    # rank_pics(featurizer, f"{dir}/lsystems/color-256/*")
    # rank_pics(featurizer, f"{dir}/natural/*")
    cluster_pics(featurizer, f"{dir}/natural/*")
    plt.show()
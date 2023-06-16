"""
Distances on line images produced by turtle interpretation of l-systems
"""
import itertools
import os
from typing import List, Tuple
import numpy as np
import cv2 as cv
import pandas as pd
from sklearn import manifold, decomposition
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from PIL import Image
from glob import glob
import Levenshtein as leven
from tqdm import tqdm

import examples
import featurizers as feat
import util
from lang.lindenmayer import LSys
from lang.tree import Tree
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
    for i in tqdm(range(n_targets)):
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


def generate_lsystem_pics_from_csv(lsys: LSys, csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    systems = (df.loc[(df.chosen == True) &
                      (df.step % 50 == 49)]
               # .sort_values(by='score', ascending=False)[:n_lsystems]
               .program)
    generate_lsystem_pics(lsys, systems, out_dir)


def generate_lsystem_pics(lsys: LSys, systems: List[str], path: str):
    """
    Generate n images from the l-system and save them to path
    """
    os.makedirs(path, exist_ok=True)
    for i, x in tqdm(enumerate(systems), total=len(systems)):
        t = lsys.parse(x)
        img = lsys.eval(t)
        Image.fromarray(img).save(f"{path}/system-{i:02d}.png")


def read_pics(path: str, n_files=None, cut_alpha=False) -> List[np.ndarray]:
    imgs = []
    filenames = sorted(glob(path)[:n_files])
    assert filenames, f"Got empty glob for {path}"
    for filename in filenames:
        with Image.open(filename) as im:
            img = np.array(im.resize((256, 256)))
            if cut_alpha:
                img = img[..., :3]
            imgs.append(img)
    return imgs


def embed_pics(featurizer: feat.Featurizer, images: List[np.ndarray]) -> np.ndarray:
    embeddings = []
    for batch in util.batched(images, batch_size=16):
        if len(batch) == 1:
            batch = [batch]
        batch = [img[..., :3] for img in batch]  # remove alpha channel
        e = featurizer.apply(batch)
        embeddings.extend(e)
    return np.stack(embeddings)


def rank_pics(featurizer: feat.Featurizer, k: int, path: str, n_files=None):
    imgs = read_pics(path, n_files=n_files)
    embeddings = embed_pics(featurizer, imgs)
    imgs = np.stack(imgs)
    plot_nearest_neighbors(targets=imgs, guesses=imgs,
                           e_targets=embeddings, e_guesses=embeddings,
                           k=k)


def rank_pics_targeted(featurizer: feat.Featurizer, k: int,
                       target_path: str, guess_path: str, n_guesses=None):
    guesses = read_pics(guess_path, n_files=n_guesses, cut_alpha=True)
    guess_embeddings = embed_pics(featurizer, guesses)
    guesses = np.stack(guesses)
    targets = read_pics(target_path, cut_alpha=True)
    target_embeddings = embed_pics(featurizer, targets)
    targets = np.stack(targets)
    plot_nearest_neighbors(targets=targets, guesses=guesses,
                           e_targets=target_embeddings, e_guesses=guess_embeddings,
                           k=k)

def mds_pics(featurizer: feat.Featurizer, path: str, n_files=None, title=None):
    imgs = read_pics(path, n_files=n_files)
    embeddings = embed_pics(featurizer, imgs)
    mds = manifold.MDS(n_components=2, random_state=0)
    points = mds.fit_transform(embeddings)
    imgs = np.stack(imgs)
    ax = util.imscatter(imgs, points, zoom=0.4, figsize=(15, 15))
    ax.title.set_text(title)


def tsne_pics(featurizer: feat.Featurizer,
              perplexity_range: List[float],
              path: str,
              n_files=None,
              save_dir=None,
              alpha=0.5):
    imgs = read_pics(path, n_files=n_files)
    embeddings = embed_pics(featurizer, imgs); print(embeddings.shape)
    embeddings = decomposition.PCA().fit_transform(embeddings); print(embeddings.shape)
    for run in tqdm(range(10)):
        for perplexity in perplexity_range:
            tsne = manifold.TSNE(
                n_components=2,
                n_iter=5000,
                perplexity=perplexity,
                n_iter_without_progress=150,
                n_jobs=2
            )
            points = tsne.fit_transform(embeddings)
            imgs = np.stack(imgs)
            ax = util.imscatter(imgs, points, zoom=0.4, alpha=alpha)
            ax.title.set_text(f"perplexity={perplexity}, run={run}")

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f"{save_dir}/perp{perplexity}_run{run}.png")
                plt.close()
            else:
                plt.show()


def measure_dataset_dist(target_path: str, guess_path: str,
                         disable_last_layer, softmax_outputs, center, sigma, vary_color):
    lsys = LSys(kind="deterministic",
                featurizer=feat.ResnetFeaturizer(
                    disable_last_layer=disable_last_layer,
                    softmax_outputs=softmax_outputs,
                    center=center,
                    sigma=sigma,
                ),
                step_length=3,
                render_depth=3,
                n_rows=256,
                n_cols=256,
                vary_color=vary_color)
    # for run_id in [
    #     "xgtuf1f7",
    #     "lbiu7veh",
    #     "jboe4u9c",
    #     "69elsujn",
    #     "i9vplkpx",
    # ]:
    #     csv_file = f"{root}/out/sweeps/2a5p4beb/{run_id}.csv"
    #     out_dir = f"{dir}/lsystems/generated-color-256/{run_id}"
    #     generate_lsystem_pics_from_csv(lsys,
    #                                    csv_path=csv_file,
    #                                    out_dir=out_dir)
    rank_pics_targeted(lsys.featurizer, k=5, target_path=target_path, guess_path=guess_path)
    plt.show()
    # mds_pics(lsys.featurizer, path=target_path, title="target")
    # plt.show()
    # if target_path != guess_path:
    #     mds_pics(lsys.featurizer, path=guess_path, title="guess")
    #     plt.show()


if __name__ == "__main__":
    # set matplotlib to dark
    plt.style.use('dark_background')
    root = "/Users/djsl/Documents/research/prob-repl"
    dir = f"{root}/out/test/images"
    # generate_lsystem_pics(lsys,
    #                       examples.lsystem_book_det_examples,
    #                       f"{dir}/lsystems/rgba-256")
    # for disable_last_layer, softmax_outputs, center in itertools.product([False, True,],
    #                                                                      [False, True,],
    #                                                                      [False, True,]):
    # measure_dataset_dist(target_path=f"{dir}/natural/*",
    #                      guess_path=f"{dir}/natural/*",
    #                      disable_last_layer=True,
    #                      softmax_outputs=False,
    #                      center=False,
    #                      sigma=0,
    #                      vary_color=True)
    measure_dataset_dist(target_path=f"{dir}/lsystems/rgba-256/*.png",
                         guess_path=f"{dir}/lsystems/generated-color-256/system-*.png",
                         disable_last_layer=True,
                         softmax_outputs=False,
                         center=False,
                         sigma=0,
                         vary_color=True)

    # rank_pics(lsys.featurizer, k=5, path=f"{dir}/lsystems/color-256/*.png")
    # tsne_pics(lsys.featurizer,
    #           perplexity_range=[2, 4, 6, 8],
    #           path=f"{dir}/natural/*",
    #           save_dir=f"{root}/out/tsne/natural/"
    #                    f"cut_last={disable_last_layer}_softmax={softmax_outputs}_center={center}/")
    # plt.show()

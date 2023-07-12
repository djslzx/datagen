import os
import sys
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import featurizers as feat
import util

tqdm.pandas()


def embed_instructions(df: pd.DataFrame, ft: feat.Featurizer, outdir: str, batch_size=10):
    """
    Embed instructions using the given featurizer
    """
    print("Embedding instructions...", file=sys.stderr)
    all_embeddings = []
    for batch in tqdm(util.batched(df["instruction"], batch_size), total=len(df)//batch_size):
        embeddings = ft.apply(batch)
        all_embeddings.extend(embeddings)
    np_embeddings = np.array(all_embeddings)
    np.save(f"{outdir}/embeddings.npy", np_embeddings)
    return np_embeddings

def write_dists(embeddings: np.ndarray, outdir: str):
    """
    Compute pairwise distances between embeddings of rows
    """
    print("Computing distances...", file=sys.stderr)
    rows = []
    for i, e1 in tqdm(enumerate(embeddings), total=len(embeddings)):
        for j, e2 in enumerate(embeddings[i+1:]):
            rows.append({"i": i, "j": j, "dist": np.linalg.norm(e1 - e2)})
    dists = pd.DataFrame(rows)
    dists.to_csv(f"{outdir}/dists.csv", index=False, float_format="%.4f")
    return dists


def dists_to_histogram(df: pd.DataFrame, outdir: str):
    """
    Convert pairwise distances to a histogram
    """
    plt.hist(df["dist"])
    plt.show()


if __name__ == "__main__":
    with open("../datasets/evol_teacher_80k.json", "r") as f:
        data = json.load(f)[:1000]
        df = pd.DataFrame(data)
    ft = feat.SentenceFeaturizer()

    os.makedirs("../out/dists/evol_teacher_80k", exist_ok=True)
    embed_instructions(df, ft, "../out/dists/evol_teacher_80k")
    embeddings = np.load("../out/dists/evol_teacher_80k/embeddings.npy")

    dists = write_dists(embeddings, "../out/dists/evol_teacher_80k")
    plt.hist(dists["dist"])
    plt.show()

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


def embed_instructions(df: pd.DataFrame, ft: feat.Featurizer, outdir: str):
    """
    Embed instructions using the given featurizer
    """
    print("Embedding instructions...", file=sys.stderr)
    df["embedding"] = df["instruction"].progress_apply(ft.apply)
    df.to_csv(f"{outdir}/embeddings.csv", index=False)
    return df

def write_dists(df: pd.DataFrame, outdir: str):
    """
    Compute pairwise distances between embeddings of rows
    """
    print("Computing distances...", file=sys.stderr)
    rows = []
    for i, r1 in tqdm(df.iterrows(), total=len(df)):
        for j, r2 in df.iterrows():
            if i < j:
                rows.append({"i": i, "j": j, "dist": np.linalg.norm(r1["embedding"] - r2["embedding"])})
    dists = pd.DataFrame(rows)
    dists.to_csv(f"{outdir}/dists.csv", index=False)
    return dists


def dists_to_histogram(df: pd.DataFrame, outdir: str):
    """
    Convert pairwise distances to a histogram
    """
    plt.hist(df["dist"])
    plt.show()

if __name__ == "__main__":
    # with open("../datasets/evol_teacher_80k.json", "r") as f:
    #     data = json.load(f)
    #     df = pd.DataFrame(data)
    ft = feat.SentenceFeaturizer()

    with open("../out/dists/evol_teacher_80k/embeddings.csv", "r") as f:
        embedding_data = json.load(f)

    embedding_df = pd.DataFrame(embedding_data)
    print(embedding_df).head()

    os.makedirs("../out/dists/evol_teacher_80k", exist_ok=True)
    write_dists(embedding_df, "../out/dists/evol_teacher_80k/")

from typing import List, Dict, Set, Iterator
import itertools as it


PROJECT_DIR = "/home/djl328/prob-repl"
DATASETS = [
    "NSCA",
    "NSE",
    "WW",
    "WD",
    "CA",
]


def tabprint(file: str, *xs: List[str]):
    print("\t".join(str(x) for x in xs), file=file)


def make_cfg(header: List[str], xs: Iterator, out_file: str):
    with open(out_file, "w") as f:
        tabprint(f, *header)
        for i, x in enumerate(xs, 1):
            try:
                tabprint(f, i, *x)
            except TypeError:
                tabprint(f, i, x)


def eval_cfg(n_chunks: int, out_file: str):
    files = [
        f"{dataset}/chunk-{i_chunk:04d}.jsonl"
        for i in range(n_chunks)
        for dataset in DATASETS
    ]
    make_cfg(["ID", "Dataset"], files, out_file)


def rollout_cfg(model_file: str, out_file: str):
    with open(model_file, "r") as f:
        models = [
            line.strip() 
            for line in f.readlines()
            if not line.strip().startswith("#")
        ]
    datasets = [
        f"{PROJECT_DIR}/datasets/wiz/hf-20:30k/{dataset}"
        for dataset in DATASETS
    ]
    make_cfg(["ID", "Dataset", "Model"], 
             it.product(datasets, models),
             out_file=out_file)


if __name__ == "__main__":
    # eval_cfg(n_chunks=100, out_file="evaluate/n100.cfg")
    rollout_cfg(model_file="rollout/models.txt", out_file="rollout/all.cfg")
    rollout_cfg(model_file="rollout/tiny-models.txt", out_file="rollout/tiny.cfg")

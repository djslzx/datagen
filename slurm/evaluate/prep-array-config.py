from typing import List, Dict, Set, Iterator
import argparse
import itertools as it


def tabprint(*xs: List[str], file: str):
    print("\t".join(str(x) for x in xs), file=file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int)
    p.add_argument("o", type=str)
    args = p.parse_args()

    datasets = [
        "NSCA",
        "NSE",
        "WW",
        "WD",
        "CA",
    ]

    with open(args.o, "w") as f:
        tabprint("ID", "Dataset", file=f)
        for i, (dataset, i_chunk) in enumerate(it.product(datasets, range(args.n)), 1):
            tabprint(i, f"{dataset}/chunk-{i_chunk:04d}.jsonl", file=f)

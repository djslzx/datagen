"""
Dataset handling:
- given file/id, fetch problems, solutions, and tests
"""
from typing import Dict, List, Optional, Tuple
import sys
import pandas as pd
from datasets import Dataset, DatasetDict, DatasetInfo
from tqdm import tqdm


def clean_solution_dataset(
        in_file: str,
        out_dir: str,
        name_map: Dict[str, str] = None,
        n_solns: int = 3,
        n_tests: int = 3,
):
    """
    Clean up datasets consisting of problems, solutions, and checkers.
    - pivot from kv to columnar form
    - rename datasets, e.g. CA-20k to CA
    - extract source from id
    - add columns for problems, solutions, and tests
    - split into separate output files by source
    - shuffle dataset
    """
    assert in_file.endswith(".jsonl"), f"Expected jsonl file, but got in_file={in_file}"

    if not name_map:
        name_map = {
            "CA-20k": "CA",
            "NS-euler": "NSE",
            "NS": "NSCA",
            "Wiz-deep": "WD",
            "Wiz-wide": "WW",
        }

    problem_keys = ["restyled problem"]
    soln_keys = [f"solution {i}" for i in range(n_solns)]
    test_keys = [f"test {i}" for i in range(n_tests)]

    def rename(s_id: str) -> str:
        s_src, s_num = s_id.split(":")
        s_src = name_map[s_src] if s_src in name_map else s_src
        return f"{s_src}:{s_num}"

    df = pd.read_json(in_file, lines=True)
    df = df[["id", "key", "value"]]
    df["id"] = df["id"].apply(rename)
    keys = ["id"] + problem_keys + soln_keys + test_keys
    df = df[df["key"].isin(keys)]
    df = df.drop_duplicates(subset=["id", "key"], keep="first")

    df = df.pivot(index="id", columns="key", values="value")
    df = df.where(pd.notnull(df), None)
    df["source"] = df.index.map(lambda x: x.split(":")[0])

    # fixme: for now, we make the simplifying assumption that all solutions
    #   are good, so use any problem/solution pair to fine-tune

    # shuffle data
    df = df.sample(frac=1)

    # split each source file into its own dataset
    for source in sorted(df["source"].unique()):
        data = df[df["source"] == source]
        rows = []
        print(f"Found {len(data)} lines in {source}, processing...", file=sys.stderr)
        for id, row in tqdm(data.iterrows(), total=len(data), desc=f"Massaging {source}"):
            for i, soln in enumerate(soln_keys):
                tests = [row[t] for t in test_keys]
                if row[soln]:
                    rows.append({
                        "id": f"{id}:{i}",
                        "source": source,
                        "problem": row["restyled problem"],
                        "solution": row[soln],
                        "tests": tests,
                    })
        ds = Dataset.from_pandas(
            pd.DataFrame.from_records(rows),
            info=DatasetInfo(
                dataset_name=source,
                description=f"{source} dataset",
            ),
        )
        tt = ds.train_test_split(test_size=0.2)
        vt = tt["test"].train_test_split(test_size=0.5)
        dd = DatasetDict({
            "train": tt["train"],
            "validation": vt["train"],
            "test": vt["test"],
        })
        dd.save_to_disk(f"{out_dir}/{source}")

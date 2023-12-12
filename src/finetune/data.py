"""
Dataset handling:
- given file/id, fetch problems, solutions, and tests
"""
import pdb
from typing import Dict, List, Optional, Tuple, Iterator
import sys
import pandas as pd
from datasets import Dataset, DatasetDict, DatasetInfo
from tqdm import tqdm
from joblib import Parallel, delayed

from root import evaluate
from root import util


def read_long_dataset_to_wide_df(
        filename: str, 
        name_map: Dict[str, str] = None, 
        n_solns: int = 3, 
        n_tests: int = 3,
) -> pd.DataFrame:
    """
    Clean up datasets consisting of problems, solutions, and checkers and return a dataframe keyed
    by id, with columns "problem", "solutions", and "tests".

    Transformations:
    - pivot from kv to columnar form
    - rename datasets, e.g. CA-20k to CA
    - extract source from id
    - group solns, tests into lists
    """
    assert filename.endswith(".jsonl"), f"Expected jsonl file, but got in_file={filename}"

    if not name_map:
        name_map = {
            "CA-20k": "CA",
            "NS-euler": "NSE",
            "NS": "NSCA",
            "Wiz-deep": "WD",
            "Wiz-wide": "WW",
        }

    def rename(s_id: str) -> str:
        s_src, s_num = s_id.split(":")
        s_src = name_map[s_src] if s_src in name_map else s_src
        return f"{s_src}:{s_num}"

    df = pd.read_json(filename, lines=True)
    df = df[["id", "key", "value"]]
    df["id"] = df["id"].apply(rename)
    df = df.drop_duplicates(subset=["id", "key"], keep="first")
    df = df.pivot(index="id", columns="key", values="value")
    df = df.where(pd.notnull(df), None)
    df["source"] = df.index.map(lambda x: x.split(":")[0])
    df = df.rename(columns={"restyled problem": "problem"})

    # keep only n_tests tests and n_solns solns
    soln_keys = [f"solution {i}" for i in range(n_solns)]
    test_keys = [f"test {i}" for i in range(n_tests)]
    df["tests"] = df.apply(lambda row: [row[k] for k in test_keys if row[k]], axis=1)
    df["solutions"] = df.apply(lambda row: [row[k] for k in soln_keys if row[k]], axis=1)
    df = df[["problem", "solutions", "tests"]]

    # remove problems with empty solutions or empty tests
    df = df[df["solutions"].apply(lambda x: len(x) > 0) &
            df["tests"].apply(lambda x: len(x) > 0)]

    return df


def prepare_hf_dataset(
        filename: str, 
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
    df = read_long_dataset_to_wide_df(
        filename=filename, 
        name_map=name_map, 
        n_solns=n_solns, 
        n_tests=n_tests
    )

    # fixme: for now, we make the simplifying assumption that all solutions
    #   are good, so use any problem/solution pair to fine-tune
    # fixme: we also assume that all tests are good

    # shuffle data
    df = df.sample(frac=1)

    # split each source file into its own dataset
    for source in sorted(df["source"].unique()):
        data = df[df["source"] == source]
        rows = []
        print(f"Found {len(data)} lines in {source}, processing...", file=sys.stderr)
        for id, row in tqdm(data.iterrows(), total=len(data), desc=f"Massaging {source}"):
            problem = row["problem"]
            tests = row["tests"]
            for i, soln in enumerate(row["solutions"]):
                rows.append({
                    "id": f"{id}:{i}",
                    "source": source,
                    "problem": problem,
                    "solution": soln,
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


def fetch_good_solns_and_tests(filename: str, source: str, n_solns: int = 3, n_tests: int = 3,
                               timeout=1) -> pd.DataFrame:
    """
    Filter out all "bad" solutions and tests.
    """
    pass


def run_solns_and_tests(df: pd.DataFrame, timeout: float) -> Iterator[dict]:
    # Run solutions in isolation
    for id, row in tqdm(df.iterrows(), total=len(df), desc="Running solutions in isolation"):
        problem = row["problem"]
        for i, soln in enumerate(row["solutions"]):
            result = evaluate.run_soln(soln, timeout)
            out = {
                "id": f"{id}:{i}",
                "problem": problem,
                "solution": soln,
                "test": None,
                **result.to_dict(prefix="result."),
            }
            for item in util.KVItem.from_dict(out):
                yield item.to_dict()

    # Run solutions with tests
    for id, row in tqdm(df.iterrows(), total=len(df), desc="Running solutions with tests"):
        problem = row["problem"]
        for i, soln in enumerate(row["solutions"]):
            for j, test in enumerate(row["tests"]):
                result = evaluate.run_soln_w_test(soln, test, timeout)
                out = {
                    "id": f"{id}:{i}:{j}",
                    "problem": problem,
                    "solution": soln,
                    "test": test,
                    **result.to_dict(prefix="result."),
                }
                for item in util.KVItem.from_dict(out):
                    yield item.to_dict()


def run_soln(ident: str, problem: str, soln: str, timeout: float) -> dict:
    result = evaluate.run_soln(soln, timeout)
    return {
        "id": ident,
        "problem": problem,
        "solution": soln,
        "test": None,
        **result.to_dict(prefix="result."),
    }


def run_soln_and_test(ident: str, problem: str, soln: str, test: str, timeout: float) -> dict:
    result = evaluate.run_soln_w_test(soln, test, timeout)
    return {
        "id": ident,
        "problem": problem,
        "solution": soln,
        "test": test,
        **result.to_dict(prefix="result."),
    }


def mp_run_solns_and_tests(df: pd.DataFrame, timeout: float, n_jobs: int = -1) -> Iterator[dict]:
    parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing")
    soln_runner = parallel(
        delayed(run_soln)(f"{ident}:{i}", row["problem"], soln, timeout)
        for ident, row in tqdm(df.iterrows(), total=len(df), desc="Running solutions in isolation")
        for i, soln in enumerate(row["solutions"])
    )
    for result in tqdm(soln_runner):
        for item in util.KVItem.from_dict(result):
            yield item.to_dict()

    test_runner = parallel(
        delayed(run_soln_and_test)(f"{ident}:{i}:{j}", row["problem"], soln, test, timeout)
        for ident, row in tqdm(df.iterrows(), total=len(df), desc="Running solutions with tests")
        for i, soln in enumerate(row["solutions"])
        for j, test in enumerate(row["tests"])
    )
    for result in tqdm(test_runner):
        for item in util.KVItem.from_dict(result):
            yield item.to_dict()


def pull_test_keys(dirname: str, children=List[str]) -> Dict[str, List[str]]:
    keys = {}
    for c in children:
        dataset = DatasetDict.load_from_disk(f"{dirname}/{c}")
        keys[c] = dataset['test']['id']
    return keys


if __name__ == "__main__":
    project_dir = "/home/djl328/prob-repl"
    small_ds = "solved/all-solved-1k.jsonl"
    full_ds = "all-solved/all-solved-20k:30k.jsonl"

    df = read_long_dataset_to_wide_df(filename=f"{project_dir}/datasets/wiz/{full_ds}")    
    df = df.head(100)
    ts = util.timestamp()
    util.incrementally_save_jsonl(
        quiet=True,
        filename=f"{project_dir}/datasets/wiz/eval-mptest-{ts}",
        # filename=f"{project_dir}/datasets/eval-all-20k:30k-{ts}",
        it=mp_run_solns_and_tests(df, timeout=30, n_jobs=-1),
    )

    # keys = pull_test_keys(dirname="../datasets/wiz/hf-20:30k/", children=["NSCA", "NSE", "WD", "WW", "CA"])
    # print(f"Collected keys:")
    # for k, vs in keys.items():
    #     print(f"  {k}: {len(vs)}")

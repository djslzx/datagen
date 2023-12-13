from typing import Dict, List
import pandas as pd
import numpy as np
from datasets import Dataset

from root import evaluate
import finetune.data as data
import finetune.models as models


def filter_solns_and_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out solutions that don't pass tests
    Filter out tests that don't permit solutions
    Filter out rows with no remaining solutions or tests
    """
    raise NotImplementedError


if __name__ == "__main__":
    df = data.fetch_solns_and_tests("../../datasets/wiz/all-solved/all-solved-20k:30k.jsonl", source="NSCA")
    print(len(df))
    df = filter_solns_and_tests(df)
    print(len(df))
    print(df)

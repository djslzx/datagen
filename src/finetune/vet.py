"""
Vet solutions and tests
"""

import pandas as pd
from pandas import Series


def soln_test_id_to_soln_id(ident: str) -> str:
    # soln-test id is of the form i:j, whereas test id is of the form j,
    #  so we remove the last colon-separated bit to get the soln id
    return ":".join(ident.split(":")[:-1])


def stable_solns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract stable solutions, i.e. solutions that don't bug out when run alone
    """
    assert "run-type" in df

    def is_stable_soln(row: dict) -> bool:
        return row["run-type"] == "soln-only" and row["result.passed"]

    return df[df.apply(is_stable_soln, axis=1)]


def stable_tests(df: pd.DataFrame, stable_solns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract stable tests wrt stable solutions: tests that don't bug out when run with stable tests,
     i.e. tests that either pass or fail (but don't crash) on stable solutions
    """
    assert "run-type" in df

    def is_stable_test(row: dict) -> bool:
        return row["run-type"] == "soln-and-test" and \
            soln_test_id_to_soln_id(row["id"]) in stable_solns_df.index and \
            (
                    row["result.passed"] or
                    row["result.exception_type"] == "TestFailed"
            )

    return df[df.apply(is_stable_test, axis=1)]


def test_passing_solns(df: pd.DataFrame, stable_solns_df: pd.DataFrame, stable_tests_df: pd.DataFrame) -> Series:
    """
    Extract stable solutions that pass all stable tests
    """
    assert "run-type" in df

    df["soln-id"] = df.apply(
        lambda row: row["id"] if row["run-type"] == "soln-only" else soln_test_id_to_soln_id(row["id"]),
        axis=1
    )

    def passes_all_stable_tests(group: pd.DataFrame) -> bool:
        return group.apply(
            lambda row: row["result.passed"] or row["id"] in stable_tests_df.index,
            axis=1,
        ).all(axis=0)

    # solution-test pairs where solution is stable
    stable_soln_pairs = (df[
        (df["soln-id"].isin(stable_solns_df.index)) &
        (df["run-type"] == "soln-and-test")
        ])
    # pairs where all stable tests pass
    stable_passers = stable_soln_pairs.groupby("soln-id").apply(passes_all_stable_tests)

    return stable_passers


def largest_consensus_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each problem's solution and test sets X, Y, extract the largest consensus set.
    """
    raise NotImplementedError


def filter_solutions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts stable solutions in `df` that pass stable tests.
    The dataframe `df` should contain a record of evaluation solution-test pairs.

    Returns a dataframe containing filtered solutions.
    """
    df["run-type"] = df["test"].apply(lambda x: "soln-only" if x is None else "soln-and-test")

    all_solns = df[df["run-type"] == "soln-only"]
    all_tests = df[df["run-type"] == "soln-and-test"]

    stable_solns_df = stable_solns(df)
    stable_tests_df = stable_tests(df, stable_solns_df)
    passing_solns_df = test_passing_solns(df, stable_solns_df, stable_tests_df)

    # todo: check that passing_solns_df keys are a subset of stable_solns_df

    n_solns = len(all_solns)
    n_tests = len(all_tests)
    n_stable_solns = len(stable_solns_df)
    n_stable_tests = len(stable_tests_df)
    n_passing_solns = len(passing_solns_df)

    print(df[df["run-type"] == "soln-only"]["result.exception_type"].value_counts())
    print(f"total lines: {len(df)}")
    print(f"stable solns: {n_stable_solns} / {n_solns} ({n_stable_solns / n_solns})")
    print(f"stable tests | stable solns: {n_stable_tests} / {n_tests} ({n_stable_tests / n_tests})")
    print(f"stable solns passing stable tests: {n_passing_solns} / {n_solns} ({n_passing_solns / n_solns})")

    # get solutions from original df keyed by filtered solutions
    return df.loc[passing_solns_df[passing_solns_df].index]

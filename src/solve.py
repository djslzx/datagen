from langchain.chat_models import ChatOpenAI
import sys

import analyze


if __name__ == "__main__":
    CHAT = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
    FILES = {
        "NS": "../datasets/wiz/novel-instruct",
        "NS-euler": "../datasets/wiz/novel-instruct-euler",
        "Wiz-wide": "../datasets/wiz/wiz-wide",
        "Wiz-deep": "../datasets/wiz/wiz-deep",
        "CA-20k": "../datasets/wiz/code-alpaca",
    }

    # load files
    df = analyze.read_problems(FILES)
    df["id"] = df.apply(lambda row: f"{row['source file']}:{row['id']}", axis=1)
    df["n"] = df["id"].apply(lambda x: int(x.split(":")[1]))

    # filter by cli args
    args = sys.argv[1:]
    assert len(args) == 2, f"Expected (name, n) but got {args}"
    print(f"Proceeding with args {args}")

    source_file, n = args
    n = int(n)
    df = df[
        (df["source file"] == source_file) & 
        (df["n"] > n)
    ]
    source_files = df["source file"].unique()
    min_entry = df["n"].min()
    print("Processing source files {source_files} with minimum entry {min_entry}")

    print(df)

    # remove annotations before processing
    df.drop(columns="n", inplace=True)

    # annotate problems w/ solns, tests
    analyze.gen_solns_and_tests(CHAT, df, n_samples=None)


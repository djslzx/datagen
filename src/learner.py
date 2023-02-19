from __future__ import annotations

import lark
import torch as T
import lightning as pl
import torch.utils.data as Tdata
from typing import *
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from grammar import Grammar, LearnedGrammar, ConvFeatureExtractor
from lindenmayer import S0LSystem
from evo import DRAW_ARGS
from zoo import zoo
import parse
import util


def eval_ttree_as_lsys(p: Tuple, level=3):
    sys_str = parse.eval_ttree_as_str(p)
    sys = S0LSystem.from_sentence(list(sys_str))
    render_str = sys.nth_expansion(level)
    return S0LSystem.draw(render_str, **DRAW_ARGS)


class LSystemDataset(Tdata.Dataset):
    """
    Reads in L-system strings and yields ASTs.
    """

    @staticmethod
    def from_files(filenames: List[str]) -> "LSystemDataset":
        lines = []
        for filename in filenames:
            with open(filename, "r") as f:
                for line in f.readlines():
                    if line.startswith("#"):  # skip comments
                        continue
                    if ":" in line:  # split out scores
                        line = line.split(" : ")[0]
                    # confirm that line is parseable
                    # TODO: restructure so we don't parse twice
                    try:
                        parse_str_to_tuple(line)
                        lines.append(line.strip())
                    except (lark.UnexpectedCharacters,
                            lark.UnexpectedToken):
                        pass
        return LSystemDataset(data=lines)

    def __init__(self, data: List[str]):
        # TODO: allow a program to generate multiple outputs (probabilistic programs)
        super(LSystemDataset).__init__()
        self.data = data

    def __getitem__(self, item):
        s = self.data[item]
        ast = parse_str_to_tuple(s)
        return s, eval_ttree_as_lsys(ast)

    def __len__(self):
        return len(self.data)


def parse_str_to_tuple(s: str) -> Tuple:
    """Parses an l-system from an l-system string s"""
    ltree = parse.parse_lsys_as_ltree(s)
    return parse.ltree_to_ttree(ltree)


def simplify_file(in_path: str, out_path: str, score_thresh=None):
    print(f"Writing simplified file to {out_path}")
    n_parse_failures, n_low_score = 0, 0
    with open(in_path, 'r') as f_in, open(out_path, 'w') as f_out:
        for i, line in enumerate(f_in.readlines()):
            if line.startswith("#"):  # skip comments
                f_out.write(line)
                continue
            if ":" in line:  # split out scores
                line, score = line.split(" : ")
                if score_thresh is not None:
                    # skip lines with low score
                    score = float(score.replace("*", ""))
                    if score <= score_thresh:
                        print(f"Skipping line {i} because of low score: {score}")
                        f_out.write("\n")
                        n_low_score += 1
                        continue
            # simplify line
            try:
                s = parse.simplify(line)
                print(f"{i}: {s}")
                f_out.write(s + "\n")
            except (lark.UnexpectedCharacters, lark.UnexpectedToken, parse.ParseError):
                print(f"Skipping line {i}")
                f_out.write("\n")
                n_parse_failures += 1
    print(f"Skipped {n_parse_failures} lines b/c of parsing failure,\n"
          f"        {n_low_score} lines b/c of low score (< 0.001)")


def summary_stats(arr: np.ndarray) -> str:
    return f"in: [mean: {np.mean(arr)}, std: {np.std(arr)}, max: {np.max(arr)}, min: {np.min(arr)}]"


def plot_egg_scatter(x: List[str], y: List[str]):
    xs = np.array([len(s) for s in x], dtype=int)
    ys = np.array([len(s) for s in y], dtype=int)

    # count up (x,y) pairs
    counts = {}
    for x, y in zip(xs, ys):
        counts[x, y] = counts.get((x, y), 0) + 1
    data = np.array([[x, y, n] for (x, y), n in counts.items()])

    # diagonal guide line
    ax = plt.gca()
    x_max = np.max(xs)
    ax.plot([0, x_max], [0, x_max], color='gray', markersize=1)

    # plot lengths
    sns.scatterplot(x=data[:, 0], y=data[:, 1], size=data[:, 2], legend=False, sizes=(10, 1000))
    plt.show()


def pluck_egg_examples(x: List[str], y: List[str], k=10):
    n = len(x)
    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)
    x_lens = np.array([len(s) for s in x], dtype=int)
    y_lens = np.array([len(s) for s in y], dtype=int)

    # remove lines that were empty after simplification (i.e. invalid lines) or which weren't simplified at all
    filter_idx = np.nonzero(y_lens * (x_lens - y_lens))[0]
    x = x[filter_idx]
    y = y[filter_idx]
    x_lens = x_lens[filter_idx]
    y_lens = y_lens[filter_idx]

    def show_progs(indices: List[int] | np.ndarray):
        for i in indices:
            p_x, p_y, n = x[i], y[i], (x_lens[i] - y_lens[i]).item()
            print(f"{n} tokens removed: {p_x} => {p_y} ({i})")

    print(f"Filtered {len(filter_idx)} programs out of {n}.")
    idx = np.argsort(-(x_lens - y_lens))  # sort by reduction in length from egg
    print("Max:")
    show_progs(idx[:k])
    print("Median:")
    m = len(idx) // 2
    show_progs(idx[m:m + k])
    print("Min:")
    show_progs(idx[-10:])


def plot_egg_cdf(x: List[str], y: List[str]):
    x_lens = np.array([len(s) for s in x], dtype=int)
    y_lens = np.array([len(s) for s in y], dtype=int)
    sns.displot(x=x_lens - y_lens, kind="ecdf")
    plt.show()


def plot_egg_pdf(x: List[str], y: List[str]):
    x_lens = np.array([len(s) for s in x], dtype=int)
    y_lens = np.array([len(s) for s in y], dtype=int)
    sns.displot(x=x_lens - y_lens, kind="kde")
    plt.show()


def read_files(in_file: str, out_file: str) -> Tuple[List[str], List[str]]:
    x, y = [], []
    with open(in_file, 'r') as f_in, open(out_file, 'r') as f_out:
        for f, arr in [(f_in, x), (f_out, y)]:
            for line in f.readlines():
                if line.strip().startswith("#"):
                    continue
                if ":" in line:
                    arr.append(line.split(" : ")[0])
                else:
                    arr.append(line.strip())

    return x, y


def lg_kwargs():
    g = Grammar.from_components(components=parse.rule_types, gram=2)
    fe = ConvFeatureExtractor(n_features=1000,
                              n_color_channels=1,
                              n_conv_channels=12,
                              bitmap_n_rows=128,
                              bitmap_n_cols=128)
    return {
        "feature_extractor": fe,
        "grammar": g,
        "parse": parse_str_to_tuple,
        "start_symbol": "LSystem",
        "learning_rate": 0.001,
    }


def train_learner(train_filenames: List[str], epochs: int):
    lg = LearnedGrammar(**lg_kwargs())
    train_dataset = LSystemDataset.from_files(train_filenames)
    train_loader = Tdata.DataLoader(train_dataset, shuffle=True)
    val_dataset = LSystemDataset([sys.to_str() for sys in zoo])
    val_loader = Tdata.DataLoader(val_dataset)
    trainer = pl.Trainer(max_epochs=epochs, auto_lr_find=True)
    trainer.tune(model=lg, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.fit(model=lg, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Untrained grammar")
    print(lg.original_grammar)
    print("Trained grammar")
    print(lg.grammar)


def load_learned_grammar(checkpt_path: str) -> LearnedGrammar:
    ckpt = T.load(checkpt_path)
    weights = ckpt["grammar_params"]
    lg = LearnedGrammar.load_from_checkpoint(checkpoint_path=checkpt_path, **lg_kwargs())
    lg.grammar.from_tensor_(weights)
    return lg


def compare_models(model1_chkpt: str, model2_chkpt: str, datasets: List[str]):
    model1 = load_learned_grammar(model1_chkpt)
    model2 = load_learned_grammar(model2_chkpt)
    model1.eval()
    model2.eval()
    dataset = LSystemDataset.from_files(datasets)
    dataloader = Tdata.DataLoader(dataset)
    for (x,), y in dataloader:
        ttree = parse_str_to_tuple(x)
        print(f"{x}\n"
              f"  model1 loss: {-model1.grammar.log_probability(model1.start_symbol, ttree)}\n"
              f"  model2 loss: {-model2.grammar.log_probability(model2.start_symbol, ttree)}")


if __name__ == "__main__":
    def usage():
        print("Usage: learner.py train|compare|simplify|egg_scatter|egg_cdf|egg_pdf|egg_pluck *args")
        exit(1)

    if len(sys.argv) <= 1:
        usage()

    choice, *args = sys.argv[1:]
    if choice == "train":
        assert len(args) == 2, "Usage: learner.py train DATASETS EPOCHS"
        train_glob, epochs = args
        train_filenames = sorted(glob(train_glob))
        epochs = int(epochs)
        train_learner(train_filenames, epochs)

    elif choice == "compare":
        assert len(args) == 3, "Usage: learner.py compare MODEL1 MODEL2 DATASETS"
        model1_path, model2_path, datasets_glob = args
        dataset_paths = sorted(glob(datasets_glob))
        compare_models(model1_path, model2_path, dataset_paths)

    elif choice == "simplify":
        assert len(args) in [2, 3], "Usage: learner.py simplify IN_PATH OUT_PATH [THRESH]"
        if len(args) == 2:
            in_path, out_path = args
            thresh = None
        else:
            in_path, out_path, thresh = args
            thresh = float(thresh)
        simplify_file(in_path, out_path, thresh)

    # Show egg stats

    elif choice == "egg_scatter":
        assert len(args) == 2, "Usage: learner.py egg_scatter PATH1 PATH2"
        x, y = read_files(*args)
        plot_egg_scatter(x, y)

    elif choice == "egg_cdf":
        assert len(args) == 2, "Usage: learner.py egg_cdf PATH1 PATH2"
        x, y = read_files(*args)
        plot_egg_cdf(x, y)

    elif choice == "egg_pdf":
        assert len(args) == 2, "Usage: learner.py egg_pdf PATH1 PATH2"
        x, y = read_files(*args)
        plot_egg_pdf(x, y)

    elif choice == "egg_pluck":
        assert len(args) == 2, "Usage: learner.py egg_pluck PATH1 PATH2"
        x, y = read_files(*args)
        pluck_egg_examples(x, y)

    else:
        usage()

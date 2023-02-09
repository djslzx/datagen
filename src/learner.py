from __future__ import annotations

import lark
import lightning as L
import torch.utils.data as Tdata
from typing import *
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from grammar import Grammar, LearnedGrammar, ConvFeatureExtractor
from lindenmayer import S0LSystem
from evo import DRAW_ARGS
from zoo import zoo
import parse
import util

Tree: TypeAlias = lark.Tree


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


def simplify_file(fname: str):
    basename = util.cut_ext(fname)
    out_fname = f"{basename}-simpl.txt"
    n_in, n_out = 0, 0
    print(f"Writing simplified file to {out_fname}")
    with open(fname, 'r') as f_in, open(out_fname, 'w') as f_out:
        for i, line in enumerate(f_in.readlines()):
            n_in += 1
            if line.startswith("#"):  # skip comments
                f_out.write(line)
                n_out += 1
                continue
            if ":" in line:  # split out scores
                line = line.split(" : ")[0]
            # simplify line
            try:
                s = parse.simplify(line)
                print(f"{i}: {s}")
                f_out.write(s + "\n")
                n_out += 1
            except (lark.UnexpectedCharacters, lark.UnexpectedToken, parse.ParseError):
                print(f"Skipping line {i}")
                f_out.write("\n")
    print(f"Wrote {n_out} out of {n_in} lines")


def check_compression(in_file: str, out_file: str, n_lines: int):
    # in-file # lines, out-file # lines
    mat = np.empty((n_lines, 2), dtype=int)
    with open(in_file, 'r') as f_in, open(out_file, 'r') as f_out:
        for i, line in enumerate(f_in.readlines()):
            mat[i, 0] = len(line)
        for i, line in enumerate(f_out.readlines()):
            mat[i, 1] = len(line)

    # print n_lines stats
    print(f"in_file mean: {np.mean(mat[:, 0])}, "
          f"std dev: {np.std(mat[:, 0])}, "
          f"out_file mean: {np.mean(mat[:, 1])}, "
          f"std dev: {np.std(mat[:, 1])}, ")

    # print compression ratio
    compression = mat[:, 1] / mat[:, 0]
    print(f"compression mean: {np.mean(compression, 0)}, "
          f"std dev: {np.std(compression, 0)}")

    # plt.plot(mat, label=("in", "out"))
    plt.scatter(np.arange(n_lines), compression)
    # plt.plot(compression)
    plt.show()


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


def train_learner():
    if len(sys.argv) != 2:
        print("Usage: learner.py TRAIN\n"
              f"got: {''.join(sys.argv)}")
        exit(1)
    _, train_glob = sys.argv
    train_filenames = sorted(glob(train_glob))

    # book_examples = [parse_str_to_tuple(s.to_code()) for s in zoo]
    lg = LearnedGrammar(**lg_kwargs())
    dataset = LSystemDataset.from_files(train_filenames)
    train_loader = Tdata.DataLoader(dataset, num_workers=3)
    trainer = L.Trainer(max_epochs=10, auto_lr_find=True)
    trainer.tune(model=lg, train_dataloaders=train_loader)
    trainer.fit(model=lg, train_dataloaders=train_loader)

    print("Untrained grammar")
    print(lg.original_grammar)
    print("Trained grammar")
    print(lg.grammar)


def simplify_files():
    if len(sys.argv) != 2:
        print("Usage: python learner.py IN_FILE")
        exit(1)

    _, in_glob = sys.argv
    for file in sorted(glob(in_glob)):
        simplify_file(file)


def load_learned_grammar(checkpt_path: str) -> LearnedGrammar:
    return LearnedGrammar.load_from_checkpoint(checkpoint_path=checkpt_path, **lg_kwargs())


def compare_models():
    if len(sys.argv) != 4:
        print("Usage: python learner.py MODEL1 MODEL2 DATASET")
        print(f"Received {len(sys.argv)} args")
        exit(1)
    _, m1, m2, ds = sys.argv
    model1 = load_learned_grammar(m1)
    model2 = load_learned_grammar(m2)
    model1.eval()
    model2.eval()
    dataset = LSystemDataset.from_files(sorted(glob(ds)))
    dataloader = Tdata.DataLoader(dataset)
    for (x,), y in dataloader:
        ttree = parse_str_to_tuple(x)
        print(f"{x}\n"
              f"  model1 loss: {-model1.grammar.log_probability(model1.start_symbol, ttree)}\n"
              f"  model2 loss: {-model2.grammar.log_probability(model2.start_symbol, ttree)}")


if __name__ == "__main__":
    # simplify_files()
    train_learner()
    # compare_models()

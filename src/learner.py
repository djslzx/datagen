from __future__ import annotations

import pdb

import lark
import lightning as L
import torch.utils.data as Tdata
from typing import *
import sys
from glob import glob

from grammar import Grammar, LearnedGrammar, ConvFeatureExtractor
from lindenmayer import S0LSystem
from evo import DRAW_ARGS
from zoo import zoo
import parse

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
    new_fname = f"{fname}-simpl.txt"
    n_lines_in, n_lines_out = 0, 0
    print(f"Writing simplified file to {new_fname}")
    with open(fname, 'r') as f_in, open(new_fname, 'w') as f_out:
        for line in f_in.readlines():
            n_lines_in += 1
            if line.startswith("#"):  # skip comments
                f_out.write(line)
                n_lines_out += 1
                continue
            if ":" in line:  # split out scores
                line = line.split(" : ")[0]
            # simplify line
            try:
                line_simpl = parse.simplify(line)
                f_out.write(line_simpl + "\n")
                n_lines_out += 1
            except (lark.UnexpectedCharacters,
                    lark.UnexpectedToken,
                    parse.ParseError):
                pass
    print(f"Wrote {n_lines_out} out of {n_lines_in} lines")


def train_learner():
    if len(sys.argv) != 3:
        print("Usage: learner.py train test\n"
              f"got: {''.join(sys.argv)}")
        exit(1)
    _, train_glob, test_glob = sys.argv
    train_filenames = sorted(glob(train_glob))
    test_filenames = sorted(glob(test_glob))

    # book_examples = [parse_str_to_tuple(s.to_code()) for s in zoo]
    g = Grammar.from_components(components=parse.rule_types, gram=2)
    fe = ConvFeatureExtractor(n_features=1000,
                              n_color_channels=1,
                              n_conv_channels=12,
                              bitmap_n_rows=128,
                              bitmap_n_cols=128)

    lg = LearnedGrammar(feature_extractor=fe, grammar=g,
                        evaluator=eval_ttree_as_lsys, parser=parse_str_to_tuple,
                        start_symbol="LSystem",
                        learning_rate=0.001)
    dataset = LSystemDataset.from_files(train_filenames)
    train_loader = Tdata.DataLoader(dataset, num_workers=3)
    trainer = L.Trainer(max_epochs=100, auto_lr_find=True)
    trainer.tune(model=lg, train_dataloaders=train_loader)
    trainer.fit(model=lg, train_dataloaders=train_loader)

    print("Untrained grammar")
    print(lg.original_grammar)
    print("Trained grammar")
    print(lg.grammar)

    test_loader = Tdata.DataLoader(LSystemDataset.from_files(test_filenames))
    for lsys_str, out in test_loader:
        ttree = parse_str_to_tuple(lsys_str)
        print(f"{lsys_str}\n"
              f"  trained loss: {-lg.grammar.log_probability(lg.start_symbol, ttree)}\n"
              f"  untrained loss: {-lg.original_grammar.log_probability(lg.start_symbol, ttree)}")


def simplify_files():
    if len(sys.argv) != 2:
        print("Usage: python learner.py IN_FILE")
        exit(1)

    _, in_glob = sys.argv
    for file in sorted(glob(in_glob)):
        simplify_file(file)


if __name__ == "__main__":
    # simplify_files()
    train_learner()

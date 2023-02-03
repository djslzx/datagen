from __future__ import annotations

import pdb

import lightning as L
import torch.utils.data as data
from typing import *
import sys
from glob import glob

from grammar import Grammar, LearnedGrammar, ConvFeatureExtractor
from lindenmayer import S0LSystem
from zoo import zoo
import parse

Tree: TypeAlias = Tuple


# types of components
components = {
    "LSystem_0": ["Axiom", "Rules", "LSystem"],
    "Nonterminal_0": ["Nonterminal"],  # F
    "Terminal_0": ["Terminal"],  # +
    "Terminal_1": ["Terminal"],  # -
    "Axiom_0": ["Nonterminal", "Axiom", "Axiom"],
    "Axiom_1": ["Terminal", "Axiom", "Axiom"],
    "Axiom_2": ["Axiom"],  # empty
    "Rules_0": ["Rule", "Rules", "Rules"],
    "Rules_1": ["Rule", "Rules"],
    "Rule_0": ["Nonterminal", "Rhs", "Rule"],
    "Rhs_0": ["Rhs", "Rhs", "Rhs"],
    "Rhs_1": ["Nonterminal", "Rhs", "Rhs"],
    "Rhs_2": ["Terminal", "Rhs", "Rhs"],
    "Rhs_3": ["Rhs"],  # empty
}

# semantics of components
str_semantics = {
    "LSystem_0": lambda axiom, rules: f"{axiom};{rules}",
    "Nonterminal_0": lambda: "F",
    "Terminal_0": lambda: "+",
    "Terminal_1": lambda: "-",
    "Axiom_0": lambda nt, axiom: f"{nt}{axiom}",
    "Axiom_1": lambda t, axiom: f"{t}{axiom}",
    "Axiom_2": lambda: "",
    "Rules_0": lambda r, rs: f"{r},{rs}",
    "Rules_1": lambda r: r,
    "Rule_0": lambda nt, rhs: f"{nt}~{rhs}",
    "Rhs_0": lambda rhs1, rhs2: f"[{rhs1}]{rhs2}",
    "Rhs_1": lambda nt, rhs: f"{nt}{rhs}",
    "Rhs_2": lambda t, rhs: f"{t}{rhs}",
    "Rhs_3": lambda: "",
}


def to_learner_ast(tree: Tree) -> Tree:
    """
    Convert tuple node ASTs, where node values are tuples of the form (NT, i),
    to ASTs with string values.
    """
    def transform(*node):
        symbol, i, *args = node
        if not args:
            return f"{symbol}_{i}"
        else:
            return f"{symbol}_{i}", *args

    return apply_to_tree(tree, transform)


def to_flat_string(ltree: Tree) -> str:
    if isinstance(ltree, tuple):
        symbol, *args = ltree
    else:
        symbol = ltree
        args = []
    str_args = [to_flat_string(arg) for arg in args]
    return str_semantics[symbol](*str_args)


def evaluate(p: Tree):
    sys_str = to_flat_string(p)
    sys = S0LSystem.from_sentence(list(sys_str))
    render_str = sys.nth_expansion(3)
    return S0LSystem.draw(render_str, d=3, theta=45, n_rows=128, n_cols=128)


class LSystemDataset(data.Dataset):
    """
    Reads in L-system strings and yields ASTs.
    """

    @staticmethod
    def from_files(filenames: List[str]) -> "LSystemDataset":
        strs = []
        for filename in filenames:
            with open(filename, "r") as f:
                for line in f.readlines():
                    if line.startswith("#"):  # skip comments
                        continue
                    if ":" in line:
                        line = line.split(" : ")[0]
                    strs.append(line.strip())
        return LSystemDataset(data=strs)

    def __init__(self, data: List[str]):
        # TODO: allow a program to generate multiple outputs (probabilistic programs)
        super(LSystemDataset).__init__()
        self.data = data

    def __getitem__(self, item):
        s = self.data[item]
        ast = parse_lsystem_to_ast(s)
        ast = to_learner_ast(ast)
        return s, evaluate(ast)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: learner.py train test\n"
              f"got: {''.join(sys.argv)}")
        exit(1)
    train_glob, test_glob = sys.argv[1:]
    train_filenames = sorted(glob(train_glob))
    test_filenames = sorted(glob(test_glob))

    # book_examples = [to_learner_ast(parse_lsystem_ast(s.to_code())) for s in zoo]
    g = Grammar.from_components(components, gram=2)
    fe = ConvFeatureExtractor(n_features=1000,
                              n_color_channels=1,
                              n_conv_channels=12,
                              bitmap_n_rows=128,
                              bitmap_n_cols=128)
    parse = lambda s: to_learner_ast(parse_lsystem_to_ast(s))
    lg = LearnedGrammar(feature_extractor=fe, grammar=g,
                        evaluator=evaluate, parser=parse,
                        start_symbol="LSystem")
    dataset = LSystemDataset.from_files(train_filenames)
    train_loader = data.DataLoader(dataset)
    trainer = L.Trainer()  # limit_train_batches=100, max_epochs=1)
    trainer.fit(model=lg, train_dataloaders=train_loader)

    print("Untrained grammar")
    print(lg.original_grammar)
    print("Trained grammar")
    print(lg.grammar)

    test_loader = data.DataLoader(LSystemDataset.from_files(test_filenames))
    for (s,), out in test_loader:
        ast = parse(s)
        print(f"{s}\n"
              f"  trained loss: {-lg.grammar.log_probability(lg.start_symbol, ast)}\n"
              f"  untrained loss: {-lg.original_grammar.log_probability(lg.start_symbol, ast)}")

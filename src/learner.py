from __future__ import annotations

import itertools

import lark
import torch as T
import lightning as pl
import torch.utils.data as Tdata
from typing import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import parse
import util
from grammar import Grammar, LearnedGrammar, ConvFeatureExtractor
from lindenmayer import S0LSystem
from evo import DRAW_ARGS
from zoo import zoo
from featurizers import ResnetFeaturizer, Featurizer


def eval_ttree_as_lsys(p: Tuple, level=3):
    sys_str = parse.eval_ttree_as_str(p)
    sys = S0LSystem.from_sentence(list(sys_str))
    return sample_from_lsys(sys, level)


def sample_from_lsys(lsys: S0LSystem, level=3) -> np.ndarray:
    render_str = lsys.nth_expansion(level)
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
        "learning_rate": 1e-5,
    }


def train_learner(train_filenames: List[str], epochs: int):
    lg = LearnedGrammar(**lg_kwargs())
    train_dataset = LSystemDataset.from_files(train_filenames)
    train_loader = Tdata.DataLoader(train_dataset, shuffle=True)
    val_dataset = LSystemDataset([sys.to_str() for sys in zoo])
    val_loader = Tdata.DataLoader(val_dataset)
    trainer = pl.Trainer(max_epochs=epochs, auto_lr_find=False)
    trainer.tune(model=lg, train_dataloaders=train_loader)
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


def test_model(model: LearnedGrammar, s: str, k: int,
               n_tries: int, n_renders_per_try: int):
    featurizer = ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True)

    in_lsys = S0LSystem.from_str(s)
    in_img = sample_from_lsys(in_lsys)
    in_v = featurizer.apply(in_img)
    in_class = featurizer.top_k_classes(in_v, 1)[0]
    mg = model.forward(in_lsys, in_img)

    # top_k = np.empty(k, dtype=object)  # store tuple of (str, img, dist)
    dists = np.empty(n_tries)
    imgs = np.empty(n_tries, dtype=object)
    strs = np.empty(n_tries, dtype=object)
    for i in range(n_tries):
        sample_s = parse.eval_ttree_as_str(mg.sample("LSystem"))
        sample_sys = S0LSystem.from_str(sample_s)

        best_dist = np.inf
        best_img = None

        for j in range(n_renders_per_try):
            sample_img = sample_from_lsys(sample_sys)
            sample_v = featurizer.apply(sample_img)
            d = np.linalg.norm(in_v - sample_v, ord=2)
            if d < best_dist:
                best_img = sample_img
                best_dist = d

        np.searchsorted

        strs[i] = sample_s
        dists[i] = best_dist
        imgs[i] = best_img

    guesses = [
        (strs[i], dists[i], imgs[i])
        for i in np.argsort(-dists)[:10]
    ]

    for guess, dist, img in guesses:
        feats = featurizer.apply(img)
        cls = featurizer.top_k_classes(feats, 1)[0]

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(in_img)
        ax[0].title.set_text(f"Target={s}\n"
                             f"class={in_class}")
        ax[1].imshow(img)
        ax[1].title.set_text(f"Guess={guess}\n"
                             f"@ dist={dist:.3e},\n"
                             f"class={cls}")
        plt.show()


if __name__ == "__main__":
    path = "../models/291573_ns/epoch=43-step=3005904.ckpt"
    model = load_learned_grammar(path)
    print(f"Model loaded from {path}")
    data = [
        "F;F~F[+F[+F]F[+F]F]F",
        "F;F~F[-F[+F[+F]F[+F]F]F]F[-F]F",
        "F;F~F",
        "F;F~FF",
        "F;F~FFF",
        "-F;F~+F",
    ]
    for s in data:
        test_model(model, s, 100, 3)


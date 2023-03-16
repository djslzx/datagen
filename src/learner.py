from __future__ import annotations

import lark
import torch as T
import lightning as pl
import torch.utils.data as Tdata
from typing import *
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

import util
from lang import Language, Tree, ParseError
from lindenmayer import LSys
from regexpr import Regex
from grammar import LearnedGrammar, ConvFeatureExtractor, SBertFeatureExtractor, FeatureExtractor
from zoo import zoo_strs


class LangDataset(Tdata.Dataset):
    """
    Reads in L-system strings and yields ASTs.
    """

    @staticmethod
    def from_files(filenames: List[str], lang: Language) -> "LangDataset":
        data = []
        for filename in filenames:
            with open(filename, "r") as f:
                for line in f.readlines():
                    if line.startswith("#"):  # skip comments
                        continue
                    if ":" in line:  # split out scores
                        line = line.split(" : ")[0]
                    try:
                        lang.parse(line)  # test that line is parseable
                        data.append(line)
                    except (lark.UnexpectedCharacters,
                            lark.UnexpectedToken,
                            ParseError):
                        pass
        return LangDataset(data, lang)

    def __init__(self, data: List[str], lang: Language):
        super(LangDataset).__init__()
        self.data = data
        self.lang = lang

    def __getitem__(self, item):
        # TODO: allow a program to generate multiple outputs (probabilistic programs)
        s = self.data[item]
        t = self.lang.parse(s)
        y = self.lang.eval(t, env={})
        return s, y

    def __len__(self):
        return len(self.data)


def lsys_grammar_kwargs(lang: Language):
    def parse(s: str) -> tuple:
        return lang.parse(s).to_tuple()

    fe = ConvFeatureExtractor(n_features=lang.featurizer.n_features,
                              n_color_channels=3,
                              n_conv_channels=12,
                              bitmap_n_rows=128,
                              bitmap_n_cols=128)
    return {
        "feature_extractor": fe,
        "grammar": lang.model,
        "parse": parse,
        "start_symbol": lang.start,
        "learning_rate": 1e-5,
    }

def grammar_kwargs(l: Language, fe: FeatureExtractor):
    def parse(s: str) -> tuple:
        return l.parse(s).to_tuple()

    return {
        "feature_extractor": fe,
        "grammar": l.model,
        "parse": parse,
        "start_symbol": l.start,
        "learning_rate": 1e-5,
    }


def train_model(lang: Language, learned_grammar: LearnedGrammar,
                train_filenames: List[str], epochs: int):
    train_dataset = LangDataset.from_files(train_filenames, lang)
    train_loader = Tdata.DataLoader(train_dataset, shuffle=True)

    # val_dataset = LangDataset(zoo_strs, lang)
    # val_loader = Tdata.DataLoader(val_dataset)

    trainer = pl.Trainer(max_epochs=epochs, auto_lr_find=False)
    trainer.tune(model=learned_grammar, train_dataloaders=train_loader)
    trainer.fit(model=learned_grammar, train_dataloaders=train_loader)


def load_learned_grammar(lang: Language, checkpt_path: str) -> LearnedGrammar:
    ckpt = T.load(checkpt_path)
    weights = ckpt["grammar_params"]
    lg = LearnedGrammar.load_from_checkpoint(checkpoint_path=checkpt_path, **lsys_grammar_kwargs(lang))
    lg.grammar.from_tensor_(weights)
    return lg


def run_model(name: str, lang: Language, v_in: np.ndarray, k: int, n_tries: int, n_renders_per_try: int):
    def best_render(t: Tree) -> Tuple[float, np.ndarray]:
        """find best render in `n_renders_pre_try` tries"""
        min_dist = np.inf
        min_output = None
        for _ in range(n_renders_per_try):
            output = lang.eval(t, env={})
            v = lang.featurizer.apply(output)
            d = dist.minkowski(v_in, v)
            if d < min_dist:
                min_output = output
                min_dist = d
        return min_dist, min_output

    # track `k` best outcomes of `n_tries` attempts
    dists = np.repeat(np.inf, k)
    imgs = np.empty((k, 128, 128), dtype=np.uint8)
    print(f"Sampling from {name}...")
    for _ in tqdm(range(1, n_tries+1)):
        # sample an L-system from the grammar
        t = lang.sample()
        if len(t) > 1000: continue  # skip absurdly long L-systems

        # choose representative render w/ distance in feature space
        d, img = best_render(t)

        # update cache of best attempts
        i = np.searchsorted(dists, d)  # sort decreasing
        if i < k and d < dists[-1]:
            dists = np.insert(dists, i, d)[:k]
            imgs = np.insert(imgs, i, img, axis=0)[:k]

    return list(zip(dists, imgs))


def run_models(named_models: Dict[str, LearnedGrammar], lsys: LSys, dataset: List[str], k: int,
               n_tries: int, n_renders_per_try: int, save_dir: str):
    sns.set_theme(style="white")
    n = len(named_models)

    for i, datum in enumerate(dataset):
        print(f"Sampling from models {list(named_models.keys())} for L-system {i}:{datum}...")

        fig, axes = plt.subplots(k + 1, n)  # show target + attempts
        plt.suptitle(f"Target: {datum}")

        # plot image once
        lsys_in = lsys.parse(datum)
        img_in = lsys.eval(lsys_in, env={})
        v_in = lsys.featurizer.apply(img_in)

        for col, (name, model) in enumerate(named_models.items()):
            guesses = run_model(
                name=name,
                lang=lsys,
                v_in=v_in,
                k=k,
                n_tries=n_tries,
                n_renders_per_try=n_renders_per_try,
            )

            # plot target in top row
            ax = axes[0, col]
            ax.imshow(img_in)
            ax.set_title(name, pad=2, fontsize=5)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # plot guesses
            for row, (d, img) in enumerate(guesses, 1):
                ax = axes[row, col]
                ax.imshow(img)
                ax.set_title(f"{d:.6e}", pad=2, fontsize=5)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        fig.set_size_inches(10, 10)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.savefig(f"{save_dir}/plot-{i}.png", bbox_inches='tight', dpi=300)


def run_models_on_datasets():
    prefix = "/home/djl328/prob-repl"
    paths = {
        "ns": f"{prefix}/models/291573_ns/epoch=43-step=3005904.ckpt",
        "ns_egg": f"{prefix}/models/291507_ns_egg/epoch=43-step=2999568.ckpt",
        "ns_egg_nov": f"{prefix}/models/294291_ns_egg_nov/epoch=49-step=2871100.ckpt",
        "random": f"{prefix}/models/294289_rand/epoch=41-step=4200000.ckpt",
        "random_egg": f"{prefix}/models/294290_rand_egg/epoch=47-step=4239744.ckpt",
    }
    models = {}
    lsys = LSys(45, 3, 3, 128, 128)
    for name, path in paths.items():
        print(f"Loading model {name} from {path}...")
        models[name] = load_learned_grammar(lang=lsys, checkpt_path=path)

    print("Loaded models: " + ", ".join(models.keys()))
    data = [
        # zoo
        "-F;F~FF-F-F+F+F-F-FF+F+FFF-F+F+FF+F-FF-F-F+F+FF,"
        "F~+FF-F-F+F+FF+FFF-F-F+FFF-F-FF+F+F-F-F+F+FF",
        "F;F~F[+F]F[-F]F",
        "F;F~F-[[F]+F]+F[+FF]-F,F~FF",

        # ns data (filt/simpl)
        "F;F~F,F~F[+F[-F]F[-F]F]F",
        "F;F~F[+F[+F]F[+F]F]F",
    ]
    t = int(time.time())
    save_dir = f"../out/plots/{t}-sample"
    util.try_mkdir(save_dir)
    lang = LSys(45, 3, 3, 128, 128)
    run_models(models, lang, data, k=5, n_tries=1000, n_renders_per_try=2, save_dir=save_dir)


def train_lsys():
    lsys = LSys(45, 3, 3, 128, 128)
    lsys_fe = ConvFeatureExtractor(n_features=lsys.featurizer.n_features,
                                   n_color_channels=3,
                                   n_conv_channels=12,
                                   bitmap_n_rows=128,
                                   bitmap_n_cols=128)
    lsys_lg = LearnedGrammar(**grammar_kwargs(lsys, lsys_fe))
    train_model(lsys, lsys_lg, train_filenames=["../datasets/lsystems/ns/ns.txt"], epochs=2)


def train_regex():
    rgx = Regex()
    rgx_fe = SBertFeatureExtractor()
    rgx_lg = LearnedGrammar(**grammar_kwargs(rgx, rgx_fe))
    train_model(rgx, rgx_lg, train_filenames=["../datasets/regex/ns/ns100x100.txt"], epochs=2)


if __name__ == "__main__":
    train_regex()

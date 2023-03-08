from __future__ import annotations

import lark
import torch as T
import lightning as pl
import torch.utils.data as Tdata
from typing import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

import util
from lang import Language, Tree, ParseError
from lindenmayer import LSys
from grammar import Grammar, LearnedGrammar, ConvFeatureExtractor
from zoo import zoo
from featurizers import ResnetFeaturizer, Featurizer


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
                        t = lang.parse(line)
                        data.append(t)
                    except (lark.UnexpectedCharacters,
                            lark.UnexpectedToken,
                            ParseError):
                        pass
        return LangDataset(data, lang)

    def __init__(self, data: List[Tree], lang: Language):
        super(LangDataset).__init__()
        self.data = data
        self.lang = lang

    def __getitem__(self, item):
        # TODO: allow a program to generate multiple outputs (probabilistic programs)
        t = self.data[item]
        y = self.lang.eval(t, env={})
        return t, y

    def __len__(self):
        return len(self.data)


def lg_kwargs(lang: Language):
    def parse(t: Tree) -> tuple:
        return t.to_tuple()
    fe = ConvFeatureExtractor(n_features=1000,  # FIXME: use n_parameters of featurizer
                              n_color_channels=1,
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


def train_learner(lang: Language, train_filenames: List[str], epochs: int):
    lg = LearnedGrammar(**lg_kwargs(lang))
    train_dataset = LangDataset.from_files(train_filenames, lang)
    train_loader = Tdata.DataLoader(train_dataset, shuffle=True)
    val_dataset = LangDataset([sys.to_str() for sys in zoo], lang)
    val_loader = Tdata.DataLoader(val_dataset)
    trainer = pl.Trainer(max_epochs=epochs, auto_lr_find=False)
    trainer.tune(model=lg, train_dataloaders=train_loader)
    trainer.fit(model=lg, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Untrained grammar")
    print(lg.original_grammar)
    print("Trained grammar")
    print(lg.grammar)


def load_learned_grammar(lang: Language, checkpt_path: str) -> LearnedGrammar:
    ckpt = T.load(checkpt_path)
    weights = ckpt["grammar_params"]
    lg = LearnedGrammar.load_from_checkpoint(checkpoint_path=checkpt_path, **lg_kwargs(lang))
    lg.grammar.from_tensor_(weights)
    return lg


def run_model(name: str, lang: Language, v_in: np.ndarray, featurizer: ResnetFeaturizer,
              k: int, n_tries: int, n_renders_per_try: int):
    def best_render(t: Tree) -> Tuple[float, np.ndarray]:
        """find best render in `n_renders_pre_try` tries"""
        min_dist = np.inf
        min_img = None
        for _ in range(n_renders_per_try):
            img = lang.eval(t, env={})
            v = featurizer.apply(img)
            dist = np.linalg.norm(v_in - v, ord=2)
            if dist < min_dist:
                min_img = img
                min_dist = dist
        return min_dist, min_img

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


def run_models(named_models: Dict[str, LearnedGrammar], lang: Language, dataset: List[str], k: int,
               n_tries: int, n_renders_per_try: int, save_dir: str):
    sns.set_theme(style="white")
    n = len(named_models)
    featurizer = ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True)

    for i, datum in enumerate(dataset):
        print(f"Sampling from models {list(named_models.keys())} for L-system {i}:{datum}...")

        fig, axes = plt.subplots(k + 1, n)  # show target + attempts
        plt.suptitle(f"Target: {datum}")

        # plot image once
        lsys_in = lang.parse(datum)
        img_in = lang.eval(lsys_in, env={})
        v_in = featurizer.apply(img_in)

        for col, (name, model) in enumerate(named_models.items()):
            guesses = run_model(
                name=name,
                lang=lang,
                featurizer=featurizer,
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
            for row, (dist, img) in enumerate(guesses, 1):
                ax = axes[row, col]
                ax.imshow(img)
                ax.set_title(f"{dist:.6e}", pad=2, fontsize=5)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        fig.set_size_inches(10, 10)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.savefig(f"{save_dir}/plot-{i}.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    paths = {
        "ns": "../models/291573_ns/epoch=43-step=3005904.ckpt",
        "ns_egg": "../models/291507_ns_egg/epoch=43-step=2999568.ckpt",
        "ns_egg_nov": "../models/294291_ns_egg_nov/epoch=49-step=2871100.ckpt",
        "random": "../models/294289_rand/epoch=41-step=4200000.ckpt",
        "random_egg": "../models/294290_rand_egg/epoch=47-step=4239744.ckpt",
    }
    models = {}
    for name, path in paths.items():
        print(f"Loading model {name} from {path}...")
        models[name] = load_learned_grammar(path)

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
    run_models(models, data, k=5, n_tries=1000, n_renders_per_try=2, save_dir=save_dir)

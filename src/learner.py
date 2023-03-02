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


def run_model(name: str, mg: Grammar, v_in: np.ndarray, featurizer: ResnetFeaturizer,
              k: int, n_tries: int, n_renders_per_try: int):
    def best_render(lsys: S0LSystem) -> Tuple[float, np.ndarray]:
        """find best render in `n_renders_pre_try` tries"""
        min_dist = np.inf
        min_img = None
        for _ in range(n_renders_per_try):
            img = sample_from_lsys(lsys)
            v = featurizer.apply(img)
            dist = np.linalg.norm(v_in - v, ord=2)
            if dist < min_dist:
                min_img = img
                min_dist = dist
        return min_dist, min_img

    # track `k` best outcomes of `n_tries` attempts
    dists = np.repeat(np.inf, k)
    imgs = np.empty((k, DRAW_ARGS["n_rows"], DRAW_ARGS["n_cols"]), dtype=np.uint8)
    print(f"Sampling from {name}...")
    for _ in tqdm(range(1, n_tries+1)):
        # sample an L-system from the grammar
        sample_s = parse.eval_ttree_as_str(mg.sample("LSystem"))
        if len(sample_s) > 1000: continue  # skip absurdly long L-systems
        sample_sys = S0LSystem.from_str(sample_s)

        # choose representative render w/ distance in feature space
        d, img = best_render(sample_sys)

        # update cache of best attempts
        i = np.searchsorted(dists, d)  # sort decreasing
        if i < k and d < dists[-1]:
            dists = np.insert(dists, i, d)[:k]
            imgs = np.insert(imgs, i, img, axis=0)[:k]

    return list(zip(dists, imgs))


def run_models(named_models: Dict[str, LearnedGrammar], dataset: List[str], k: int,
               n_tries: int, n_renders_per_try: int, save_dir: str):
    sns.set_theme(style="white")
    n = len(named_models)
    featurizer = ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True)

    for i, datum in enumerate(dataset):
        print(f"Sampling from models {list(named_models.keys())} for L-system {i}:{datum}...")

        fig, axes = plt.subplots(k + 1, n)  # show target + attempts
        plt.suptitle(f"Target: {datum}")

        # plot image once
        lsys_in = S0LSystem.from_str(datum)
        img_in = sample_from_lsys(lsys_in)
        v_in = featurizer.apply(img_in)

        for col, (name, model) in enumerate(named_models.items()):
            guesses = run_model(
                name=name,
                mg=model.forward(lsys_in, img_in),
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
    run_models(models, data, k=5, n_tries=1000, n_renders_per_try=2, save_dir=save_dir)

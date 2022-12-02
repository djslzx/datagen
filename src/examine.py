from math import sqrt, ceil
from evo import ROLLOUT_DEPTH, DRAW_ARGS
from lindenmayer import S0LSystem
import util


def plot_outputs(filename: str):
    with open(filename, "r") as f:
        imgs = []
        labels = []
        for line in f.readlines():
            if ':' in line:
                sys_str, score = line.split(' : ')
                if not score.strip().endswith('*'): continue  # skip unselected children
            else:
                sys_str = line
                score = ""

            sys = S0LSystem.from_sentence(list(sys_str))
            img = S0LSystem.draw(sys.nth_expansion(ROLLOUT_DEPTH), **DRAW_ARGS)
            imgs.append(img)
            labels.append(f"{sys_str}\n{score}")

    n_cols = ceil(sqrt(len(imgs)))
    n_rows = ceil(len(imgs)/n_cols)
    util.plot(title=filename, imgs=imgs, shape=(n_rows, n_cols), labels=labels, saveto=f"{filename}.png")


if __name__ == '__main__':
    for i in range(5):
        # fname = f"../out/ns/nolen50/pcfg-1669947420-nolen-gen-{i}.txt"
        fname = f".cache/pcfg-1669965929-nolen-gen-{i}.txt"
        print(f"Plotting {fname}")
        plot_outputs(fname)
    plot_outputs(".cache/pcfg-1669965929-nolen-arkv.txt")
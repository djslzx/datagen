from math import sqrt, ceil
from evo import ROLLOUT_DEPTH, DRAW_ARGS
from lindenmayer import S0LSystem
import util


def plot_outputs(filename: str):
    with open(filename, "r") as f:
        imgs = []
        labels = []
        for line in f.readlines():
            sys_str, score = line.split(' : ')
            if not score.strip().endswith('*'): continue  # skip unselected children
            sys = S0LSystem.from_sentence(list(sys_str))
            img = S0LSystem.draw(sys.nth_expansion(ROLLOUT_DEPTH), **DRAW_ARGS)
            imgs.append(img)
            labels.append(f"{sys_str}\n{score}")

            if len(imgs) == 25:
                util.plot(imgs, shape=(5, 5), labels=labels)
                imgs = []
                labels = []

        # plot remaining imgs
        if imgs:
            n_rows = int(sqrt(len(imgs)))
            n_cols = ceil(len(imgs)//n_rows)
            util.plot(imgs, shape=(n_rows, n_cols), labels=labels)


if __name__ == '__main__':
    for i in [0, 1, 2, 3, 4]:
        fname = f".cache/pcfg-1669943468-ignore-length-gen-{i}.txt"
        print(f"Plotting {fname}")
        plot_outputs(fname)

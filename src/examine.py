from math import sqrt, ceil
from evo import ROLLOUT_DEPTH, DRAW_ARGS
from lindenmayer import S0LSystem
import util


def plot_outputs(filename: str, batch_size=36):
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
            labels.append(f"{score}")

    n_cols = ceil(sqrt(batch_size))
    n_rows = ceil(batch_size / n_cols)
    n_batches = ceil(len(imgs) / batch_size)
    for i in range(n_batches):
        img_batch = imgs[i * batch_size: (i+1) * batch_size]
        label_batch = labels[i * batch_size: (i+1) * batch_size]
        util.plot(title=filename,
                  imgs=img_batch,
                  shape=(n_rows, n_cols),
                  labels=label_batch,
                  saveto=f"{filename}-{i}.png")


if __name__ == '__main__':
    for i in range(100):
        fname = f"../out/ns/nolen-1669959013/pcfg-1669959013-nolen-gen-{i}.txt"
        # fname = f"../out/ns/nolen50/pcfg-1669947420-nolen-gen-{i}.txt"
        # fname = f".cache/pcfg-1669965929-nolen-gen-{i}.txt"
        print(f"Plotting {fname}")
        plot_outputs(fname)
    # plot_outputs("../out/ns/nolen-1669947420/pcfg-1669947420-nolen-arkv.txt")
    # plot_outputs("../out/ns/nolen-1669947420/pcfg-1669947420-nolen-arkv.txt")

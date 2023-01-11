from math import sqrt, ceil
from glob import glob
import sys

from evo import ROLLOUT_DEPTH, DRAW_ARGS
from lindenmayer import S0LSystem
from featurizers import ResnetFeaturizer
import util


classifier = ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True)


def plot_outputs(filename: str, batch_size=36, save=True):
    with open(filename, "r") as f:
        imgs = []
        labels = []
        for line in f.readlines():
            # skip comment lines
            if line.startswith('#'):
                print(line)
                continue
            if ':' in line:
                sys_str, score = line.split(' : ')
                if not score.strip().endswith('*'): continue  # skip unselected children
            else:
                sys_str = line
                score = ""

            sys = S0LSystem.from_sentence(list(sys_str))
            img = S0LSystem.draw(sys.nth_expansion(ROLLOUT_DEPTH), **DRAW_ARGS)
            imgs.append(img)

            # check resnet classifier output
            features = classifier.apply(img)
            top_class = classifier.top_k_classes(features, k=1)[0]
            score += f" ({top_class})"
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
                  saveto=f"{filename}-{i}.png" if save else None)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: examine.py FILE_GLOB SAVE")
        print(sys.argv)
        exit(1)

    file_glob, save = sys.argv[1:]
    save = save == "True"

    for fname in sorted(glob(file_glob)):
        print(f"Plotting {fname} with save={save}")
        plot_outputs(fname, save=save)

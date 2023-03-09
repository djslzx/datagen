from math import sqrt, ceil
from glob import glob
import sys
import numpy as np
from tqdm import tqdm
from typing import *

from lang import Language
from lindenmayer import LSys
from regexpr import Regex
from featurizers import ResnetFeaturizer
import util


def read_outfile(filename: str) -> Generator[Tuple[str, str], None, None]:
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith('#'):
                # skip comments
                continue
            if ':' in line:
                s, score = line.split(' : ')
                score = score.strip()
                if not score.endswith('*'): continue  # skip unselected children
            else:
                s = line
                score = ""
            yield s, score


def plot_lsys_outputs(filename: str, batch_size=36, len_cap=1000, save=True):
    lsys = LSys(theta=45, step_length=3, render_depth=3, n_rows=128, n_cols=128)
    classifier = ResnetFeaturizer(disable_last_layer=False, softmax_outputs=True)
    imgs = []
    labels = []
    for s, score in read_outfile(filename):
        # skip l-systems that take too long to render
        if len(s) <= len_cap:
            t = lsys.parse(s)
            img = lsys.eval(t, env={})
            imgs.append(img)

            # check resnet classifier output
            features = classifier.apply(img)
            top_class = classifier.top_k_classes(features, k=1)[0]
            labels.append(f"{score} ({top_class})")
        else:
            imgs.append(np.zeros((128, 128)))
            labels.append(f"skipped (len={len(s)})")

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


def show_regex_outputs(filename: str, n_samples: int):
    r = Regex()
    for s, score in read_outfile(filename):
        print(s)
        t = r.parse(s)
        print(f"Samples from {s} w/ score {score}:")
        print([r.eval(t, env={}) for _ in range(n_samples)])


if __name__ == '__main__':
    def usage():
        print("Usage: view.py FILE_GLOB (regex | lsystem) [save]")
        print(f"Received: {sys.argv}")
        exit(1)

    if len(sys.argv) < 2:
        usage()

    file_glob = sys.argv[1]
    save = "save" in sys.argv[2:]
    regex_kind = "regex" in sys.argv[2:]
    lsys_kind = "lsystem" in sys.argv[2:]

    for fname in sorted(glob(file_glob)):
        print(f"Viewing {fname} with save={save}")
        if regex_kind:
            show_regex_outputs(fname, n_samples=10)
        elif lsys_kind:
            plot_lsys_outputs(fname, save=save)
        else:
            usage()
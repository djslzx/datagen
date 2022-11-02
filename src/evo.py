"""
Test out evolutionary search algorithms for data augmentation.
"""
from typing import List, Dict, Set, Tuple, Iterator
import pickle
import time
import pdb

from lindenmayer import S0LSystem
from inout import log_io
from resnet import featurize
from datagen import GENERAL_MG
import util

PCFG_CACHE_PREFIX = f"cache/pcfg-"


def mutate(specimens: List[S0LSystem], n_samples: int, smoothing=0.5) -> Iterator[S0LSystem]:
    """
    Produce the next generation of L-systems from a set of L-system specimens.

    Fit a PCFG to the specimens using inside-outside with smoothing, then sample
    from the PCFG to get 'mutated' L-systems.
    """
    # check cached PCFG
    genomes = [specimen.to_sentence() for specimen in specimens]
    hash_str = "\n".join(" ".join(genome) for genome in genomes) + str(smoothing)
    hash_val = util.md5_hash(hash_str)
    cache_file = f"{PCFG_CACHE_PREFIX}{hash_val}"

    try:
        with open(cache_file, "rb") as f:
            g_fit = pickle.load(f)
        print(f"Found cached file, loaded fitted PCFG from {cache_file}: {g_fit}")

    except FileNotFoundError:
        print("No cached file found, running inside-outside...")
        g = GENERAL_MG.to_CNF().normalized().log()
        g_fit = log_io(g, genomes, smoothing, verbose=True)
        print(f"Fitted PCFG: {g_fit}")

        # cache pcfg
        print(f"Fitted PCFG, saving to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(g_fit, f)

    # sample from fitted PCFG
    for i in range(n_samples):
        sentence = g_fit.exp().iterate_fully()  # TODO: check that this doesn't run excessively
        sys = S0LSystem.from_sentence(sentence)
        yield sys


if __name__ == '__main__':
    next_gen = mutate(specimens=[S0LSystem("F", {"F": ["F+F", "F-F"]})],
                      n_samples=3,
                      smoothing=0.01)
    for sys in next_gen:
        print(sys)

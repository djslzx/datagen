import pickle
import time

from lindenmayer import S0LSystem, LSYSTEM_MG
from cfg import PCFG
from inout import autograd_io, log_io
import book_zoo
import util

DIR = "../out/io-samples/"


def fit_mg(zoo_limit=None):
    # extract & render training specimens
    specimens = [sys for sys, angle in book_zoo.zoo[:zoo_limit]]
    for i, specimen in enumerate(specimens):
        for j in range(3):
            d, s = specimen.expand_until(1000)
            S0LSystem.render(s, d=5, theta=43, filename=f"{DIR}/ref-{i:03d}-{j:03d}")

    # fit meta-grammar using inside-outside
    corpus = [s.to_sentence() for s in specimens]
    mg = LSYSTEM_MG.to_bigram().to_CNF().normalized().log()
    mg = log_io(mg, corpus, verbose=True).exp()
    print(f"Finished tuning: {mg}")

    print("Saving metagrammar...")
    t = time.time()
    with open(f"{DIR}/mg-{t}.dat", "wb") as f:
        pickle.dump(mg, f)
    with open(f"{DIR}/mg-{t}.log", "w") as f:
        f.write(f"{mg}")
    return mg


def sample_from_mg(mg: PCFG, n_systems: int, n_samples_per_system: int):
    for i in range(n_systems):
        print(f"Generating {i}-th L-system from tuned grammar...")
        try:
            sentence = mg.iterate_fully()
            lsys = S0LSystem.from_sentence(sentence)
            print(f"L-system {i}: {lsys}")

            with open(f"{DIR}/sys-{i:03d}.log", "w") as f:
                f.write("".join(sentence) + '\n')
                f.write(f"{lsys}")

            for j in range(n_samples_per_system):
                print(f"  Rendering {j}-th sample...")
                d, s = lsys.expand_until(1000)
                S0LSystem.render(s, d=5, theta=43,
                                 filename=f"{DIR}/sys-{i:03d}-{j:03d}")
            print()
        except ValueError:
            print(f"Produced uninterpretable L-system, e.g. {mg.iterate_until(100)}")
            pass


if __name__ == '__main__':
    with open("../out/saved-mgs/mg-1666938123.479085.dat", "rb") as f:
        g = pickle.load(f)
    print(g)
    sample_from_mg(g, n_systems=100, n_samples_per_system=3)

    # fit_and_sample(n_systems=20, n_samples_per_system=3, zoo_limit=None)
    # check_io_autograd(io_iters=100, n_samples=10, zoo_limit=1)

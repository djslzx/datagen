import random
import numpy as np
from typing import List
import lindenmayer

def coinflip(p: float):
    assert 0 <= p <= 1, f"p is not a probability, p={p}"
    return random.random() < p

def random_grammar(alphabet: List[str], n_rules_cap: int, n_successors_cap: int):
    """Generate a bracketed stochastic L-system with axiom F."""

    # generate production rules
    n_productions = random.randint(1, n_rules_cap)
    productions = {}
    for i in range(n_productions):
        # choose predecessor
        # TODO: don't hard-code, allow intermediate vars
        predecessor = "F"

        # choose successors
        n_successors = random.randint(1, n_successors_cap)
        successors = [random_successor(alphabet) for j in range(n_successors)]
        
        # add to productions
        productions[predecessor] = successors

    # generate production rule probabilities
    distribution = {}
    for predecessor, successors in productions.items():
        weights = np.random.rand(len(successors))
        normed_weights = weights / weights.sum() # normalize to sum to 1
        distribution[predecessor] = normed_weights

    return lindenmayer.SOLSystem(
        axiom="F",
        productions=productions,
        distribution=distribution
    )

def random_balanced_brackets(n_bracket_pairs: int) -> List[str]:
    out = ['[', ']']
    for _ in range(n_bracket_pairs - 1):
        start_pos = random.randint(0, len(out) - 1)
        end_pos = random.randint(start_pos + 1, len(out))
        out.insert(start_pos, "[")
        out.insert(end_pos, "]")
    return out

def random_successor(alphabet: List[str]) -> str:
    n_letters = int(random.gammavariate(5, 2)) # letters not including brackets
    n_bracket_pairs = random.randint(0, n_letters // 3) # number of pairs of brackets

    # randomly insert brackets
    succ = random_balanced_brackets(n_bracket_pairs)

    # randomly insert other letters
    for i in range(n_letters):
        pos = random.randint(0, len(succ) - 1)
        letter = random.choice(alphabet)
        succ.insert(pos, letter)

    return "".join(succ)


if __name__ == '__main__':
    g = random_grammar(
        alphabet=['F', '+', '-'],
        n_rules_cap=3,
        n_successors_cap=4,
    )
    print(g)
    iters = 6
    for level, word in enumerate(g.expansions(iters)):
        print(word)
        lindenmayer.LSystem.render(
            word,
            length=5,
            angle=45,
            filename=f'../imgs/random-grammar-{level:02d}'
        )

import random
from typing import Dict, List, Tuple
import util


class CFG:
    """
    A context-free grammar.  Terminals and nonterminals are represented as
    strings. The rule A ::= B C | D | ... is represented as the mapping
    A -> [[B, C], [D]].

    Terminals and nonterminals are not input explicitly -- they are inferred
    from the given rules. If a word is a predecessor in the rule list, then
    it is a nonterminal.  Otherwise, it is a terminal.
    """

    def __init__(self, rules: Dict[str, List[List[str]]]):
        assert all(succ and all(succ) for pred, succ in rules.items()), \
            "All rule RHS should be nonempty; " \
            "each element should also be nonempty"
        self.rules = rules

    def __str__(self):
        rules = "\n  ".join(
            f"{pred} -> {succs}"
            for pred, succs in self.rules.items())
        return "CFG: {\n  " + rules + "\n}"

    def _is_nt(self, letter: str) -> bool:
        return letter in self.rules

    def _choose_successor(self, letter: str) -> List[str]:
        if letter not in self.rules:
            return [letter]
        else:
            return random.choice(self.rules[letter])

    def apply(self, word: List[str]) -> List[str]:
        """
        Nondeterministically apply one of the production rules to
        a letter in the word.
        """
        # Only choose nonterminals to expand
        nonterminals = [i for i, letter in enumerate(word)
                        if self._is_nt(letter)]
        if not nonterminals:
            return word
        index = random.choice(nonterminals)
        letter = word[index]
        expansion = self._choose_successor(letter)
        return word[:index] + expansion + word[index + 1:]

    def fixpoint(self, word: List[str]) -> List[str]:
        """Keep applying rules to the word until it stops changing."""
        prev = word
        current = self.apply(word)
        while current != prev:
            prev = current
            current = self.apply(current)
        return current

    def iterate(self, word: List[str], n: int) -> List[str]:
        """Apply rules to the word `n` times."""
        s = word
        for _ in range(n):
            s = self.apply(s)
        return s

    def iterate_until(self, word: List[str], length: int) -> str:
        """Apply rules to the word until its length is >= `length`."""
        s = word
        while len(s) < length:
            cache = s
            s = self.apply(s)
            if s == cache:
                break
        return self._to_str(s)

    def _to_str(self, word: List[str]) -> str:
        """Turn the word representation into a single Turtle string."""
        filtered = [letter
                    for letter in word
                    if letter not in self.rules.keys()]
        return "".join(filtered)


class PCFG(CFG):
    """
    A probabilistic context-free grammar.  Terminals and nonterminals are
    represented as strings.  All of the rules for a given nonterminal should
    be bundled together into a list.  Each of these rules has an associated
    probability.  An example ruleset is
    {
      "A" : [(["B"], 0.5),
             (["C", "D"], 0.5)]
    }
    This represents a grammar where A expands to B or CD with equal
    probability.
    """

    def __init__(self, rules: Dict[str, List[Tuple[List[str], float]]]):
        assert all(succs and all(succs) for pred, succs in rules.items()), \
            "All RHS should be nonempty"
        self.rules = rules

    def __str__(self) -> str:
        rules = "\n  ".join(
            f"{pred} ->\n    " +
            ",\n    ".join(f"{succ} @ {weight:.2f}"
                           for succ, weight in succs)
            for pred, succs in self.rules.items()
        )
        return "PCFG: {\n  " + rules + "\n}"

    def _choose_successor(self, letter: str) -> List[str]:
        if letter not in self.rules:
            return [letter]
        else:
            succs = self.rules[letter]
            popn = [succ for succ, weight in succs]
            weights = [weight for succ, weight in succs]
            return random.choices(population=popn,
                                  weights=weights,
                                  k=1)[0]


if __name__ == '__main__':
    cfg = CFG(rules={
        "a": [["a", "b"],
              ["a"]],
        "b": [["b", "b"],
              ["b"]],
    })
    pcfg = PCFG(rules={
        "a": [(["b", "a"], 0.5),
              (["a"], 0.5)],
        "b": [(["b"], 0.5),
              (["c", "b"], 0.5)],
    })
    print(cfg)
    print(pcfg)
    print(
        # cfg.iterate(["a"], 10),
        pcfg.iterate(["a"], 10)
    )

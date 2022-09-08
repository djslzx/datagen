import random
from typing import Dict, List, Tuple, Iterator, Union
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

    def is_nonterminal(self, letter: str) -> bool:
        return letter in self.rules

    def is_terminal(self, letter: str) -> bool:
        return letter not in self.rules

    def _choose_successor(self, letter: str) -> List[str]:
        if letter not in self.rules:
            return [letter]
        else:
            return random.choice(self.rules[letter])

    def iter_rules(self) -> Iterator[Tuple[str, List[str]]]:
        for pred, succs in self.rules.items():
            for succ in succs:
                yield pred, succ

    def apply(self, word: List[str]) -> List[str]:
        """
        Nondeterministically apply one of the production rules to
        a letter in the word.
        """
        # Only choose nonterminals to expand
        nonterminals = [i for i, letter in enumerate(word)
                        if self.is_nonterminal(letter)]
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
                    if letter not in self.rules]
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
    Word = List[str]
    Letter = str

    def __init__(self,
                 rules: Dict[Letter, List[Word]],
                 weights: Union[str, Dict[Letter, List[float]]],
                 to_cnf=False):
        assert all(succs and all(succs) for pred, succs in rules.items()), \
            "All RHS should be nonempty"
        assert all(util.unique(succs) for pred, succs in rules.items()), \
            "All successors should be unique wrt a predecessor"
        self.rules = rules
        if weights == "uniform":
            self.weights = {
                pred: [1] * len(succs)
                for pred, succs in rules.items()
            }
        else:
            self.weights = weights
        if to_cnf:
            self.to_CNF()

    def __eq__(self, other):
        return isinstance(other, PCFG) and \
            (self.rules, self.weights) == (other.rules, other.weights)

    def from_list(rs: List[Tuple[Letter, Word, float]]) -> 'PCFG':
        rules = {}
        weights = {}
        for letter, word, weight in sorted(rs, key=lambda x: x[0]):
            if letter not in rules:
                rules[letter] = [word]
                weights[letter] = [weight]
            else:
                rules[letter].append(word)
                weights[letter].append(weight)
        return PCFG(rules, weights)

    def as_list(self) -> List[Tuple[Letter, Word, float]]:
        return [
            (letter, word, weight)
            for letter in self.rules
            for word, weight in zip(self.rules[letter], self.weights[letter])
        ]

    def add_rule(self, pred: Letter, succ: Word, weight: float) -> 'PCFG':
        return PCFG.from_list(self.as_list() + (pred, succ, weight))

    def rm_rule(self, pred: str, succ: List[str]) -> 'PCFG':
        if pred not in self.rules:
            return self
        else:
            return PCFG.from_list(filter(
                lambda x: x[:2] != (pred, succ),
                self.as_list()
            ))

    def to_CNF(self) -> 'PCFG':
        """Convert to Chomsky normal form"""
        def contains_terminals(word: List[str]) -> bool:
            for letter in word:
                if self.is_terminal(letter):
                    return True
            return False

    def _term(self) -> 'PCFG':
        # eliminate rules with nonsolitary terminals
        def nt(c) -> str:
            return f"_TERM_{c}"

        rules = []
        for rule in self.as_list():
            pred, succ, weight = rule
            if len(succ) == 1:
                rules.append(rule)
            else:
                # replace all terminals with a nonterminal
                new_succ = []
                for c in succ:
                    if self.is_terminal(c):
                        rules.append((nt(c), [c], 1))
                        new_succ.append(nt(c))
                    else:
                        new_succ.append(c)
                rules.append((pred, new_succ, weight))

        return PCFG.from_list(rules)

    def _bin(self) -> 'PCFG':
        # eliminate rhs with more than 2 nonterms
        # assumes that TERM has been run, so any rules containing terminals
        # only contain a single terminal
        def nt(pred, i, j) -> str:
            return f"_BIN_{pred}_{i}_{j}"

        rules = []
        for pred in self.rules:
            succs, weights = self.rules[pred], self.weights[pred]
            for i, (succ, weight) in enumerate(zip(succs, weights)):
                if len(succ) > 2:
                    rules.append((pred, [succ[0], nt(pred, i, 1)], 1))
                    j = 1
                    for c in succ[1:-2]:
                        rules.append((nt(pred, i, j), [c, nt(pred, i, j+1)], 1))
                        j += 1
                    rules.append((nt(pred, i, j), succ[-2:], 1))
                else:
                    rules.append((pred, succ, weight))
        return PCFG.from_list(rules)

    def _del(self):
        pass

    def __str__(self) -> str:
        rules = "\n  ".join(
            f"{pred} ->\n    " +
            "\n    ".join(f"{succ} @ {weight:.2f}"
                          for succ, weight in zip(self.rules[pred],
                                                  self.weights[pred]))
            for pred in self.rules
        )
        return "PCFG: {\n  " + rules + "\n}"

    def _choose_successor(self, letter: str) -> List[str]:
        if letter not in self.rules:
            return [letter]
        else:
            return random.choices(population=self.rules[letter],
                                  weights=self.weights[letter],
                                  k=1)[0]


def test_to_CNF():
    g = PCFG(
        rules={
            "A": [["B", "C", "D", "E", "F"],
                  ["D", "E", "F"]],
            "B": [["b"]],
            "C": [["c1", "c2", "c3"], ["c4"]],
            "D": [["D", "d", "D"]],
            "E": [["e"]],
            "F": [["f"]],
        },
        weights="uniform",
    )
    termed = PCFG(
        rules={
            "A": [["B", "C", "D", "E", "F"],
                  ["D", "E", "F"]],
            "B": [["b"]],
            "C": [["_TERM_c1", "_TERM_c2", "_TERM_c3"], ["c4"]],
            "D": [["D", "_TERM_d", "D"]],
            "E": [["e"]],
            "F": [["f"]],
            "_TERM_c1": [["c1"]],
            "_TERM_c2": [["c2"]],
            "_TERM_c3": [["c3"]],
            "_TERM_d": [["d"]],
        },
        weights="uniform",
    )
    g_termed = g._term()
    assert g_termed == termed, f"Expected\n{termed},\nBut got\n{g_termed}"
    binned = PCFG(
        rules={
            "A": [["B", "_BIN_A_0_1"],
                  ["D", "_BIN_A_1_1"]],
            "_BIN_A_0_1": [["C", "_BIN_A_0_2"]],
            "_BIN_A_0_2": [["D", "_BIN_A_0_3"]],
            "_BIN_A_0_3": [["E", "F"]],
            "_BIN_A_1_1": [["E", "F"]],
            "B": [["b"]],
            "C": [["_TERM_c1", "_BIN_C_0_1"], ["c4"]],
            "_BIN_C_0_1": [["_TERM_c2", "_TERM_c3"]],
            "D": [["D", "_BIN_D_0_1"]],
            "_BIN_D_0_1": [["_TERM_d", "D"]],
            "E": [["e"]],
            "F": [["f"]],
            "_TERM_c1": [["c1"]],
            "_TERM_c2": [["c2"]],
            "_TERM_c3": [["c3"]],
            "_TERM_d": [["d"]],
        },
        weights="uniform",
    )
    g_binned = g_termed._bin()
    assert g_binned == binned, f"Expected\n{binned}, but got\n{g_binned}"
    print(" [+] passed test_to_CNF")


if __name__ == '__main__':
    # cfg = CFG(rules={
    #     "a": [["a", "b"],
    #           ["a"]],
    #     "b": [["b", "b"],
    #           ["b"]],
    # })
    # pcfg = PCFG(rules={
    #     "a": [(["b", "a"], 0.5),
    #           (["a"], 0.5)],
    #     "b": [(["b"], 0.5),
    #           (["c", "b"], 0.5)],
    # })
    # print(cfg)
    # print(pcfg)
    # print(
    #     # cfg.iterate(["a"], 10),
    #     pcfg.iterate(["a"], 10)
    # )
    test_to_CNF()

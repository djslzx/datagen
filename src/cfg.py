import random
import itertools as it
from typing import Dict, List, Tuple, Union, Set
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

    def __init__(self, start: str, rules: Dict[str, List[List[str]]]):
        assert start in rules, f"Starting symbol {start} not found in rules"
        assert all(succ and all(succ) for pred, succ in rules.items()), \
            "All rule RHS should be nonempty; " \
            "each element should also be nonempty"
        self.start = start
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

    def fixpoint(self) -> List[str]:
        """Keep applying rules to the word until it stops changing."""
        prev = self.start
        current = self.apply(self.start)
        while current != prev:
            prev = current
            current = self.apply(current)
        return current

    def iterate(self, n: int) -> List[str]:
        """Apply rules to the starting word `n` times."""
        s = self.start
        for _ in range(n):
            s = self.apply(s)
        return s

    def iterate_until(self, length: int) -> str:
        """Apply rules to the starting word until its length is >= `length`."""
        s = self.start
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
                 start: Letter,
                 rules: Dict[Letter, List[Word]],
                 weights: Union[str, Dict[Letter, List[float]]]):
        assert start in rules, f"Starting word {start} not found in rules"
        assert all(succs and all(succs) for pred, succs in rules.items()), \
            "All RHS should be nonempty"
        assert all(util.unique(succs) for pred, succs in rules.items()), \
            "All successors should be unique wrt a predecessor"
        self.start = start
        self.rules = rules
        if weights == "uniform":
            self.set_uniform_weights()
        else:
            self.weights = weights

    def __eq__(self, other):
        return isinstance(other, PCFG) and \
            (self.rules, self.weights) == (other.rules, other.weights)

    def set_uniform_weights(self):
        self.weights = {
            pred: [1 / len(succs)] * len(succs)
            for pred, succs in self.rules.items()
        }

    def from_rule_list(start: Letter,
                       rs: List[Tuple[Letter, Word, float]]) -> 'PCFG':
        """Construct a PCFG from a list of rules with weights"""
        rules = {}
        weights = {}
        for letter, word, weight in sorted(rs, key=lambda x: x[0]):
            if letter not in rules:
                rules[letter] = [word]
                weights[letter] = [weight]
            else:
                rules[letter].append(word)
                weights[letter].append(weight)
        return PCFG(start, rules, weights)

    def as_rule_list(self) -> List[Tuple[Letter, Word, float]]:
        """View a PCFG as a list of rules with weights"""
        return [
            (letter, word, weight)
            for letter in self.rules
            for word, weight in zip(self.rules[letter], self.weights[letter])
        ]

    def add_rule(self, pred: Letter, succ: Word, weight: float) -> 'PCFG':
        """Construct a PCFG by adding a rule to the current PCFG; immutable"""
        return PCFG.from_rule_list(self.start,
                                   self.as_rule_list() + (pred, succ, weight))

    def rm_rule(self, pred: str, succ: List[str]) -> 'PCFG':
        """
        Construct a PCFG by remove a rule from the current PCFG; immutable
        """
        if pred not in self.rules:
            return self
        else:
            return PCFG.from_rule_list(
                self.start,
                filter(
                    lambda x: x[:2] != (pred, succ),
                    self.as_rule_list()
                )
            )

    def to_CNF(self) -> 'PCFG':
        """Convert to Chomsky normal form; immutable"""
        return self._start()._term()._bin()._del()._unit()

    def _start(self) -> 'PCFG':
        """Eliminate the start symbol from any RHS"""
        return PCFG.from_rule_list(
            "_START_",
            self.as_rule_list() + [("_START_", self.start, 1)]
        )

    def _term(self) -> 'PCFG':
        """Eliminate rules with nonsolitary terminals"""
        def nt(c) -> str:
            return f"_TERM_{c}_"

        rules = []
        for pred in self.rules:
            succs, weights = self.rules[pred], self.weights[pred]
            for i, (succ, weight) in enumerate(zip(succs, weights)):
                if len(succ) == 1:
                    rules.append((pred, succ, weight))
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

        return PCFG.from_rule_list(self.start, rules)

    def _bin(self) -> 'PCFG':
        """
        Eliminate rules whose rhs has more than 2 nonterminals.
        Assumes that _term() has been run, so any rules containing terminals
        should only contain a single terminal.
        """
        def nt(pred, i, j) -> str:
            return f"_BIN_{pred}_{i}_{j}_"

        rules = []
        for pred in self.rules:
            succs, weights = self.rules[pred], self.weights[pred]
            for i, (succ, weight) in enumerate(zip(succs, weights)):
                if len(succ) > 2:
                    rules.append((pred, [succ[0], nt(pred, i, 1)], 1))
                    j = 1
                    for c in succ[1:-2]:
                        rules.append((
                            nt(pred, i, j),
                            [c, nt(pred, i, j+1)],
                            1
                        ))
                        j += 1
                    rules.append((nt(pred, i, j), succ[-2:], 1))
                else:
                    rules.append((pred, succ, weight))
        return PCFG.from_rule_list(self.start, rules)

    def nullables(self) -> Set[Letter]:
        """
        Returns the set of nullable nonterminals in the grammar, paired
        with the sets of nonterminals that, when nulled, null the initial
        nonterminal.
        """
        # find all rules that produce the empty string
        srcs = [
            nt
            for nt, prods in self.rules.items()
            if nt != self.start and [""] in prods
        ]
        if not srcs:
            return {}

        # recursively set nonterminals as nullable or not
        # using dynamic programming
        cache = {nt: True for nt in srcs}
        cache[self.start] = False

        def is_nullable(letter) -> bool:
            out = None
            if letter in cache:
                return cache[letter]
            elif self.is_terminal(letter):
                out = False
            elif any(all(is_nullable(c) for c in rule)
                     for rule in self.rules[letter]):
                out = True
            else:
                out = False

            cache[letter] = out
            return out

        return {nt for nt in self.rules if is_nullable(nt)}

    def _del(self) -> 'PCFG':
        """
        Eliminate rules of the form A -> eps, where A != S.
        """
        # FIXME: assumes that all nulling patterns are equally likely
        nullable_nts = self.nullables()
        rules = []

        for pred, succ, weight in self.as_rule_list():
            # if a nonterminal in the successor is nullable, then
            # then add a version of the rule that does not contain
            # the nullable successor
            succs = []
            nullable_i = [i for i, c in enumerate(succ) if c in nullable_nts]
            for indices in util.language_plus(nullable_i):
                s = util.remove_from_string(succ, indices)
                if s and s not in self.rules[pred]:
                    succs.append(s)

            if succs:
                w = weight / (len(succs) + 1)
                if succ != ['']:
                    rules.append((pred, succ, w))
                for s in succs:
                    rules.append((pred, s, w))
            elif succ != ['']:
                rules.append((pred, succ, weight))

        condensed_rules = []
        for (pred, succ), grp in it.groupby(sorted(rules),
                                            key=lambda x: x[:2]):
            weights = [x[2] for x in grp]
            condensed_rules.append((pred, succ, sum(weights)))

        return PCFG.from_rule_list(self.start, condensed_rules)

    def _unit(self) -> 'PCFG':
        """
        Eliminate rules of the form A -> B, where A and B
        are both nonterminals.
        """
        cache = {}
        contracted = set()

        def contract(nt) -> List:
            if nt in cache:
                return cache[nt]

            rules = []
            for s, w in zip(self.rules[nt], self.weights[nt]):
                if len(s) > 1 or self.is_terminal(s[0]):
                    rules.append((s, w))
                else:
                    contracted.add(s[0])
                    for ss, sw in contract(s[0]):
                        rules.append((ss, w * sw))

            cache[nt] = rules
            return rules

        return PCFG.from_rule_list(
            self.start,
            [(p, s, w)
             for p in self.rules
             if p not in contracted
             for s, w in contract(p)]
        )

    def __str__(self) -> str:
        rules = "\n  ".join(
            f"{pred} ->\n    " +
            "\n    ".join(f"{succ} @ {weight:.2f}"
                          for succ, weight in zip(self.rules[pred],
                                                  self.weights[pred]))
            for pred in self.rules
        )
        return ("PCFG: {\n  start=" + self.start +
                "\n  rules=\n  " + rules + "\n}")

    def rule_eq(self, other: 'PCFG') -> bool:
        if self.rules.keys() != other.rules.keys():
            return False
        return all(sorted(self.rules[k]) == sorted(other.rules[k])
                   for k in self.rules)

    def _choose_successor(self, letter: str) -> List[str]:
        if letter not in self.rules:
            return [letter]
        else:
            return random.choices(population=self.rules[letter],
                                  weights=self.weights[letter],
                                  k=1)[0]


def test_term():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B", "C", "D", "E", "F"],
                      ["D", "E", "F"],
                      [""]],
                "B": [["b"]],
                "C": [["c1", "c2", "c3"],
                      ["c4"],
                      [""]],
                "D": [["D", "d", "D"]],
                "E": [["e"]],
                "F": [["f"]],
            },
            weights="uniform"),
         PCFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B", "C", "D", "E", "F"],
                      ["D", "E", "F"],
                      [""]],
                "B": [["b"]],
                "C": [["_TERM_c1_", "_TERM_c2_", "_TERM_c3_"],
                      ["c4"],
                      [""]],
                "D": [["D", "_TERM_d_", "D"]],
                "E": [["e"]],
                "F": [["f"]],
                "_TERM_c1_": [["c1"]],
                "_TERM_c2_": [["c2"]],
                "_TERM_c3_": [["c3"]],
                "_TERM_d_": [["d"]],
            },
            weights="uniform")),
    ]
    for x, y in cases:
        out = x._term()
        assert out == y, f"Failed test_term: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_term")


def test_bin():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B", "C", "D", "E", "F"],
                      ["D", "E", "F"]],
                "B": [["b"]],
                "C": [["c"]],
                "D": [["d"]],
                "E": [["e"]],
                "F": [["f"]],
            },
            weights="uniform",
        ),
            PCFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B", "_BIN_A_0_1_"],
                      ["D", "_BIN_A_1_1_"]],
                "_BIN_A_0_1_": [["C", "_BIN_A_0_2_"]],
                "_BIN_A_0_2_": [["D", "_BIN_A_0_3_"]],
                "_BIN_A_0_3_": [["E", "F"]],
                "_BIN_A_1_1_": [["E", "F"]],
                "B": [["b"]],
                "C": [["c"]],
                "D": [["d"]],
                "E": [["e"]],
                "F": [["f"]],
            },
            weights="uniform",
        )),
    ]
    for x, y in cases:
        out = x._bin()
        assert out.rule_eq(y), f"Failed test_bin: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_bin")


def test_nullable():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],
                "A": [["a"], [""]],
            },
            weights="uniform",
        ), {"A"}),
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],
                "A": [["a"]],
            },
            weights="uniform",
        ), {}),
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],       # nullable
                "A": [["B"], ["C", "a"]],  # nullable
                "B": [["C"]],              # nullable
                "C": [["x"], [""]],        # nullable
            },
            weights="uniform",
        ), {"A", "B", "C"}),
    ]
    for x, y in cases:
        out = x.nullables()
        assert out == y, f"Expected {y}, got {out}"
    print(" [+] Passed test_nullable")


def test_del():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A", "b", "B"], ["C"]],
                "A": [["a"], [""]],
                "B": [["A", "A"], ["A", "C"]],
                "C": [["b"], ["c"]],
            },
            weights="uniform",
        ),
            PCFG(
            start="S",
            rules={
                "S": [["A", "b", "B"], ["C"], ["b", "B"], ["A", "b"], ["b"]],
                "A": [["a"]],
                "B": [["A", "A"], ["A", "C"], ["A"], ["C"]],
                "C": [["b"], ["c"]],
            },
            weights={
                "S": [1, 1, 0.33, 0.33, 0.33],
                "A": [1],
                "B": [0.5, 0.33, 0.83, 0.33],
                "C": [1, 1]
            }
        )),
        (PCFG(
            start="S",
            rules={
                "S": [["A", "s1"], ["A1"], ["A1", "s1"], ["A2", "A1", "s2"]],
                "A": [["A1"]],
                "A1": [["A2"]],
                "A2": [["a2"], [""]],
            },
            weights="uniform",
         ),
         PCFG(
            start="S",
            rules={
                "S": [["A", "s1"], ["s1"],
                      ["A1"],
                      ["A1", "s1"],
                      ["A2", "A1", "s2"], ["A2", "s2"], ["A1", "s2"], ["s2"]],
                "A": [["A1"]],
                "A1": [["A2"]],
                "A2": [["a2"]]
            },
            weights={
                "S": [0.5, 0.5,
                      1,
                      1,
                      0.25, 0.25, 0.25, 0.25],
                "A": [1],
                "A1": [1],
                "A2": [1],
            }
        )),
    ]
    # FIXME: account for weights
    for x, y in cases:
        out = x._del()
        assert out.rule_eq(y), f"Failed test_del: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_del")


def test_unit():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s", "s", "s"]],
                "A": [["a", "b", "c"], ["e", "f"]],
            },
            weights="uniform",
         ),
         PCFG(
            start="S",
            rules={
                "S": [["a", "b", "c"], ["e", "f"], ["s", "s", "s"]]
            },
            weights="uniform",
         )),
        (PCFG(
            start="S",
            rules={
                "S": [["A", "B"], ["C"]],
                "A": [["a"]],
                "B": [["b"]],
                "C": [["c"]],
            },
            weights="uniform",
         ),
         PCFG(
            start="S",
            rules={
                "S": [["A", "B"], ["c"]],
                "A": [["a"]],
                "B": [["b"]],
            },
            weights="uniform",
         )),
        (PCFG(
            start="S",
            rules={
                "S": [["A"]],
                "A": [["B"]],
                "B": [["C"]],
                "C": [["c"]],
            },
            weights="uniform",
         ),
         PCFG(
            start="S",
            rules={
                "S": [["c"]],
            },
            weights="uniform",
         )),
    ]
    for x, y in cases:
        out = x._unit()
        assert out.rule_eq(y), \
            f"Failed test_unit: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_unit")


def test_to_CNF():
    test_term()
    test_bin()
    test_nullable()
    test_del()
    test_unit()


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

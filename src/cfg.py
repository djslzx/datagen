import random
import itertools as it
from typing import Dict, List, Tuple, Union, Set
import pdb
import torch as T

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
        assert all(succ and (isinstance(succ, str) or (isinstance(succ, list)
                                                       and all(succ)))
                   for pred, succ in rules.items()), \
            "All rule RHS should be nonempty; " \
            "each element should also be nonempty"
        assert all(succ == start or any(succ in other_succs
                                        for other_succs in rules.values())
                   for pred, succs in rules.items()
                   for succ in succs), \
            "Each nonterminal should appear in the RHS of a rule, " \
            "unless the nonterminal is the start symbol"
        self.start = start
        self.rules = {
            pred: [(succ.split() if isinstance(succ, str) else succ)
                   for succ in succs]
            for pred, succs in rules.items()
        }

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

    def step(self, word: List[str]) -> List[str]:
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
        prev = [self.start]
        current = self.step(self.start)
        while current != prev:
            prev = current
            current = self.step(current)
        return current

    def iterate_fully(self, debug=False) -> List[str]:
        s = [self.start]
        if debug:
            print(s)
        while any(self.is_nonterminal(w) for w in s):
            s = self.step(s)
            if debug:
                print(" ".join(s))
        return s

    def iterate(self, n: int) -> List[str]:
        """Apply rules to the starting word `n` times."""
        s = [self.start]
        for _ in range(n):
            s = self.step(s)
        return s

    def iterate_until(self, length: int) -> List[str]:
        """Apply rules to the starting word until its length is >= `length`."""
        s = [self.start]
        while len(s) < length:
            cache = s
            s = self.step(s)
            if s == cache:
                break
        return s


class PCFG(T.nn.Module, CFG):
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
    Word = str
    Sentence = List[Word]
    Eps = 'ε'
    Empty = [Eps]

    def __init__(self,
                 start: Word,
                 rules: Dict[Word, List[Union[str, Sentence]]],
                 weights: Union[str, Dict[Word, List[float]]] = "uniform"):
        PCFG.check_rep(start, rules, weights)
        super(PCFG, self).__init__()
        self.start = start
        self.rules = {
            pred: [(succ.split() if isinstance(succ, str) else succ)
                   for succ in succs]
            for pred, succs in rules.items()
        }
        if weights == "uniform":
            self.weights = T.nn.ParameterDict({
                k: T.ones(len(v), dtype=T.float64) / len(v)
                for k, v in rules.items()
            })
        else:
            self.weights = T.nn.ParameterDict({
                k: T.tensor(v, dtype=T.float64) / sum(v)
                for k, v in weights.items()
            })

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def check_rep(start, rules, weights):
        assert start in rules, f"Starting word {start} not found in rules"
        assert all(succ and (isinstance(succ, str) or (isinstance(succ, list)
                                                       and all(succ)))
                   for pred, succ in rules.items()), \
            "All rule RHS should be nonempty; " \
            "each element should also be nonempty"
        assert all(succ == start or any(succ in other_succs
                                        for other_succs in rules.values())
                   for succs in rules.values()
                   for succ in succs), \
            "Each nonterminal should appear in the RHS of a rule, " \
            "unless the nonterminal is the start symbol"
        for pred, succs in rules.items():
            ok, pair = util.unique(succs)
            assert ok, \
                "All successors should be unique wrt a predecessor, " \
                f"but got duplicate {pair} in {pred} -> {succs}"

    def __eq__(self, other):
        return self.approx_eq(other, threshold=10 ** -2)

    def approx_eq(self, other, threshold):
        """
        Checks whether two PCFGs are structurally (not semantically) equivalent
        and have roughly the same parameters.

        This does not check whether the grammars produce the same set of
        strings.
        """
        return isinstance(other, PCFG) and \
            self.rules == other.rules and \
            all(util.approx_eq(w1, w2, threshold)
                for nt in self.rules
                for w1, w2 in zip(self.weights[nt], other.weights[nt]))

    def struct_eq(self, other: 'PCFG') -> bool:
        """
        Checks whether two PCFGs have the same structure.
        """
        if self.rules.keys() != other.rules.keys():
            return False
        return all(sorted(self.rules[k]) == sorted(other.rules[k])
                   for k in self.rules)

    def is_in_CNF(self) -> bool:
        """
        Checks whether the PCFG is in Chomsky normal form.
        In CNF, all rules should be of the form:
         - A -> BC
         - A -> a
         - S -> empty
        """
        for p, xs, _ in self.as_rule_list():
            if len(xs) > 2 or len(xs) < 1:
                return False
            elif len(xs) == 1:
                a = xs[0]
                if (self.is_nonterminal(a) or
                   (p != self.start and xs == PCFG.Empty)):
                    return False
            elif len(xs) == 2:
                B, C = xs
                if (self.is_terminal(B) or self.is_terminal(C) or
                   B == self.start or C == self.start):
                    return False
        return True

    def normalized(self, c=0.1) -> 'PCFG':
        """
        Returns a PCFG with similar relative weights, but normalized
        with smoothing constant c.
        """
        return PCFG(
            start=self.start,
            rules=self.rules,
            weights={
                pred: (ws + c) / T.sum(ws + c)
                for pred, ws in self.weights.items()
            }
        )

    def weight(self, pred: Word, succ: Sentence) -> float:
        if pred in self.rules:
            for s, w in zip(self.rules[pred], self.weights[pred]):
                if s == succ:
                    return w
        return 0

    def __len__(self) -> int:
        return sum(len(succs) for pred, succs in self.rules.items())

    @property
    def nonterminals(self) -> List[Word]:
        return self.rules.keys()

    def from_rule_list(start: Word,
                       rules: List[Tuple[Word, Sentence, float]]) -> 'PCFG':
        """Construct a PCFG from a list of rules with weights"""
        words = {}
        weights = {}
        for letter, word, weight in sorted(rules, key=lambda x: x[0]):
            if letter not in words:
                words[letter] = [word]
                weights[letter] = [weight]
            else:
                words[letter].append(word)
                weights[letter].append(weight)
        return PCFG(start, words, weights)

    def as_rule_list(self) -> List[Tuple[Word, Sentence, float]]:
        """View a PCFG as a list of rules with weights"""
        return [
            (letter, word, weight)
            for letter in self.rules
            for word, weight in zip(self.rules[letter], self.weights[letter])
        ]

    def add_rule(self, pred: Word, succ: Sentence, weight: float) -> 'PCFG':
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

    def to_bigram(self) -> 'PCFG':
        """
        Expand the PCFG into a bigram model.

        Annotate all RHS appearances of a nonterminal with an index,
        then duplicate the resulting rules with the LHS being each
        unique annotated RHS.

        Given an nt X appearing in k RHS's, annotate each X with an index i
        to yield X_i, which replaces each RHS appearance of X.
        Then, add rules X_i -> ... for each i.
        """
        # annotate nonterminals with indices
        rules = []
        nt_counts = {nt: 0 for nt in self.nonterminals}
        for p, s, w in self.as_rule_list():
            new_s = []
            for word in s:
                if self.is_nonterminal(word):
                    nt_counts[word] += 1
                    new_s.append(f"{word}_{nt_counts[word]}")
                else:
                    new_s.append(word)
            rules.append((p, new_s, w))

        # duplicate annotated rules
        duped_rules = []
        for p, s, w in rules:
            # handle nonterminals that never appear in a RHS
            # by taking max of (1, n)
            n = max(1, nt_counts[p])
            for i in range(1, n+1):
                duped_rules.append((f"{p}_{i}", s, w))

        # add transitions to transformed start symbol
        n = max(1, nt_counts[self.start])
        for i in range(1, n+1):
            duped_rules.append((self.start, [f"{self.start}_{i}"], 1/n))

        return PCFG.from_rule_list(self.start, duped_rules)

    def explode(self) -> 'PCFG':
        """
        Explode the grammar exponentially.
        """
        # count the number of appearances of each nonterminal
        nt_counts = {nt: 0 for nt in self.nonterminals}
        for p, s, w in self.as_rule_list():
            for word in s:
                if self.is_nonterminal(word):
                    nt_counts[word] += 1

        # annotate RHS nonterminals with all possible indices
        annotated_rules = []
        for p, s, w in self.as_rule_list():
            factors = []
            for word in s:
                if self.is_nonterminal(word):
                    n = max(1, nt_counts[word])
                    factors.append([f"{word}_{i}"
                                    for i in range(1, n+1)])
                else:
                    factors.append([word])

            prod = list(it.product(*factors))
            m = len(prod)
            for new_s in prod:
                annotated_rules.append((p, list(new_s), w/m))

        # duplicate LHS of annotated rules
        duped_rules = []
        for p, s, w in annotated_rules:
            n = max(1, nt_counts[p])
            for i in range(1, n+1):
                duped_rules.append((f"{p}_{i}", s, w))

        # add transitions to transformed start symbol
        n = max(1, nt_counts[self.start])
        rules = duped_rules + [(self.start, [f"{self.start}_{i}"], 1/n)
                               for i in range(1, n+1)]

        return PCFG.from_rule_list(self.start, rules)

    def to_CNF(self, debug=False) -> 'PCFG':
        """Convert to Chomsky normal form; immutable"""
        if self.is_in_CNF():
            return self

        ops = {
            'start': lambda x: x._start(),
            'term': lambda x: x._term(),
            'bin': lambda x: x._bin(),
            'del': lambda x: x._del(),
            'unit': lambda x: x._unit(),
        }
        g = self
        for name, op in ops.items():
            g_new = op(g)
            if debug:
                print(f"Performing {name} on\n{g}\n"
                      f"yielded\n{g_new}")
            g = g_new
        return g

    def _start(self) -> 'PCFG':
        """Eliminate the start symbol from any RHS"""
        return PCFG.from_rule_list(
            "_start_",
            self.as_rule_list() + [("_start_", [self.start], 1)]
        )

    def _term(self) -> 'PCFG':
        """Eliminate rules with nonsolitary terminals"""
        def nt(c) -> str:
            return f"_term_{c}_"

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
                            nt_rule = (nt(c), [c], 1)
                            if nt_rule not in rules:
                                rules.append(nt_rule)
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
            return f"_bin_{pred}_{i}_{j}_"

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

    def nullables(self) -> Set[Word]:
        """
        Returns the set of nullable nonterminals in the grammar, paired
        with the sets of nonterminals that, when nulled, null the initial
        nonterminal.
        """
        # find all rules that produce the empty string
        srcs = [
            nt
            for nt, prods in self.rules.items()
            if nt != self.start and PCFG.Empty in prods
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
            elif any(all(is_nullable(c) for c in rule if c != letter)
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
                if succ != PCFG.Empty:
                    rules.append((pred, succ, w))
                for s in succs:
                    rules.append((pred, s, w))
            elif succ != PCFG.Empty:
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
                # remove identity productions
                if s == [nt]:
                    continue
                h = s[0]
                if len(s) > 1 or self.is_terminal(h):
                    rules.append((s, w))
                else:
                    contracted.add(h)
                    for ss, sw in contract(h):
                        rules.append((ss, w * sw))
            cache[nt] = rules
            return cache[nt]

        def used(nt, rules) -> bool:
            """
            Determines whether a nonterminal is used in the
            productions of other nonterminals in `rules`.
            """
            return any(nt != p and nt in s
                       for p, s, _ in rules)

        # run contract()
        rules = [(p, s, w)
                 for p in self.rules
                 for s, w in contract(p)]

        # filter out unused nonterminals (disconnected graph components)
        filtered_rules = []
        for p, s, w in rules:
            if p == self.start or used(p, rules) \
               and (p, s, w) not in filtered_rules:
                filtered_rules.append((p, s, w))

        return PCFG.from_rule_list(self.start, filtered_rules)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        def denote_t_or_nt(xs):
            return [f"{x}" if self.is_nonterminal(x) else f"`{x}`"
                    for x in xs]

        rules = "\n  ".join(
            f"{pred} ->\n    " +
            "\n    ".join(f"{' '.join(denote_t_or_nt(succ))} @ {weight:.10f}"
                          for succ, weight in zip(self.rules[pred],
                                                  self.weights[pred]))
            for pred in self.rules
        )
        return ("PCFG: {\n  start=" + self.start +
                "\n  rules=\n  " + rules + "\n}")

    def _choose_successor(self, letter: str) -> List[str]:
        if letter not in self.rules:
            return [letter]
        else:
            return random.choices(population=self.rules[letter],
                                  weights=self.weights[letter],
                                  k=1)[0]


def test_explode():
    cases = [
        (PCFG.from_rule_list(
            start="S",
            rules=[
                ("S", ["A"], 1),
                ("A", ["a"], 1),
            ],
        ), PCFG.from_rule_list(
            start="S",
            rules=[
                ("S", ["S_1"], 1),

                ("S_1", ["A_1"], 1),
                ("A_1", ["a"], 1),
            ],
        )),
        (PCFG.from_rule_list(
            start="E",
            rules=[
                ("E", ["-", "E"], 0.5),
                ("E", ["E", "+", "E"], 0.5),
            ],
        ), PCFG.from_rule_list(
            start="E",
            rules=[
                ("E", ["E_1"], 0.333),
                ("E", ["E_2"], 0.333),
                ("E", ["E_3"], 0.333),

                ("E_1", ["-", "E_1"], 0.5 / 3),
                ("E_1", ["-", "E_2"], 0.5 / 3),
                ("E_1", ["-", "E_3"], 0.5 / 3),
                ("E_1", ["E_1", "+", "E_1"], 0.5 / 9),
                ("E_1", ["E_1", "+", "E_2"], 0.5 / 9),
                ("E_1", ["E_1", "+", "E_3"], 0.5 / 9),
                ("E_1", ["E_2", "+", "E_1"], 0.5 / 9),
                ("E_1", ["E_2", "+", "E_2"], 0.5 / 9),
                ("E_1", ["E_2", "+", "E_3"], 0.5 / 9),
                ("E_1", ["E_3", "+", "E_1"], 0.5 / 9),
                ("E_1", ["E_3", "+", "E_2"], 0.5 / 9),
                ("E_1", ["E_3", "+", "E_3"], 0.5 / 9),

                ("E_2", ["-", "E_1"], 0.5 / 3),
                ("E_2", ["-", "E_2"], 0.5 / 3),
                ("E_2", ["-", "E_3"], 0.5 / 3),
                ("E_2", ["E_1", "+", "E_1"], 0.5 / 9),
                ("E_2", ["E_1", "+", "E_2"], 0.5 / 9),
                ("E_2", ["E_1", "+", "E_3"], 0.5 / 9),
                ("E_2", ["E_2", "+", "E_1"], 0.5 / 9),
                ("E_2", ["E_2", "+", "E_2"], 0.5 / 9),
                ("E_2", ["E_2", "+", "E_3"], 0.5 / 9),
                ("E_2", ["E_3", "+", "E_1"], 0.5 / 9),
                ("E_2", ["E_3", "+", "E_2"], 0.5 / 9),
                ("E_2", ["E_3", "+", "E_3"], 0.5 / 9),

                ("E_3", ["-", "E_1"], 0.5 / 3),
                ("E_3", ["-", "E_2"], 0.5 / 3),
                ("E_3", ["-", "E_3"], 0.5 / 3),
                ("E_3", ["E_1", "+", "E_1"], 0.5 / 9),
                ("E_3", ["E_1", "+", "E_2"], 0.5 / 9),
                ("E_3", ["E_1", "+", "E_3"], 0.5 / 9),
                ("E_3", ["E_2", "+", "E_1"], 0.5 / 9),
                ("E_3", ["E_2", "+", "E_2"], 0.5 / 9),
                ("E_3", ["E_2", "+", "E_3"], 0.5 / 9),
                ("E_3", ["E_3", "+", "E_1"], 0.5 / 9),
                ("E_3", ["E_3", "+", "E_2"], 0.5 / 9),
                ("E_3", ["E_3", "+", "E_3"], 0.5 / 9),
            ],
        )),
        (PCFG.from_rule_list(
            start="A",
            rules=[
                ("A", ["a"], 0.5),
                ("A", ["B", "C"], 0.5),

                ("B", ["B", "B"], 0.5),
                ("B", ["C"], 0.5),

                ("C", ["A"], 1),
            ],
        ), PCFG.from_rule_list(
            start="A",
            rules=[
                ("A", ["A_1"], 1),

                ("A_1", ["a"], 0.5),
                ("A_1", ["B_1", "C_1"], 0.5 / 6),
                ("A_1", ["B_1", "C_2"], 0.5 / 6),
                ("A_1", ["B_2", "C_1"], 0.5 / 6),
                ("A_1", ["B_2", "C_2"], 0.5 / 6),
                ("A_1", ["B_3", "C_1"], 0.5 / 6),
                ("A_1", ["B_3", "C_2"], 0.5 / 6),

                ("B_1", ["B_1", "B_1"], 0.5 / 9),
                ("B_1", ["B_1", "B_2"], 0.5 / 9),
                ("B_1", ["B_1", "B_3"], 0.5 / 9),
                ("B_1", ["B_2", "B_1"], 0.5 / 9),
                ("B_1", ["B_2", "B_2"], 0.5 / 9),
                ("B_1", ["B_2", "B_3"], 0.5 / 9),
                ("B_1", ["B_3", "B_1"], 0.5 / 9),
                ("B_1", ["B_3", "B_2"], 0.5 / 9),
                ("B_1", ["B_3", "B_3"], 0.5 / 9),
                ("B_1", ["C_1"], 0.5 / 2),
                ("B_1", ["C_2"], 0.5 / 2),

                ("B_2", ["B_1", "B_1"], 0.5 / 9),
                ("B_2", ["B_1", "B_2"], 0.5 / 9),
                ("B_2", ["B_1", "B_3"], 0.5 / 9),
                ("B_2", ["B_2", "B_1"], 0.5 / 9),
                ("B_2", ["B_2", "B_2"], 0.5 / 9),
                ("B_2", ["B_2", "B_3"], 0.5 / 9),
                ("B_2", ["B_3", "B_1"], 0.5 / 9),
                ("B_2", ["B_3", "B_2"], 0.5 / 9),
                ("B_2", ["B_3", "B_3"], 0.5 / 9),
                ("B_2", ["C_1"], 0.5 / 2),
                ("B_2", ["C_2"], 0.5 / 2),

                ("B_3", ["B_1", "B_1"], 0.5 / 9),
                ("B_3", ["B_1", "B_2"], 0.5 / 9),
                ("B_3", ["B_1", "B_3"], 0.5 / 9),
                ("B_3", ["B_2", "B_1"], 0.5 / 9),
                ("B_3", ["B_2", "B_2"], 0.5 / 9),
                ("B_3", ["B_2", "B_3"], 0.5 / 9),
                ("B_3", ["B_3", "B_1"], 0.5 / 9),
                ("B_3", ["B_3", "B_2"], 0.5 / 9),
                ("B_3", ["B_3", "B_3"], 0.5 / 9),
                ("B_3", ["C_1"], 0.5 / 2),
                ("B_3", ["C_2"], 0.5 / 2),

                ("C_1", ["A_1"], 1),

                ("C_2", ["A_1"], 1),
            ],
        )),
    ]
    for g, y in cases:
        out = g.explode()
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_explode")


def test_to_bigram():
    # annotate repeated nonterminals with indices, then make
    # copies of the original rules with however many indices were used
    cases = [
        (PCFG.from_rule_list(
            start="S",
            rules=[
                ("S", ["a"], 1),
            ],
        ), PCFG.from_rule_list(
            start="S",
            rules=[
                ("S", ["S_1"], 1),
                ("S_1", ["a"], 1),
            ],
        )),
        (PCFG.from_rule_list(
            start="S",
            rules=[
                ("S", ["A"], 1),
                ("A", ["a"], 1),
            ],
        ), PCFG.from_rule_list(
            start="S",
            rules=[
                ("S", ["S_1"], 1),
                ("S_1", ["A_1"], 1),
                ("A_1", ["a"], 1),
            ],
        )),
        (PCFG.from_rule_list(
            start="S",
            rules=[
                ("S", ["a", "S"], 0.333),
                ("S", ["b", "S"], 0.333),
                ("S", ["c", "S"], 0.333),
            ],
        ), PCFG.from_rule_list(
            start="S",
            rules=[
                ("S", ["S_1"], 0.333),
                ("S", ["S_2"], 0.333),
                ("S", ["S_3"], 0.333),

                # 3 rules ^ 2
                ("S_1", ["a", "S_1"], 0.333),
                ("S_1", ["b", "S_2"], 0.333),
                ("S_1", ["c", "S_3"], 0.333),

                ("S_2", ["a", "S_1"], 0.333),
                ("S_2", ["b", "S_2"], 0.333),
                ("S_2", ["c", "S_3"], 0.333),

                ("S_3", ["a", "S_1"], 0.333),
                ("S_3", ["b", "S_2"], 0.333),
                ("S_3", ["c", "S_3"], 0.333),
            ],
        )),
        (PCFG.from_rule_list(
            start="E",
            rules=[
                ("E", ["-", "E"], 0.5),
                ("E", ["E", "+", "E"], 0.5),
            ],
        ), PCFG.from_rule_list(
            start="E",
            rules=[
                ("E", ["E_1"], 0.333),
                ("E", ["E_2"], 0.333),
                ("E", ["E_3"], 0.333),

                # 2 rules ^ 2
                ("E_1", ["-", "E_1"], 0.5),
                ("E_1", ["E_2", "+", "E_3"], 0.5),

                ("E_2", ["-", "E_1"], 0.5),
                ("E_2", ["E_2", "+", "E_3"], 0.5),

                ("E_3", ["-", "E_1"], 0.5),
                ("E_3", ["E_2", "+", "E_3"], 0.5),
            ],
        )),
        (PCFG.from_rule_list(
            start="A",
            rules=[
                ("A", ["a"], 0.25),
                ("A", ["B", "C"], 0.25),
                ("A", ["B"], 0.25),
                ("A", ["C"], 0.25),
                ("B", ["B", "B"], 0.5),
                ("B", ["C"], 0.5),
                ("C", ["A"], 1),
            ],
        ), PCFG.from_rule_list(
            start="A",
            rules=[
                ("A", ["A_1"], 1),

                ("A_1", ["a"], 0.25),
                ("A_1", ["B_1", "C_1"], 0.25),
                ("A_1", ["B_2"], 0.25),
                ("A_1", ["C_2"], 0.25),

                ("B_1", ["B_3", "B_4"], 0.5),
                ("B_1", ["C_3"], 0.5),
                ("B_2", ["B_3", "B_4"], 0.5),
                ("B_2", ["C_3"], 0.5),
                ("B_3", ["B_3", "B_4"], 0.5),
                ("B_3", ["C_3"], 0.5),
                ("B_4", ["B_3", "B_4"], 0.5),
                ("B_4", ["C_3"], 0.5),

                ("C_1", ["A_1"], 1),
                ("C_2", ["A_1"], 1),
                ("C_3", ["A_1"], 1),
            ],
        )),
    ]
    for g, y in cases:
        out = g.to_bigram()
        assert out == y, f"Expected {y}, but got {out}"
    print(" [+] passed test_to_bigram")


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
                "C": [["_term_c1_", "_term_c2_", "_term_c3_"],
                      ["c4"],
                      [""]],
                "D": [["D", "_term_d_", "D"]],
                "E": [["e"]],
                "F": [["f"]],
                "_term_c1_": [["c1"]],
                "_term_c2_": [["c2"]],
                "_term_c3_": [["c3"]],
                "_term_d_": [["d"]],
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
                "A": [["B", "_bin_A_0_1_"],
                      ["D", "_bin_A_1_1_"]],
                "_bin_A_0_1_": [["C", "_bin_A_0_2_"]],
                "_bin_A_0_2_": [["D", "_bin_A_0_3_"]],
                "_bin_A_0_3_": [["E", "F"]],
                "_bin_A_1_1_": [["E", "F"]],
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
        assert out.struct_eq(y), f"Failed test_bin: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_bin")


def test_nullable():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A"], ["s"]],
                "A": [["a"], PCFG.Empty],
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
                "C": [["x"], PCFG.Empty],        # nullable
            },
            weights="uniform",
        ), {"A", "B", "C"}),
    ]
    for x, y in cases:
        out = x.nullables()
        assert out == y, f"Expected {y}, got {out}"
    print(" [+] passed test_nullable")


def test_del():
    cases = [
        (PCFG(
            start="S",
            rules={
                "S": [["A", "b", "B"], ["C"]],
                "A": [["a"], PCFG.Empty],
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
                "A2": [["a2"], PCFG.Empty],
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
        assert out.struct_eq(y), f"Failed test_del: Expected\n{y},\ngot\n{out}"
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
                "S": [["a", "b", "c"], ["e", "f"], ["s", "s", "s"]],
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
        (PCFG(
            start='S0',
            rules={
                'S0': [['S']],
                'S': [['A'], ['A', 'B']],
                'A': [['A', 'A']],
                'B': [['B', 'A']],
            },
            weights="uniform",
         ),
         PCFG(
             start='S0',
             rules={
                 'S0': [['A', 'A'], ['A', 'B']],
                 'A': [['A', 'A']],
                 'B': [['B', 'A']],
             },
             weights='uniform',
        )),
    ]
    for x, y in cases:
        out = x._unit()
        assert out.struct_eq(y), \
            f"Failed test_unit: Expected\n{y},\ngot\n{out}"
    print(" [+] passed test_unit")


def test_to_CNF():
    test_term()
    test_bin()
    test_nullable()
    test_del()
    test_unit()


def demo_to_CNF():
    pcfgs = [
        # PCFG(
        #     start="S",
        #     rules={
        #         "S": [["A", "B"], ["A"]],
        #         "A": [["A", "A"]],
        #         "B": [["B", "A"]],
        #     },
        #     weights="uniform",
        # ),
        PCFG(
            start="AXIOM",
            rules={
                "AXIOM": [
                    ["M", "F"],
                ],
                "M": [
                    ["M", "F", "+"],
                    ["M", "F", "-"],
                    ["F"],
                ],
                "RHS": [
                    ["F", "[", "PLUSES", "+", "F", "INNER", "]", "RHS", "F"],
                    ["F", "[", "MINUSES", "-", "F", "INNER", "]", "RHS", "F"],
                    ["F", "INNER"],
                ],
                "INNER": [
                    ["INNER", "PLUSES", "FS"],
                    ["INNER", "MINUSES", "FS"],
                    ["INNER", "FS"],
                ],
                "PLUSES": [
                    ["+", "PLUSES"],
                    ["+"],
                ],
                "MINUSES": [
                    ["-", "MINUSES"],
                    ["-"],
                ],
                "FS": [
                    ["FS", "F"],
                    ["F"],
                ],
            },
            weights="uniform",
        ),
    ]
    for pcfg in pcfgs:
        print(pcfg)
        print(pcfg.to_CNF())


def test_is_in_CNF():
    cases = [
        # unit
        ({
            "S": [["A"]],
            "A": [["a"]],
        }, False),
        ({
            "S": [["a"]],
        }, True),
        # term/nonterm mix
        ({
            "S": [["A", "b"]],
            "A": [["a"]],
        }, False),
        ({
            "S": [["A", "B"]],
            "A": [["a"]],
            "B": [["b"]],
        }, True),
        ({
            "S": [["A", "B"], PCFG.Empty],
            "A": [["a"]],
            "B": [["b"]],
        }, True),
        # three succs
        ({
            "S": [["A", "B", "C"]],
            "A": [["a"]],
            "B": [["b"]],
            "C": [["c"]],
        }, False),
        # empty successor in non-start nonterminal
        ({
            "S": [["A", "B"]],
            "A": [PCFG.Empty],
            "B": [["b"]],
        }, False),

    ]
    for rules, y in cases:
        g = PCFG("S", rules, "uniform")
        out = g.is_in_CNF()
        if out != y:
            print(f"Failed test_is_in_CNF for {g}: "
                  f"Expected {y}, but got {out}")
            pdb.set_trace()
            g.is_in_CNF()
            exit(1)
    print(" [+] passed test_is_in_CNF")


if __name__ == '__main__':
    # cfg = CFG(
    #     start="a",
    #     rules={
    #         "a": [["a", "b"], ["a"]],
    #         "b": [["b", "b"], ["b"]],
    #     }
    # )
    # pcfg = PCFG(
    #     start="a",
    #     rules={
    #         "a": [["b", "a"], ["a"]],
    #         "b": [["b"], ["c", "b"]],
    #     },
    #     weights="uniform",
    # )
    # print(cfg)
    # print(cfg.iterate(10))

    # print(pcfg)
    # print(pcfg.iterate(10))

    test_to_CNF()
    test_to_bigram()
    test_explode()
    test_is_in_CNF()
    # demo_to_CNF()

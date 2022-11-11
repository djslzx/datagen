import math
import random
import itertools as it
from typing import Dict, List, Tuple, Union, Set, Iterable
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
    Word = str
    Sentence = List[Word]
    Eps = 'ε'
    Empty = [Eps]

    def __init__(self, start: str, rules: Dict[str, List[Union[str, Sentence]]]):
        CFG.check_rep(start, rules)
        self.start = start
        self.rules = {
            pred: [(succ.split() if isinstance(succ, str) else succ)
                   for succ in succs]
            for pred, succs in rules.items()
        }

    @staticmethod
    def check_rep(start, rules):
        if start not in rules:
            raise ValueError(f"Starting symbol {start} not found in rules")
        if not all(succ and (isinstance(succ, str) or
                             (isinstance(succ, list) and all(succ)))
                   for pred, succ in rules.items()):
            raise ValueError("All rule RHS should be nonempty, "
                             "and each element should also be nonempty")
        if not all(succ == start or any(succ in other_succs
                                        for other_succs in rules.values())
                   for pred, succs in rules.items()
                   for succ in succs):
            raise ValueError("Each nonterminal should appear in the RHS of a rule, "
                             "unless the nonterminal is the start symbol")
        for pred, succs in rules.items():
            ok, pair = util.unique(succs)
            if not ok:
                raise ValueError("All successors should be unique wrt a predecessor, "
                                 f"but got duplicate {pair} in {pred} -> {succs}")

    @staticmethod
    def from_rules(start: str, rules: List[Tuple[Word, Sentence]]) -> 'CFG':
        rules_dict = {}
        for pred, succ in rules:
            if pred in rules_dict:
                rules_dict[pred].append(succ)
            else:
                rules_dict[pred] = [succ]
        return CFG(start, rules_dict)

    def __eq__(self, other):
        if not isinstance(other, CFG):
            return False
        return (self.start == other.start and
                self.rules == other.rules)

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
        current = self.step([self.start])
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
    represented as strings.  All the rules for a given nonterminal should
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
                 weights: Union[str, Dict[Word, List[float]]] = "uniform",
                 log_mode=False):
        CFG.check_rep(start, rules)
        super(PCFG, self).__init__()

        self.start = start
        self.rules = {
            pred: [(succ.split() if isinstance(succ, str) else succ)
                   for succ in succs]
            for pred, succs in rules.items()
        }

        self.log_mode = log_mode
        maybe_log = (lambda x: x.log() if self.log_mode else x)

        if weights == "uniform":
            self.weights = T.nn.ParameterDict({
                k: maybe_log(T.ones(len(v), dtype=T.float64) / len(v))
                for k, v in rules.items()
            })
        else:
            self.weights = T.nn.ParameterDict({
                k: (T.tensor(v, dtype=T.float64) if not isinstance(v, T.Tensor)
                    else v.double())
                for k, v in weights.items()
            })

    def __hash__(self):
        return hash(str(self))

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
            self.start == other.start and \
            self.log_mode == other.log_mode and \
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

    def is_normalized(self, tolerance=1e-3) -> bool:
        for pred, weights in self.weights.items():
            if (not self.log_mode and abs(1 - sum(weights)) >= tolerance) or \
               (self.log_mode and abs(T.logsumexp(weights, dim=0).item()) >= tolerance):
                return False
        return True

    def weight(self, pred: Word, succ: Sentence) -> T.Tensor:
        if pred in self.rules:
            for s, w in zip(self.rules[pred], self.weights[pred]):
                if s == succ:
                    return w
        return T.tensor(-T.inf) if self.log_mode else T.tensor(0)

    def __len__(self) -> int:
        return sum(len(succs) for pred, succs in self.rules.items())

    def apply_to_weights(self, f) -> 'PCFG':
        return PCFG.from_weighted_rules(
            self.start,
            [(p, s, f(w)) for p, s, w in self.as_rule_list()]
        )

    def log(self) -> 'PCFG':
        g = self.apply_to_weights(T.log)
        g.log_mode = True
        return g

    def exp(self) -> 'PCFG':
        g = self.apply_to_weights(T.exp)
        g.log_mode = False
        return g

    @property
    def nonterminals(self) -> List[Word]:
        return list(self.rules.keys())

    @staticmethod
    def from_weighted_rules(start: Word, rules: Iterable[Tuple[Word, Sentence, float]]) -> 'PCFG':
        """Construct a PCFG from an iterable of weighted rules"""
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

    def as_rule_list(self) -> Iterable[Tuple[Word, Sentence, float]]:
        """View a PCFG as a list of rules with weights"""
        for nt in self.rules:
            for i, succ in enumerate(self.rules[nt]):
                yield nt, succ, self.weights[nt][i]

    def to_bigram(self) -> 'PCFG':
        """
        Expand the PCFG into a bigram model.

        Annotate all RHS appearances of a nonterminal with an index,
        then duplicate the resulting rules with the LHS being each
        unique annotated RHS.

        Given a nt X appearing in k RHS's, annotate each X with an index i
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

        return PCFG.from_weighted_rules(self.start, duped_rules)

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

        return PCFG.from_weighted_rules(self.start, rules)

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
        # TODO: move this to CFG so we don't have to muck around with weights
        return g

    def _start(self) -> 'PCFG':
        """Eliminate the start symbol from any RHS"""
        return PCFG.from_weighted_rules(
            "_start_",
            it.chain(self.as_rule_list(), [("_start_", [self.start], 1)])
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

        return PCFG.from_weighted_rules(self.start, rules)

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
        return PCFG.from_weighted_rules(self.start, rules)

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
            return set()

        # recursively set nonterminals as nullable or not
        # using dynamic programming
        cache = {nt: True for nt in srcs}
        cache[self.start] = False

        def is_nullable(letter) -> bool:
            if letter in cache:
                return cache[letter]
            cache[letter] = not self.is_terminal(letter) and \
                            any(all(is_nullable(c) for c in rule if c != letter)
                                for rule in self.rules[letter])
            return cache[letter]

        return {nt for nt in self.rules if is_nullable(nt)}

    def _del(self) -> 'PCFG':
        """
        Eliminate rules of the form A -> eps, where A != S.
        """
        # FIXME: assumes that all nulling patterns are equally likely
        nullable_nts = self.nullables()
        rules = []

        for pred, succ, weight in self.as_rule_list():
            # if a nonterminal in the successor is nullable,
            # then add a version of the rule that does not contain
            # the nullable successor
            succs = []
            nullable_i = {i for i, c in enumerate(succ) if c in nullable_nts}
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

        return PCFG.from_weighted_rules(self.start, condensed_rules)

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

        return PCFG.from_weighted_rules(self.start, filtered_rules)

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
        return ("PCFG: {"
                f"\n  log_mode={self.log_mode}"
                f"\n  start={self.start}"
                f"\n  rules=\n  {rules}\n}}")

    def _choose_successor(self, letter: str) -> List[str]:
        if letter not in self.rules:
            return [letter]
        else:
            return random.choices(population=self.rules[letter],
                                  weights=self.weights[letter],
                                  k=1)[0]

from __future__ import annotations

import torch as T
import numpy as np
import random
import itertools as it
from typing import *
import copy

from max_heap import MaxHeap
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

    def __init__(self, start: str, rules: Dict[str, Iterable[str | Sentence]]):
        CFG.check_rep(start, rules)
        self.start: CFG.Word = start
        self.rules: Dict[str, List[CFG.Sentence]] = {
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
        if any(pred in succs for pred, succs in rules.items()):
            raise ValueError("No nonterminal should map to itself")
        if any(succs == [CFG.Empty] or succs == CFG.Empty
               for succs in rules.values()):
            raise ValueError("All RHS should have at least one non-epsilon successor")
        if not all(pred == start or any(pred in succ
                                        for succs in rules.values()
                                        for succ in succs)
                   for pred in rules.keys()):
            raise ValueError("Each nonterminal should appear in the RHS of a rule, "
                             "unless the nonterminal is the start symbol")
        for pred, succs in rules.items():
            ok, pair = util.unique(succs)
            if not ok:
                raise ValueError("All successors should be unique wrt a predecessor, "
                                 f"but got duplicate {pair} in {pred} -> {succs}")

    @staticmethod
    def from_rules(start: str, rules: Iterable[Tuple[Word, Sentence]]) -> 'CFG':
        rules_dict = {}
        for pred, succ in rules:
            if pred in rules_dict:
                rules_dict[pred].append(succ)
            else:
                rules_dict[pred] = [succ]
        return CFG(start, rules_dict)

    def as_rules(self) -> Iterator[Tuple[Word, Sentence]]:
        """
        Returns an iterator over the individual rules in the grammar.
        """
        for pred, succs in self.rules.items():
            for succ in succs:
                yield pred, succ

    def __eq__(self, other):
        return (isinstance(other, CFG) and
                self.start == other.start and
                self.nonterminals == other.nonterminals and
                all(sorted(self.rules[nt]) == sorted(other.rules[nt])
                    for nt in self.nonterminals))

    def __str__(self):  # pragma: no cover
        rules = "\n  ".join(
            f"{pred} -> {succs}"
            for pred, succs in self.rules.items())
        return f"CFG({self.start}): ""{\n  " + rules + "\n}"

    def is_nonterminal(self, letter: str) -> bool:
        return letter in self.rules

    def is_terminal(self, letter: str) -> bool:
        return letter not in self.rules

    @property
    def nonterminals(self) -> Iterable[Word]:
        return self.rules.keys()

    def successor(self, nt: Word) -> Sentence:
        return random.choice(self.rules[nt])

    def step(self, sentence: List[str]) -> Sentence:
        """
        Non-deterministically apply one of the production rules to
        a word in the sentence.
        """
        # Only choose nonterminals to expand
        nonterminals = [i for i, word in enumerate(sentence)
                        if self.is_nonterminal(word)]
        if not nonterminals:
            return sentence
        index = random.choice(nonterminals)
        nt = sentence[index]
        expansion = self.successor(nt)
        return sentence[:index] + expansion + sentence[index + 1:]

    def iterate_fully(self, debug=False) -> Sentence:
        s = [self.start]
        if debug:  # pragma: no cover
            print(s)
        while any(self.is_nonterminal(w) for w in s):
            s = self.step(s)
            if debug:  # pragma: no cover
                print(" ".join(s))
        return s

    def iterate_until(self, length: int) -> Sentence:
        """
        Apply rules to the starting word until its length is >= `length`.
        Exit if a fixpoint is reached.
        """
        s = [self.start]
        while len(s) < length:
            cache = s
            s = self.step(s)
            if s == cache:
                break
        return s

    def can_generate(self, sentence: Sentence) -> bool:
        """Checks whether the grammar can produce a given sentence."""
        def has_parses(G: CFG, nt: CFG.Word, tgt: CFG.Sentence) -> bool:
            for succ in G.rules[nt]:
                if len(succ) == 1 and succ == tgt:
                    return True
                elif len(succ) == 2:
                    a, b = succ
                    for i in range(1, len(tgt)):
                        if has_parses(G, a, tgt[:i]) and has_parses(G, b, tgt[i:]):
                            return True
            return False

        g = self.to_CNF()
        return any(has_parses(g, nt, sentence) for nt in g.nonterminals)

    def to_bigram(self) -> 'CFG':
        """
        Expand the CFG into a bigram model.

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
        for p, s in self.as_rules():
            new_s = []
            for word in s:
                if self.is_nonterminal(word):
                    nt_counts[word] += 1
                    new_s.append(f"{word}_{nt_counts[word]}")
                else:
                    new_s.append(word)
            rules.append((p, new_s))

        # duplicate annotated rules
        duped_rules = []
        for p, s in rules:
            # handle nonterminals that never appear in a RHS by taking max of (1, n)
            n = max(1, nt_counts[p])
            for i in range(1, n+1):
                duped_rules.append((f"{p}_{i}", s))

        # add transitions to transformed start symbol
        n = max(1, nt_counts[self.start])
        for i in range(1, n+1):
            duped_rules.append((self.start, [f"{self.start}_{i}"]))

        return CFG.from_rules(self.start, duped_rules)

    def explode(self) -> 'CFG':
        """
        Explode the grammar exponentially.
        """
        # count the number of appearances of each nonterminal
        nt_counts = {nt: 0 for nt in self.nonterminals}
        for p, s in self.as_rules():
            for word in s:
                if self.is_nonterminal(word):
                    nt_counts[word] += 1

        # annotate RHS nonterminals with all possible indices
        annotated_rules = []
        for p, s in self.as_rules():
            factors = []
            for word in s:
                if self.is_nonterminal(word):
                    n = max(1, nt_counts[word])
                    factors.append([f"{word}_{i}"
                                    for i in range(1, n+1)])
                else:
                    factors.append([word])

            prod = list(it.product(*factors))
            for new_s in prod:
                annotated_rules.append((p, list(new_s)))

        # duplicate LHS of annotated rules
        duped_rules = []
        for p, s in annotated_rules:
            n = max(1, nt_counts[p])
            for i in range(1, n+1):
                duped_rules.append((f"{p}_{i}", s))

        # add transitions to transformed start symbol
        n = max(1, nt_counts[self.start])
        rules = duped_rules + [(self.start, [f"{self.start}_{i}"])
                               for i in range(1, n+1)]

        return CFG.from_rules(self.start, rules)

    def is_in_CNF(self) -> bool:
        """
        Checks whether the CFG is in Chomsky normal form.
        In CNF, all rules should be of the form:
         - A -> BC
         - A -> a
         - S -> empty
        """
        for p, xs in self.as_rules():
            if len(xs) > 2 or len(xs) < 1:
                return False
            elif len(xs) == 1:
                a = xs[0]
                if (self.is_nonterminal(a) or
                   (p != self.start and xs == CFG.Empty)):
                    return False
            elif len(xs) == 2:
                B, C = xs
                if (self.is_terminal(B) or self.is_terminal(C) or
                   B == self.start or C == self.start):
                    return False
        return True

    def to_CNF(self, debug=False) -> 'CFG':
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
            if debug:  # pragma: no cover
                print(f"Performing {name} on\n{g}\n"
                      f"yielded\n{g_new}")
            g = g_new
        return g

    def _start(self) -> 'CFG':
        """Eliminate the start symbol from any RHS"""
        return CFG.from_rules(
            "_start_",
            it.chain(self.as_rules(), [("_start_", [self.start])])
        )

    def _term(self) -> 'CFG':
        """Eliminate rules with non-solitary terminals"""
        def nt(c) -> str:
            return f"_term_{c}_"

        rules: List[Tuple[CFG.Word, CFG.Sentence]] = []
        for pred, succs in self.rules.items():
            for i, succ in enumerate(succs):
                if len(succ) == 1:
                    rules.append((pred, succ))
                else:
                    # replace all terminals with a nonterminal
                    new_succ = []
                    for word in succ:
                        if self.is_terminal(word):
                            nt_rule = (nt(word), [word])
                            if nt_rule not in rules:
                                rules.append(nt_rule)
                            new_succ.append(nt(word))
                        else:
                            new_succ.append(word)
                    rules.append((pred, new_succ))

        return CFG.from_rules(self.start, rules)

    def _bin(self) -> 'CFG':
        """
        Eliminate rules whose rhs has more than 2 nonterminals.
        Assumes that _term() has been run, so any rules containing terminals
        should only contain a single terminal.
        """
        def nt(p, i, j) -> str:
            return f"_bin_{p}_{i}_{j}_"

        rules: List[Tuple[CFG.Word, CFG.Sentence]] = []
        for pred, succs in self.rules.items():
            for i, succ in enumerate(succs):
                if len(succ) > 2:
                    rules.append((pred, [succ[0], nt(pred, i, 1)]))
                    j = 1
                    for c in succ[1:-2]:
                        rules.append((nt(pred, i, j),
                                      [c, nt(pred, i, j+1)]))
                        j += 1
                    rules.append((nt(pred, i, j), succ[-2:]))
                else:
                    rules.append((pred, succ))
        return CFG.from_rules(self.start, rules)

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
            if nt != self.start and CFG.Empty in prods
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
            cache[letter] = (not self.is_terminal(letter) and
                             any(all(is_nullable(c) for c in rule if c != letter)
                                 for rule in self.rules[letter]))
            return cache[letter]

        return {nt for nt in self.rules if is_nullable(nt)}

    def _del(self) -> 'CFG':
        """
        Eliminate rules of the form A -> eps, where A != S.
        """
        nullable_nts = self.nullables()
        rules = []

        for pred, succ in self.as_rules():
            # if a nonterminal in the successor is nullable,
            # then add a version of the rule that does not contain
            # the nullable successor
            succs = []
            nullable_indices = {i for i, c in enumerate(succ) if c in nullable_nts}
            for indices in util.language_plus(nullable_indices):
                s = util.remove_at_pos(succ, indices)
                if s and s not in self.rules[pred]:
                    succs.append(s)

            if succs:
                if succ != CFG.Empty:
                    rules.append((pred, succ))
                for s in succs:
                    rules.append((pred, s))
            elif succ != CFG.Empty:
                rules.append((pred, succ))

        condensed_rules = []
        for (pred, succ), grp in it.groupby(sorted(rules), key=lambda x: x[:2]):
            condensed_rules.append((pred, succ))

        return CFG.from_rules(self.start, condensed_rules)

    def _unit(self) -> 'CFG':
        """Eliminate rules of the form A -> B, where A and B are both nonterminals."""
        cache = {}

        def contract(nt) -> List:
            """
            Returns the sentences that `nt` maps to after contracting chains of the form
            A -> B -> ... -> Z into A -> Z.  Uses dynamic programming (via `cache`).
            """
            if nt in cache:
                return cache[nt]
            rules: List[CFG.Sentence] = []
            for succ in self.rules[nt]:
                # remove identity productions
                if succ == [nt]:
                    continue
                h = succ[0]
                if len(succ) > 1 or self.is_terminal(h):
                    # ignore rules of len >1
                    rules.append(succ)
                else:
                    # recurse
                    rules.extend(contract(h))
            cache[nt] = rules
            return cache[nt]

        def used(nt: CFG.Word, rules: List[Tuple[CFG.Word, CFG.Sentence]]) -> bool:
            """Determines whether `nt` is used in the productions of other nonterminals in `rules`."""
            return any(nt != pred and nt in succ
                       for pred, succ in rules)

        # run contract()
        rules = [(nt, succ)
                 for nt in self.rules
                 for succ in contract(nt)]

        # filter out unused nonterminals (disconnected graph components)
        filtered_rules = []
        for p, s in rules:
            if p == self.start or used(p, rules) \
               and (p, s) not in filtered_rules:
                filtered_rules.append((p, s))

        return CFG.from_rules(self.start, filtered_rules)


class PCFG(T.nn.Module):
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
    def __init__(self,
                 start: CFG.Word,
                 rules: Dict[str, Collection[str | CFG.Sentence]],
                 weights: str | Dict[CFG.Word, List[float] | T.Tensor] = "uniform",
                 log_mode=False):
        super(PCFG, self).__init__()
        self.cfg = CFG(start, rules)
        # use PCFG's successor function without subclassing CFG
        self.cfg.successor = lambda nt: self.successor(nt)
        self.log_mode = log_mode

        if weights == "uniform":
            self.weights = T.nn.ParameterDict({
                pred: ((lambda x: x.log() if log_mode else x)
                       (T.ones(len(succs), dtype=T.float64) / len(succs)))
                for pred, succs in rules.items()
            })
        else:
            self.weights = T.nn.ParameterDict({
                k: (T.tensor(v, dtype=T.float64) if not isinstance(v, T.Tensor)
                    else v.clone().double())
                for k, v in weights.items()
            })

    @staticmethod
    def from_CFG(cfg: CFG, weights: str | Dict[CFG.Word, List[float] | T.Tensor] = "uniform", **kvs) -> 'PCFG':
        return PCFG(cfg.start, cfg.rules, weights, **kvs)

    @property
    def start(self) -> CFG.Word:
        return self.cfg.start

    def rules(self) -> Iterable[Tuple[CFG.Word, List[CFG.Sentence]]]:
        return self.cfg.rules.items()

    def successors(self, nt: CFG.Word) -> Iterable[CFG.Sentence]:
        return self.cfg.rules[nt]

    def successor(self, letter: str) -> List[str]:
        return random.choices(population=self.cfg.rules[letter],
                              weights=self.weights[letter],
                              k=1)[0]

    def iterate_fully(self) -> CFG.Sentence:
        return self.cfg.iterate_fully()

    def iterate_until(self, length: int) -> CFG.Sentence:
        return self.cfg.iterate_until(length)

    def __len__(self) -> int:
        return sum(len(succs) for succs in self.cfg.rules.values())

    def __eq__(self, other):
        return self.approx_eq(other, threshold=10 ** -2)

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __str__(self) -> str:
        def denote_t_or_nt(xs):
            return [f"{x}" if self.is_nonterminal(x) else f"`{x}`"
                    for x in xs]

        rules = "\n  ".join(
            f"{nt} ->\n    " +
            "\n    ".join(f"{' '.join(denote_t_or_nt(succ))} @ {weight:.10f}"
                          for succ, weight in zip(self.cfg.rules[nt],
                                                  self.weights[nt]))
            for nt in self.cfg.nonterminals
        )
        return ("PCFG: {"
                f"\n  log_mode={self.log_mode}"
                f"\n  start={self.cfg.start}"
                f"\n  rules=\n  {rules}\n}}")

    def is_in_CNF(self) -> bool:
        return self.cfg.is_in_CNF()

    def is_nonterminal(self, sym: CFG.Word) -> bool:
        return self.cfg.is_nonterminal(sym)

    @property
    def nonterminals(self) -> Iterable[CFG.Word]:
        return self.cfg.nonterminals

    def approx_eq(self, other, threshold):
        """
        Checks whether two PCFGs are syntactically (not semantically) equivalent
        and have roughly the same parameters.
        """
        return isinstance(other, PCFG) and \
            self.cfg == other.cfg and \
            self.log_mode == other.log_mode and \
            all(util.vec_approx_eq(self.weights[nt], other.weights[nt], threshold)
                for nt in self.cfg.nonterminals)

    def normalized(self, c=0.1) -> 'PCFG':
        """
        Returns a PCFG with similar relative weights, but normalized
        with smoothing constant c.
        """
        return PCFG.from_CFG(
            self.cfg,
            weights={
                pred: (ws + c) / T.sum(ws + c)
                for pred, ws in self.weights.items()
            }
        )

    def is_normalized(self, tolerance=1e-3) -> bool:
        for weights in self.weights.values():
            if (not self.log_mode and abs(1 - sum(weights)) >= tolerance) or \
               (self.log_mode and abs(T.logsumexp(weights, dim=0).item()) >= tolerance):
                return False
        return True

    def weight(self, pred: CFG.Word, succ: CFG.Sentence) -> T.Tensor:
        if pred in self.cfg.nonterminals:
            try:
                i = self.cfg.rules[pred].index(succ)
                return self.weights[pred][i]
            except ValueError:
                pass
        return T.tensor(-T.inf) if self.log_mode else T.tensor(0)

    def apply_to_weights(self, f) -> 'PCFG':
        return PCFG.from_weighted_rules(
            self.cfg.start,
            [(p, s, f(w)) for p, s, w in self.as_weighted_rules()]
        )

    def log(self) -> 'PCFG':
        g = self.apply_to_weights(T.log)
        g.log_mode = True
        return g

    def exp(self) -> 'PCFG':
        g = self.apply_to_weights(T.exp)
        g.log_mode = False
        return g

    def copy(self) -> 'PCFG':
        g = self.apply_to_weights(lambda x: x)
        g.log_mode = self.log_mode
        return g

    @staticmethod
    def from_weighted_rules(start: CFG.Word,
                            weighted_rules: Iterable[Tuple[CFG.Word, CFG.Sentence, float]]) -> 'PCFG':
        """Construct a PCFG from an iterable of weighted rules"""
        rules = {}
        weights = {}
        for pred, succ, weight in sorted(weighted_rules, key=lambda x: x[0]):
            if pred not in rules:
                rules[pred] = [succ]
                weights[pred] = [weight]
            else:
                rules[pred].append(succ)
                weights[pred].append(weight)
        return PCFG(start, rules, weights)

    def as_weighted_rules(self) -> Iterable[Tuple[CFG.Word, CFG.Sentence, T.Tensor]]:
        """View a PCFG as a list of rules with weights"""
        for nt in self.cfg.nonterminals:
            for i, succ in enumerate(self.cfg.rules[nt]):
                yield nt, succ, self.weights[nt][i]


class Grammar:

    Symbol = str

    def __init__(self, rules: Dict[Symbol, List[Tuple[float | T.Tensor, Tuple | Symbol]]]):
        """
        Probabilistic Context Free Grammar (PCFG) over program expressions

        `rules`: mapping from non-terminal symbol to list of productions.
        each production is a tuple of (log probability, form), where
        log probability is that generating ﾻfrom the nonterminal will use that production
        and form is either :
          (1) a tuple of the form (constructor, non-terminal-1, non-terminal-2, ...).
             `constructor` should be a component in the DSL, such as '+' or '*', which takes arguments
          (2) a `constructor`, where `constructor` should be a component in the DSL,
          such as '0' or 'x', which takes no arguments

        Non-terminals can be anything that can be hashed and compared for equality,
        such as strings, integers, and tuples of strings/integers.
        """
        self.rules = rules

    def __str__(self) -> str:
        return self.pretty_print()

    def pretty_print(self) -> str:
        pretty = ""
        for symbol, productions in self.rules.items():
            for probability, form in productions:
                pretty += f"{symbol} ::= "
                if isinstance(form, tuple):
                    pretty += "constructor='" + form[0] + "', args=" + ",".join(map(str, form[1:]))
                else:
                    pretty += "constructor=" + form
                pretty += "\tw.p. " + str(probability) + "\n"
        return pretty

    @staticmethod
    def from_components(components: Dict[Symbol, List[Symbol]], gram) -> 'Grammar':
        """
        Builds and returns a `Grammar` (ie PCFG) from typed DSL components
        Initializes the probabilities to be the same for every single rule

        gram=1: unigram
        gram=2: bigram
        """
        assert gram in [1, 2]
        symbols = {typ for component_type in components.values() for typ in component_type}

        if gram == 1:
            def make_form(name, tp):
                assert len(tp) >= 1
                if len(tp) == 1: return name
                else: return tuple([name] + list(tp[:-1]))

            rules = {symbol: [(0., make_form(component_name, component_type))
                              for component_name, component_type in
                              components.items() if component_type[-1] == symbol]
                     for symbol in symbols}

        elif gram == 2:
            for parent, parent_type in components.items():
                if len(parent_type) == 1: continue  # this is not a function, so cannot be a parent
                for argument_index, argument_type in enumerate(parent_type[:-1]):
                    symbols.add((parent, argument_index, argument_type))

            rules = {}
            for symbol in symbols:
                rules[symbol] = []
                if isinstance(symbol, tuple):
                    return_type = symbol[-1]
                else:
                    return_type = symbol

                for component, component_type in components.items():
                    if component_type[-1] == return_type:
                        if len(component_type) == 1:
                            form = component
                        else:
                            form = tuple([component] +
                                         [(component, argument_index, argument_type)
                                          for argument_index, argument_type in enumerate(component_type[:-1])])
                        rules[symbol].append((0., form))

        return Grammar(rules)

    def normalize_(self) -> 'Grammar':
        """
        Destructively modifies grammar so that all the probabilities sum to one
        Has some extra logic so that if the log probabilities are coming from a neural network,
         everything is handled properly, but you don't have to worry about that
        """

        def norm(productions):
            z, _ = productions[0]
            if isinstance(z, T.Tensor):
                z = T.logsumexp(T.stack([log_probability for log_probability, _ in productions]), 0)
            else:
                for log_probability, _ in productions[1:]:
                    z = np.logaddexp(z, log_probability)

            return [(log_probability - z, production) for log_probability, production in productions]

        self.rules = {symbol: norm(productions)
                      for symbol, productions in self.rules.items()}

        return self

    def uniform_(self) -> 'Grammar':
        """
        Destructively modifies grammar so that all the probabilities are uniform across each nonterminal symbol
        """
        self.rules = {symbol: [(0., form) for _, form in productions]
                      for symbol, productions in self.rules.items()}
        return self.normalize_()

    @property
    def n_parameters(self) -> int:
        """
        Returns the number of real-valued parameters in the probabilistic grammar
        (technically, this is not equal to the number of degrees of freedom,
        because we have extra constraints that the probabilities must sum to one
        across each nonterminal symbol)
        """
        return sum(len(productions) for productions in self.rules.values())

    def from_tensor_(self, tensor):
        """
        Destructively modifies grammar so that the continuous parameters come from the provided pytorch tensor
        """
        assert tensor.shape[0] == self.n_parameters
        index = 0
        for symbol in sorted(self.rules.keys(), key=str):
            for i in range(len(self.rules[symbol])):
                _, form = self.rules[symbol][i]
                self.rules[symbol][i] = (tensor[index], form)
                index += 1
        assert self.n_parameters == index

    def sample(self, nonterminal):
        """
        Samples a random expression built from the space of syntax trees generated by `nonterminal`
        """
        # productions: all the ways that we can produce expressions from this nonterminal symbol
        productions = self.rules[nonterminal]

        # sample from multinomial distribution given by log probabilities in `productions`
        log_probabilities = [log_probability for log_probability, form in productions]
        probabilities = np.exp(np.array(log_probabilities))
        i = np.argmax(np.random.multinomial(1, probabilities))
        _, rule = productions[i]
        if isinstance(rule, tuple):
            # this rule is a function that takes arguments
            constructor, *arguments = rule
            return tuple([constructor] + [self.sample(argument) for argument in arguments])
        else:
            # this rule is just a terminal symbol
            constructor = rule
            return constructor

    def log_probability(self, nonterminal, expression) -> T.Tensor:
        """
        Returns the log probability of sampling `expression` starting from the symbol `nonterminal`
        """
        for log_probability, rule in self.rules[nonterminal]:
            if isinstance(expression, tuple) and isinstance(rule, tuple) and expression[0] == rule[0]:
                child_log_probability = sum(self.log_probability(child_symbol, child_expression)
                                            for child_symbol, child_expression in zip(rule[1:], expression[1:]))
                return log_probability + child_log_probability

            if expression == rule:
                return log_probability

        raise ValueError("could not calculate probability of expression giving grammar")

    def top_down_generator(self, start_symbol):
        """
        Best-first top-down enumeration of programs generated from the PCFG

        start_symbol: a nonterminal in the grammar. Should have: `start_symbol in self.rules.keys()`

        Yields a generator.
        Each generated return value is of the form: (log probability, expression)
        The expressions should be in non-increasing order of (log) probability
        Every expression that can be generated from the grammar should eventually be yielded
        """
        heap = MaxHeap()
        heap.push(0., start_symbol)

        def next_non_terminal(syntax_tree):
            for non_terminal in self.rules:
                if non_terminal == syntax_tree:
                    return non_terminal

            if not isinstance(syntax_tree, tuple):  # leaf
                return None

            arguments = syntax_tree[1:]
            for argument in arguments:
                argument_next = next_non_terminal(argument)
                if argument_next is not None:
                    return argument_next

            return None  # none of the arguments had a next non-terminal symbol to expand

        def finished(syntax_tree):
            return next_non_terminal(syntax_tree) is None

        def substitute_next_non_terminal(syntax_tree, expansion):
            for non_terminal in self.rules:
                if non_terminal == syntax_tree:
                    return expansion

            if not isinstance(syntax_tree, tuple):  # leaf
                return None  # failure

            function = syntax_tree[0]
            arguments = list(syntax_tree[1:])
            performed_substitution = False
            for argument_index, argument in enumerate(arguments):
                argument_new = substitute_next_non_terminal(argument, expansion)
                if argument_new is not None:
                    arguments[argument_index] = argument_new
                    performed_substitution = True
                    break

            if performed_substitution:
                return tuple([function] + arguments)
            else:
                return None

        while not heap.empty():
            log_probability, syntax_tree = heap.pop()

            if finished(syntax_tree):
                yield log_probability, syntax_tree
                continue

            non_terminal = next_non_terminal(syntax_tree)

            for production_log_probability, production in self.rules[non_terminal]:
                new_probability = production_log_probability + log_probability
                new_syntax_tree = substitute_next_non_terminal(syntax_tree, production)
                assert new_syntax_tree is not None, "should never happen"
                heap.push(new_probability, new_syntax_tree)

    def top_down_synthesize(self, start_symbol, input_outputs, evaluate: Callable):
        """
        Wrapper over top_down_generator that checks to see if generated programs satisfy input outputs

        start_symbol: a nonterminal in the grammar. Should have: `start_symbol in self.rules.keys()`
        input_outputs: list of pairs of input-outputs

        returns: (number_of_programs_enumerated, first_program_that_worked_on_input_outputs)
        """
        for j, (expression_log_probability, expression) in \
                enumerate(self.top_down_generator(start_symbol)):
            if all(o == evaluate(expression, i) for i, o in input_outputs):
                return 1 + j, expression


class LearnedGrammar:

    def __init__(self, feature_extractor: FeatureExtractor, template_grammar: Grammar):
        self.feature_extractor = feature_extractor
        # make a deep copy so that we can mutate it without causing problems
        self.grammar = copy.deepcopy(template_grammar)

        # keep around the original for comparison
        self.original_grammar = copy.deepcopy(template_grammar.normalize_())

        # get the number of output features
        n_features = feature_extractor.n_features

        # a simple linear model, but you could use a more complex neural network
        self.f_theta = T.nn.Linear(n_features, template_grammar.n_parameters).float()

    def train(self, start_symbol, training_examples, input_domain, evaluate: Callable, steps=10000):
        """
        Train learned model, which adjusts $theta$ to maximize the log probability of training examples
        given training specs derived from those examples

        start_symbol: which symbol in the grammar is used to start generating a program
        training_examples: a list of programs
        input_domain: a list of environments that the programs will be evaluated on to generate outputs
        steps: number of steps of gradient descent to perform
        """
        optimizer = T.optim.Adam(self.f_theta.parameters())
        recent_losses, recent_log_likelihoods = [], []

        for step in range(steps):
            program = random.choice(training_examples)
            input_outputs = [(i, evaluate(program, i)) for i in input_domain]

            features = self.feature_extractor.extract(input_outputs)
            features = T.tensor(features).float()
            projected_features = self.f_theta(features)

            self.grammar.from_tensor_(projected_features)
            self.grammar.normalize_()
            loss = -self.grammar.log_probability(start_symbol, program)

            recent_losses.append(loss.detach().numpy())
            recent_log_likelihoods.append(-self.original_grammar.log_probability(start_symbol, program))
            if len(recent_losses) == 100:
                print("training step", step + 1, "\n\tavg loss (negative log likelihood of expression, w/ learning)",
                      sum(recent_losses) / len(recent_losses),
                      "\n\t      avg negative log likelihood of expression, w/o learning",
                      sum(recent_log_likelihoods) / len(recent_log_likelihoods))
                recent_losses, recent_log_likelihoods = [], []

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def get_grammar(self, input_outputs):
        features = self.feature_extractor.extract(input_outputs)
        self.grammar.from_tensor_(self.f_theta(T.tensor(features).float()))
        self.grammar.normalize_()
        for symbol in self.grammar.rules.keys():
            for i in range(len(self.grammar.rules[symbol])):
                p, form = self.grammar.rules[symbol][i]
                self.grammar.rules[symbol][i] = (p.detach().cpu().numpy(), form)
        return copy.deepcopy(self.grammar)


class FeatureExtractor:

    @property
    def n_features(self) -> int:
        raise NotImplementedError

    def extract(self, spec: List[Tuple]) -> np.ndarray:
        raise NotImplementedError


class DummyFeatureExtractor(FeatureExtractor):

    def __init__(self):
        pass

    @property
    def n_features(self) -> int:
        return 1

    def extract(self, spec: List[Tuple]) -> np.ndarray:
        return np.array([1.])


class FixedLengthFeatureExtractor(FeatureExtractor):

    def __init__(self, n_exprs, max_expr_tokens: int, lexicon: List[str]):
        self.max_expr_tokens = max_expr_tokens    # length of feature vector output
        self.n_exprs = n_exprs
        self.lexicon = lexicon  # tokens to expect
        self.token_to_index = {s: i for i, s in enumerate(self.lexicon)}

    @property
    def n_features(self) -> int:
        return self.max_expr_tokens * self.n_exprs

    def extract(self, spec: List[Tuple]) -> np.ndarray:
        # take outputs and map to fixed-length vectors of indices
        raise NotImplementedError

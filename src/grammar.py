from __future__ import annotations
import torch as T
import lightning as pl
import numpy as np
from typing import *
import copy
from sentence_transformers import SentenceTransformer

from max_heap import MaxHeap
import util


class Grammar:

    Symbol = Union[str, tuple]

    def __init__(self, rules: Dict[Symbol, List[Tuple[float | T.Tensor, tuple | Symbol]]],
                 components: Dict[Symbol, List[Symbol]], gram: int):
        """
        Probabilistic Context Free Grammar (PCFG) over program expressions

        `rules`: mapping from non-terminal symbol to list of productions.
        each production is a tuple of (log probability, form), where
        log probability is that generating from the nonterminal will use that production
        and form is either :
          (1) a tuple of the form (constructor, non-terminal-1, non-terminal-2, ...).
             `constructor` should be a component in the DSL, such as '+' or '*', which takes arguments
          (2) a `constructor`, where `constructor` should be a component in the DSL,
          such as '0' or 'x', which takes no arguments

        Non-terminals can be anything that can be hashed and compared for equality,
        such as strings, integers, and tuples of strings/integers.
        """
        self.rules = rules
        self.components = components
        self.gram = gram

    def __str__(self) -> str:
        return self.pretty_print()

    def __eq__(self, other) -> bool:
        return self.rules == other.rules

    def pretty_print(self) -> str:
        pretty = ""
        for nt, prods in sorted(self.rules.items(), key=lambda x: str(x[0])):
            for w, form in prods:
                pretty += f"{nt} ::= "
                if isinstance(form, tuple):
                    pretty += f"constructor='{form[0]}', args={','.join(map(str, form[1:]))}"
                else:
                    pretty += f"constructor={form}"
                pretty += "\tw.p. " + str(w) + "\n"
        return pretty

    def weight(self, nt: Symbol, form: Symbol) -> float:
        for w, prod in self.rules[nt]:
            if form == prod:
                return w
        return -np.inf

    @staticmethod
    def from_components(components: Dict[Symbol, List[Symbol]], gram) -> 'Grammar':
        """
        Builds and returns a `Grammar` (ie PCFG) from typed DSL components
        Initializes the probabilities to be the same for every single rule

        gram=1: unigram
        gram=2: bigram
        """
        # "symbols" is the set of nonterminals in the grammar (also construed as types)
        symbols = {t
                   for comp_t in components.values()
                   for t in comp_t}
        if gram == 1:
            def make_form(name, t):
                assert len(t) >= 1
                if len(t) == 1: return name
                else: return tuple([name] + list(t[:-1]))
            rules = {sym: [(0., make_form(comp, comp_t))
                           for comp, comp_t in components.items()
                           if comp_t[-1] == sym]
                     for sym in symbols}
            return Grammar(rules, components, gram)
        elif gram == 2:
            # Add symbols of the form (sym1, 0, sym2) given that sym1's first arg can be sym2.
            # We won't really use the old symbols, but they'll come in handy when we want to grow expressions
            # without knowing where the start nt came from.
            symbols |= {
                (parent, arg_i, arg_t)
                for parent, parent_t in components.items()
                if len(parent_t) > 1
                for arg_i, arg_t in enumerate(parent_t[:-1])
            }
            rules = {}
            for sym in symbols:
                rules[sym] = []
                sym_t = sym[-1] if isinstance(sym, tuple) else sym
                for comp, comp_t in components.items():
                    if comp_t[-1] == sym_t:
                        if len(comp_t) == 1:
                            form = comp
                        else:
                            form = tuple([comp] + [(comp, arg_i, arg_t)
                                                   for arg_i, arg_t in enumerate(comp_t[:-1])])
                        rules[sym].append((0., form))
            return Grammar(rules, components, gram)
        else:
            raise ValueError(f"Expected gram in {{1, 2}} but got {gram}")

    def from_unigram_counts_(self, counts: Dict[str, int], alpha=0.):
        assert self.gram == 1
        for nt, prods in self.rules.items():
            for i, (_, form) in enumerate(prods):
                name = form[0] if isinstance(form, tuple) else form
                self.rules[nt][i] = np.log(counts.get(name, 0) + alpha), form
        self.normalize_(0.)

    def from_bigram_counts_(self, counts: Dict[Tuple[str, int, str] | str, int], alpha=0.):
        """
        Reweight the grammar using counts of how many times each rule was observed.
        `alpha` is the additive smoothing parameter.

        Counts should have keys of the form (x, i, y), where
        - x is the parent node
        - i is the index of the child node in the parent's argument list
        - y is the child node

        NOTE: because of the way sampling is performed from a single nonterminal
        (usually a unigram symbol), bigram grammars also need to have unigram counts
        """
        assert self.gram == 2
        for nt, prods in self.rules.items():
            if isinstance(nt, tuple):
                # use bigram counts
                for i, (_, form) in enumerate(prods):
                    parent, j, *_ = nt
                    child = form[0] if isinstance(form, tuple) else form
                    k = counts.get((parent, j, child), 0)
                    self.rules[nt][i] = np.log(k + alpha), form  # k can be 0 -> issues
            else:
                # use unigram counts
                for i, (_, form) in enumerate(prods):
                    name = form[0] if isinstance(form, tuple) else form
                    self.rules[nt][i] = np.log(counts.get(name, 0) + alpha), form

        self.normalize_(0.)

    def normalize_(self, alpha=0.) -> 'Grammar':
        """
        Destructively modifies grammar so that all the probabilities sum to one.
        `alpha` is the additive smoothing parameter.

        Has some extra logic so that everything is handled properly if the
         log probabilities come from a neural network.
         """
        assert isinstance(alpha, float) or isinstance(alpha, int), \
            f"Expected alpha to be a float or int but got {type(alpha)}"

        def add_log(a: T.Tensor | float, b: float) -> T.Tensor | float:
            if alpha > 0:
                if isinstance(a, T.Tensor):
                    return T.logaddexp(a, T.tensor(b).log())
                else:
                    return np.logaddexp(a, np.log(b))
            else:
                return a

        def norm(symbol, prods):
            if len(prods) == 0:
                raise ValueError(f"Empty production list for symbol {symbol}")
            z, _ = prods[0]
            if isinstance(z, T.Tensor):
                z = T.logsumexp(T.stack([w for w, _ in prods]), 0)
            else:
                for w, _ in prods[1:]:
                    z = np.logaddexp(z, w)
            return [(add_log(w, alpha) - add_log(z, alpha * len(prods)), form)
                    for w, form in prods]

        self.rules = {symbol: norm(symbol, productions)
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
        i = 0
        for symbol, prods in sorted(self.rules.items(), key=lambda x: str(x[0])):
            for j, (_, form) in enumerate(prods):
                self.rules[symbol][j] = (tensor[i], form)
                i += 1
        assert self.n_parameters == i

    def from_tensor(self, tensor: T.Tensor) -> "Grammar":
        """
        Generates a new grammar whose continuous parameters come from the provided pytorch tensor.
        Non-destructive, unlike from_tensor_.
        """
        assert tensor.shape[0] == self.n_parameters, \
            f"Grammar has {self.n_parameters} params but received a tensor with shape {tensor.shape}"
        rules = copy.deepcopy(self.rules)
        i = 0
        for sym, prods in sorted(rules.items(), key=lambda x: str(x[0])):
            for j, (_, form) in enumerate(prods):
                rules[sym][j] = (tensor[i], form)
                i += 1
        g = Grammar(rules, self.components, self.gram)
        assert g.n_parameters == i
        return g

    def to_tensor(self) -> T.Tensor:
        """Returns a tensor containing the continuous parameters of the grammar"""
        t = T.empty(self.n_parameters)
        i = 0
        for sym in sorted(self.rules.keys(), key=str):
            for w, _ in self.rules[sym]:
                t[i] = w
                i += 1
        return t

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
        ******** memoize this
        """
        for log_probability, rule in self.rules[nonterminal]:
            if isinstance(expression, tuple) and isinstance(rule, tuple) and expression[0] == rule[0]:
                child_log_probability = sum(self.log_probability(child_symbol, child_expression)
                                            for child_symbol, child_expression in zip(rule[1:], expression[1:]))
                return log_probability + child_log_probability

            if expression == rule:
                return log_probability

        raise ValueError("Could not calculate probability of expression giving grammar for "
                         f"nonterminal={nonterminal}, expression={expression}")

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


class LearnedGrammar(pl.LightningModule):

    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 grammar: Grammar,
                 parse: Callable[[str], Tuple],
                 start_symbol: str | Tuple,
                 learning_rate: float):
        """
        feature_extractor: maps each program output to a numerical vector
        grammar: the template grammar whose parameters should be tuned
        evaluator: runs a program tree, returning the program's output
        start_symbol: the grammar's start symbol
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.grammar = copy.deepcopy(grammar)
        self.parse = parse
        self.start_symbol = start_symbol
        self.learning_rate = learning_rate
        self.original_grammar = copy.deepcopy(grammar.normalize_())
        self.f_theta = T.nn.Linear(feature_extractor.n_features, grammar.n_parameters).float()

    def configure_optimizers(self) -> Any:
        return T.optim.Adam(self.f_theta.parameters(), lr=self.learning_rate)

    def forward(self, x, y) -> Grammar:
        features = self.feature_extractor.extract([x], [y]).float()
        projected_features = self.f_theta(features)
        self.grammar.from_tensor_(projected_features.detach())
        self.grammar.normalize_()
        return self.grammar

    def training_step(self, batch, batch_idx):
        """
        Train learned model, which adjusts $theta$ to maximize the log probability of training examples
        given training specs derived from those examples
        """
        # extract features, then project onto grammar
        (x,), y = batch  # x: List[str], y: List[output]
        features = self.feature_extractor.extract(x, y).float()
        projected_features = self.f_theta(features)
        self.grammar.from_tensor_(projected_features)
        self.grammar.normalize_()

        # compute loss
        loss = -self.grammar.log_probability(self.start_symbol, self.parse(x))

        # logging
        self.log("train_loss", loss)
        self.log("train_loss_per_token", loss / len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        (x,), y = batch
        loss = -self.grammar.log_probability(self.start_symbol, self.parse(x))
        self.log("val_loss", loss)
        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        # log grammar with checkpoint
        checkpoint["grammar_params"] = self.grammar.to_tensor()

    def get_grammar(self, x: List[Tuple], y: List[Any]):
        features = self.feature_extractor.extract(x, y)
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

    def extract(self, x: List, y: List[Any]) -> T.Tensor:
        """Maps a list of programs and their outputs to a list of real-valued feature vectors"""
        raise NotImplementedError


class DummyFeatureExtractor(FeatureExtractor):

    def __init__(self):
        pass

    @property
    def n_features(self) -> int:
        return 1

    def extract(self, x: List, y: List[Any]) -> T.Tensor:
        return T.ones(len(x))


class SBertFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    @property
    def n_features(self) -> int:
        return 384

    def extract(self, x: List, y: List[Any]) -> T.Tensor:
        return self.model.encode(y, convert_to_tensor=True)[0]


class ConvFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 n_features: int,
                 n_color_channels: int,
                 n_conv_channels: int,
                 bitmap_n_rows: int,
                 bitmap_n_cols: int,
                 batch_size: int = 1):
        self._n_features = n_features
        self.n_conv_channels = n_conv_channels
        self.n_color_channels = n_color_channels
        self.bitmap_n_rows = bitmap_n_rows
        self.bitmap_n_cols = bitmap_n_cols
        self.batch_size = batch_size

        def conv_block(in_channels: int, out_channels: int, kernel_size=3) -> T.nn.Module:
            return T.nn.Sequential(
                T.nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                T.nn.BatchNorm2d(out_channels),
                T.nn.ReLU(),
            )

        self.conv = T.nn.Sequential(
            conv_block(n_color_channels, n_conv_channels),
            conv_block(n_conv_channels, n_conv_channels),
            conv_block(n_conv_channels, n_conv_channels),
            conv_block(n_conv_channels, n_conv_channels),
            T.nn.Flatten(),
            T.nn.Linear(n_conv_channels * bitmap_n_rows * bitmap_n_cols, n_features),
        )

    @property
    def n_features(self) -> int:
        return self._n_features

    def extract(self, x: List, y: List[Any]) -> T.Tensor:
        outputs = T.stack([img.float() for img in y])
        outputs = self.conv(outputs)
        return outputs.squeeze().detach()

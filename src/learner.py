from __future__ import annotations
import torch as T
import numpy as np
import random
from typing import *
import copy

from max_heap import MaxHeap


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
        ******** memoize this
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

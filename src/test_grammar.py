from copy import deepcopy
from grammar import *

simple_components = {
    "add": ["expr", "expr", "expr"],
    "abs": ["expr", "expr"],
}

list_components = {
    "sort": ["list", "list"],
    "reverse": ["list", "list"],
    "tail": ["list", "list"],
    "map+1": ["list", "list"],
    "map-1": ["list", "list"],
    "filter<0": ["list", "list"],
    "filter>0": ["list", "list"],
    "filter_even": ["list", "list"],
    "filter_odd": ["list", "list"],
    "l": ["list"]
}


def test_to_from_tensor():
    grammars = [
        Grammar.from_components(simple_components, gram=1),
        Grammar.from_components(simple_components, gram=2),
        Grammar.from_components(list_components, gram=1),
        Grammar.from_components(list_components, gram=2),
    ]
    for g in grammars:
        # g -> [t] -> h
        t = g.to_tensor()
        h = g.from_tensor(t)
        assert h == g, f"g={g}, h={h}"

        # [t] -> h -> [t']
        tp = h.to_tensor()
        assert T.equal(t, tp)

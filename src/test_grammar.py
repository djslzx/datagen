from grammar import *

simple_components = {
    "add": ["expr", "expr", "expr"],
    "neg": ["expr", "expr"],
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
    # cases = [
    #     T.tensor([])
    # ]
    g = Grammar.from_components(simple_components, gram=1)
    print(g.as_tensor())


if __name__ == "__main__":
    test_to_from_tensor()
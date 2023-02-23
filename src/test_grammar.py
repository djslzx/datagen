from grammar import *
import parse

add_components = {
    "add": ["Int", "Int", "Int"],
    "1": ["Int"],
}

int_components = {
    "add": ["Int", "Int", "Int"],
    "abs": ["Int", "Int"],
    "0": ["Int"],
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
        Grammar.from_components(int_components, gram=1),
        Grammar.from_components(int_components, gram=2),
        Grammar.from_components(list_components, gram=1),
        Grammar.from_components(list_components, gram=2),
    ]
    for g in grammars:
        # g -> [t] -> h
        g.normalize_()
        t = g.to_tensor()
        h = g.from_tensor(t)
        assert h == g, f"g={g}, h={h}"

        # [t] -> h -> [t']
        tp = h.to_tensor()
        assert T.equal(t, tp)


def test_normalize():
    pass


def test_from_bigram_counts_add():
    g = Grammar.from_components(add_components, gram=2)
    s = ('add', ('add', '1', '1'), '1')
    counts = parse.count_bigram(s)
    print()
    print(counts)
    print(g)
    g.from_bigram_counts_(counts, alpha=1)
    print(g)


def test_from_bigram_counts():
    corpus = [
        "F;F~F",
    ]
    parsed_corpus = [
        parse.parse_lsys(x) for x in corpus
    ]
    counts = parse.multi_count_bigram(parsed_corpus)
    g = Grammar.from_components(parse.rule_types, gram=2)
    g.from_bigram_counts_(counts, alpha=0)
    print(g)


if __name__ == "__main__":
    pass
    # for sym, prods in g.rules.items():
    #     for w, prod in prods:
    #         print(sym, prod)
    # test_from_bigram_counts()

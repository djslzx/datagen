from grammar import *
import parse

nat_components = {
    "add": ["Int", "Int", "Int"],
    "1": ["Int"],
}

int_components = {
    "add": ["Int", "Int", "Int"],
    "neg": ["Int", "Int"],
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
        # w -> [g] -> w', check w == w'
        t = g.to_tensor()
        w = T.rand(t.shape)
        gp = g.from_tensor(w)
        wp = gp.to_tensor()
        assert T.equal(w, wp), f"w -> [g] -> w', w={w}, w'={wp}"

        #  g' -> w' -> g'', check g' == g''
        gpp = gp.from_tensor(wp)
        assert gpp == gp


def test_normalize_nat_unigram():
    g = Grammar.from_components(nat_components, gram=1)
    # 2 components in tensor: weights for add, 1
    g = g.from_tensor(T.tensor([1, 0]).log())
    g.normalize_(alpha=0)
    normed_weights = g.to_tensor()
    ans = T.tensor([1, 0]).log()
    assert util.vec_approx_eq(normed_weights, ans), \
        f"Expected {ans} but got {normed_weights}"

    g = g.from_tensor(T.tensor([1, 0]).log())
    g.normalize_(alpha=1)
    normed_weights = g.to_tensor()
    ans = T.tensor([2/3, 1/3]).log()
    assert util.vec_approx_eq(normed_weights, ans), \
        f"Expected {ans} but got {normed_weights}"


def test_normalize_nat_bigram():
    g = Grammar.from_components(nat_components, gram=2)
    g = g.from_tensor(T.tensor([0, 1, 2, 3, 4, 5]).log())
    g.normalize_(alpha=0)
    normed_weights = g.to_tensor()
    ans = T.tensor([0, 1, 2/5, 3/5, 4/9, 5/9]).log()
    assert util.vec_approx_eq(normed_weights, ans), \
        f"Expected {ans.exp()} but got {normed_weights.exp()}"

    g = g.from_tensor(T.tensor([0, 1, 2, 3, 4, 5]).log())
    g.normalize_(alpha=3)
    normed_weights = g.to_tensor()
    ans = T.tensor([3/7, 4/7, 5/11, 6/11, 7/15, 8/15]).log()
    assert util.vec_approx_eq(normed_weights, ans), \
        f"Expected {ans.exp()} but got {normed_weights.exp()}"


def test_from_bigram_counts_add():
    g = Grammar.from_components(nat_components, gram=2)
    """
    (add, 0, Int) -> add, (add, 0, Int) -> one,
    (add, 1, Int) -> add, (add, 1, Int) -> one,
    int -> add, int -> one
    """
    s = ('add', ('add', '1', '1'), '1')
    counts = parse.count_bigram(s)
    assert counts == {('add', 0, 'add'): 1,
                      ('add', 0, '1'): 1,
                      ('add', 1, '1'): 2}

    g.from_bigram_counts_(counts, alpha=1)
    t = g.to_tensor()
    ans = T.tensor([1/2, 1/2, 1/4, 3/4, 1/2, 1/2]).log()
    assert T.equal(t, ans), f"Expected {ans} but got {t}"


def test_from_bigram_counts_lsys():
    # doesn't actually test anything aside from checking that the pieces fit together
    corpus = [
        "F;F~F",
    ]
    parsed_corpus = [parse.parse_lsys(x) for x in corpus]
    counts = parse.multi_count_bigram(parsed_corpus)
    g = Grammar.from_components(parse.rule_types, gram=2)
    g.from_bigram_counts_(counts, alpha=0)
    print(g)

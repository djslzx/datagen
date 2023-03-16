use egg::{*, rewrite as rw};

define_language! {
    enum Regex {
        // operators
        "maybe" = Maybe(Id),
        "star" = Star(Id),
        "plus" = Plus(Id),
        "bracket" = Bracket(Id),
        "or" = Or([Id; 2]),
        "seq" = Seq([Id; 2]),

        // character classes
        "dot" = Dot,
        "alpha" = Alpha,
        "digit" = Digit,
        "upper" = Upper,
        "lower" = Lower,
        "whitespace" = Whitespace,
        "literal" = Literal,
    }
}

fn make_rules() -> Vec<Rewrite<Regex, ()>> {
    vec![
        // x?? => x?
        // (c) => c if c is a character class
        // x*? => x*
        //

        // unnecessary brackets: X~[Y] => X~Y
        rewrite!("unnecessary-brackets";
            "(arrow ?x (symbol (bracket ?y)))" =>
            "(arrow ?x ?y)"
        ),
    ]
}
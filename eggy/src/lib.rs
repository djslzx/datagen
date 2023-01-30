use egg::*;
use pyo3::prelude::*;

define_language! {
    enum LSystem {
        "lsystem" = LSystem([Id; 2]),
        "axiom" = Axiom([Id; 1]),
        "symbol" = Symbol([Id; 1]),
        "bracket" = Bracket([Id; 1]),
        "symbols" = Symbols([Id; 2]),
        "nonterm" = Nonterm([Id; 1]),
        "term" = Term([Id; 1]),
        "rule" = Rule([Id; 1]),
        "rules" = Rules([Id; 2]),
        "arrow" = Arrow([Id; 2]),
        "F" = Draw,
        "f" = Jump,
        "+" = Add,
        "-" = Sub,
    }
}

fn make_rules() -> Vec<Rewrite<LSystem, ()>> {
    vec![
        rewrite!("tree-commute";
            "(symbols ?x ?y)" => "(symbols ?y ?x)"),
        rewrite!("tree-assoc";
            "(symbols (symbols ?a ?b) ?c)" => "(symbols ?a (symbols ?b ?c)))"),
        // +- => ''
        rewrite!("zero-turn";
            "(symbols (symbols (symbol (term +)) (symbol (term -))) ?x)" => "?x"),
        // TODO: handle empty symbols list from zero turn reductions?
        rewrite!("bracket-dup";
            "(symbols (bracket ?x) ?x)" => "?x"),
        rewrite!("rules-commute-one";
            "(rules ?a (rule ?b))" => "(rules ?b (rule ?a))"),
        rewrite!("rules-commute-mult";
            "(rules ?a (rules ?b ?c))" => "(rules ?b (rules ?a ?c))"),
        rewrite!("rules-dup-mult";
            "(rules ?a (rules ?a ?b))" => "?b"),
    ]
}

/// parse an expression, simplify it using egraph, and pretty print it back out
#[pyfunction]
fn simplify(s: &str) -> PyResult<String> {
    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<LSystem> = s.parse().unwrap();

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let runner = Runner::default().with_expr(&expr).run(&make_rules());

    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(root);
    Ok(best.to_string())
}

#[pymodule]
fn eggy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify() {
        let xs = vec![
            "(lsystem (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))",
            "(lsystem (axiom (symbol (nonterm F))) (rule (arrow F (symbols (symbol (term +)) (symbols (symbol (term -)) (symbols (symbol (term +)) (symbols (symbol (term -)) (symbols (symbol (term -)) (symbols (symbol (term +)) (symbols (symbol (term +)) (symbols (symbol (term +)) (symbols (symbol (term -)) (symbols (symbol (term -)) (symbol (nonterm F)))))))))))))))",
            "(lsystem (axiom (symbol (nonterm F))) (rules (arrow F (symbol (nonterm F))) (rules (arrow F (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))))",
            "(lsystem (axiom (symbol (nonterm F))) (rules (arrow F (symbol (nonterm F))) (rules (arrow F (symbols (symbol (term +)) (symbols (symbol (term -)) (symbol (nonterm F))))) (rule (arrow F (symbol (nonterm F)))))))",
            "(lsystem (axiom (symbol (nonterm F))) (rules (arrow F (symbol (nonterm F))) (rules (arrow F (symbols (symbol (nonterm F)) (symbol (nonterm F)))) (rules (arrow F (symbol (nonterm F))) (rule (arrow F (symbols (symbol (nonterm F)) (symbol (nonterm F)))))))))",
            "(lsystem (axiom (symbol (nonterm F))) (rules (arrow F (symbols (symbol (nonterm F)) (symbols (bracket (symbols (symbol (term +)) (symbol (nonterm F)))) (symbol (nonterm F))))) (rules (arrow F (symbol (nonterm F))) (rule (arrow F (symbols (symbol (nonterm F)) (symbols (bracket (symbols (symbol (term +)) (symbol (nonterm F)))) (symbol (nonterm F)))))))))",
        ];
        for x in xs {
            println!("'{}',", simplify(x).unwrap());
        }
        let input = "(lsystem (axiom (symbol (nonterm F))) (rule (arrow F (symbols (symbol (term +)) (symbols (symbol (term -)) (symbols (symbol (term +)) (symbols (symbol (term -)) (symbols (symbol (term -)) (symbols (symbol (term +)) (symbols (symbol (term +)) (symbols (symbol (term +)) (symbols (symbol (term -)) (symbols (symbol (term -)) (symbol (nonterm F)))))))))))))))";
        let output = "(lsystem (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))";
        let result = simplify(input).unwrap();
        assert_eq!(result, output);
    }
}

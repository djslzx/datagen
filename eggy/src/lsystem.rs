use egg::{*, rewrite as rw};

define_language! {
    enum LSystem {
        "lsystem" = LSystem([Id; 3]),
        "angle" = Angle(Id),
        "axiom" = Axiom(Id),
        "symbols" = Symbols([Id; 2]),
        "symbol" = Symbol(Id),
        "bracket" = Bracket(Id),
        "nonterm" = Nonterm(Id),
        "term" = Term(Id),
        "rules" = Rules([Id; 2]),
        "rule" = Rule(Id),
        "arrow" = Arrow([Id; 2]),
        "F" = Draw,
        "f" = Hop,
        "L" = L,
        "R" = R,
        "+" = Add,
        "-" = Sub,
        "nil" = Nil,
        Num(i32),
    }
}

type EGraph = egg::EGraph<LSystem, TurnFold>;

// Annotate e-graph with whether an e-class has at least one draw or hop.
// We need this to do simplifications like [(-|+)*] => empty
#[derive(Default)]
pub struct TurnFold;
impl Analysis<LSystem> for TurnFold {
    type Data = bool;  // defines the domain D: true if expression has Draw/Hop

    fn make(egraph: &EGraph, enode: &LSystem) -> Self::Data {
        match enode {
            LSystem::Draw | LSystem::Hop => true,
            LSystem::Add | LSystem::Sub | LSystem::Nil => false,
            _ => enode.any(|c| egraph[c].data)
        }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_max(to, from)
    }
}

fn var(s: &str) -> Var {
    s.parse().unwrap()
}

fn has_only_turns(v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    // Check that the substitution's data is false (i.e. no Draw/Hop)
    move |egraph, _, subst| !egraph[subst[v]].data
}

/// parse an expression, simplify it using egraph, and pretty print it back out
pub fn simplify(s: &str) -> String {
    let rules = &[
        // empty turn
        rw!("empty-turn-plus-minus";
            "(symbols (term +) (symbols (term -) ?x))" => "?x"),
        rw!("empty-turn-minus-plus";
            "(symbols (term -) (symbols (term +) ?x))" => "?x"),
        rw!("empty-turn-plus-minus-end";
            "(symbols (term +) (symbol (term -)))" => "nil"),
        rw!("empty-turn-minus-plus-end";
            "(symbols (term -) (symbol (term +)))" => "nil"),

        // nil handling
        rw!("collapse-nil-symbols-lhs";
            "(symbols nil ?x)" => "?x"),
        rw!("collapse-nil-symbols-rhs";
            "(symbols ?x nil)" => "(symbol ?x)"),
        rw!("collapse-nil-symbol";
            "(symbol nil)" => "nil"),
        rw!("collapse-nil-axiom";
            "(lsystem ?x nil ?y)" => "nil"),
        rw!("collapse-nil-rhs";
            "(arrow ?x nil)" => "nil"),
        rw!("collapse-nil-rule";
            "(rule nil)" => "nil"),
        rw!("collapse-nil-rules-lhs";
            "(rules nil ?x)" => "?x"),
        rw!("collapse-nil-rules-rhs";
            "(rules ?x nil)" => "(rule ?x)"),
        rw!("collapse-nil-all-rules";
            "(lsystem ?x ?y nil)" => "nil"),

        // retracing: X[X] => X
        rw!("retracing";
            "(symbols (bracket ?x) ?x)" => "?x"),

        // nested brackets: [[E]] -> [E]
        rw!("nested-brackets";
            "(bracket (symbol (bracket ?x)))" => "(bracket ?x)"),
        // nested brackets: [E[E]] -> [EE]: TODO (need annotations?)

        // only turns
        rw!("brackets-with-turns-only";
            "(bracket ?x)" => "nil"
            if has_only_turns(var("?x"))),
        rw!("axiom-with-turns-only";
            "(axiom ?x)" => "nil"
            if has_only_turns(var("?x"))),
        // disallow only having turns in rule rhs
        rw!("rhs-with-turns-only";
            "(arrow ?x ?y)" => "nil"
            if has_only_turns(var("?y"))
        ),
    ];

    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<LSystem> = s.parse().unwrap();

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let runner = Runner::<LSystem, TurnFold, ()>::default()
        .with_expr(&expr).run(rules);

    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(root);
    best.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle() {
        // 90;F;F~F
        let input = "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))";
        let ans = "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))";
        let output = simplify(input);
        assert_eq!(output, ans)
    }

    #[test]
    fn test_empty_turn() {
        let inputs = vec![
            // F;F~F
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))",
            // F;F~+-+--+++--F
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbols (term +) \
            (symbols (term -) (symbols (term +) (symbols (term -) (symbols (term -) \
            (symbols (term +) (symbols (term +) (symbols (term +) (symbols (term -) \
            (symbols (term -) (symbol (nonterm F)))))))))))))))",
            // "F;F~-+F+-",
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbols (term -) \
            (symbols (term +) (symbols (nonterm F) (symbols (term +) (symbol (term -)))))))))",
        ];
        let ans = "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))";
        for input in inputs {
            let output = simplify(input);
            assert_eq!(output, ans)
        }
    }

    #[test]
    fn test_extra_brackets() {
        // keep rhs brackets: X~[Y] => X~Y
        assert_eq!( // F;F~[F]
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (bracket (symbol (nonterm F)))))))"),
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (bracket (symbol (nonterm F)))))))"
        );
        assert_eq!( // F;F~[FF+FF]
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (bracket (symbols (nonterm F) (symbols (nonterm F) (symbols (term +) (symbols (nonterm F) (symbol (nonterm F)))))))))))"),
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (bracket (symbols (nonterm F) (symbols (nonterm F) (symbols (term +) (symbols (nonterm F) (symbol (nonterm F)))))))))))"
        );

        // rm nested brackets: [[X]] -> [X]
        assert_eq!( // [[F]];F~F -> [F];F~F
            simplify("(lsystem (angle 90) (axiom (symbol (bracket (symbol (bracket (symbol (nonterm F))))))) (rule (arrow F (symbol (nonterm F)))))"),
            "(lsystem (angle 90) (axiom (symbol (bracket (symbol (nonterm F))))) (rule (arrow F (symbol (nonterm F)))))"
        );
        assert_eq!( // [[[[[[[F]]]]]]];F~F -> [F];F~F
            simplify("(lsystem (angle 90) (axiom (symbol (bracket (symbol (bracket (symbol (bracket (symbol (bracket (symbol (bracket (symbol (bracket (symbol (bracket (symbol (nonterm F))))))))))))))))) (rule (arrow F (symbol (nonterm F)))))"),
            "(lsystem (angle 90) (axiom (symbol (bracket (symbol (nonterm F))))) (rule (arrow F (symbol (nonterm F)))))"
        )
    }

    #[test]
    fn test_retracing() {
        // retracing: [X]X => X
        // F;F~[F]F
        assert_eq!(
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) \
            (rule (arrow F (symbols (bracket (symbol (nonterm F))) (symbol (nonterm F))))))"),
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))"
        );
        // F;F~[FF]FF
        assert_eq!(
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) \
            (rule (arrow F (symbols (bracket (symbols (nonterm F) (symbol (nonterm F)))) \
            (symbols (nonterm F) (symbol (nonterm F)))))))"),
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) \
            (rule (arrow F (symbols (nonterm F) (symbol (nonterm F))))))"
        );
        // F;F~[+F-F]+F-F
        assert_eq!(
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) \
            (rule (arrow F (symbols (bracket (symbols (term +) (symbols (nonterm F) (symbols (term -) (symbol (nonterm F)))))) \
            (symbols (term +) (symbols (nonterm F) (symbols (term -) (symbol (nonterm F)))))))))"),
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F \
            (symbols (term +) (symbols (nonterm F) (symbols (term -) (symbol (nonterm F))))))))"
        );
    }

    #[test]
    fn test_bracketed_turns_only() {
        // only turns in brackets: [---+-+...--+] => empty symbols
        assert_eq!(
            // F;F~[-+-+---]F[++++] => F;F~F
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbols (bracket (symbols (term -) (symbols (term +) (symbols (term -) (symbols (term +) (symbols (term -) (symbols (term -) (symbol (term -))))))))) (symbols (nonterm F) (symbol (bracket (symbols (term +) (symbols (term +) (symbols (term +) (symbol (term +))))))))))))"),
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))"
        );
    }

    #[test]
    fn test_axiom_turns_only() {
        // only turns in axiom => empty axiom
        assert_eq!(
            // [---];F~F => empty
            simplify("(lsystem (angle 90) (axiom (symbol (bracket (symbols (term -) (symbols (term -) (symbol (term -))))))) (rule (arrow F (symbol (nonterm F)))))"),
            "nil"
        )
    }

     #[test]
    fn test_empty_rules() {
        assert_eq!(
            // F;F~+ => empty
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (term +)))))"),
            "nil"
        )
    }

    #[test]
    fn test_nil_collapse() {
        // F;F~nil => nil
        assert_eq!(
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol nil))))"),
            "nil"
        );
        // F;F~F,F~nil => F;F~F
        assert_eq!(
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) \
                                  (rules (arrow F (symbol (nonterm F))) \
                                         (rule (arrow F (symbol nil))))))"),
            "(lsystem (angle 90) (axiom (symbol (nonterm F))) (rule (arrow F (symbol (nonterm F)))))"
        );
        // F;F~nil,F~nil => nil
        assert_eq!(
            simplify("(lsystem (angle 90) (axiom (symbol (nonterm F))) \
                               (rules (arrow F nil) \
                                      (rule (arrow F nil))))"),
            "nil"
        );
    }
}
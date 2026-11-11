use super::*;
use crate::{challenge, constant, derived, opening, NoChallenges, NoInputs, NoOutputs};
use jolt_field::{Fr, Ring};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Opening {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Derived {
    Offset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Challenge {
    Gamma,
}

struct Mock;

impl SymbolicSumcheck for Mock {
    type RelationId = u8;
    type OpeningId = Opening;
    type DerivedId = Derived;
    type ChallengeId = Challenge;
    type Shape = ();
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new((): ()) -> Self {
        Self
    }

    fn id() -> u8 {
        7
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> Expr<F, Opening, Derived, Challenge> {
        Expr::zero()
    }

    fn output_expression<F: Ring>(&self) -> Expr<F, Opening, Derived, Challenge> {
        let two = constant::<F, _, _, _>(F::one() + F::one());
        (two * opening(Opening::A) + challenge(Challenge::Gamma))
            * (opening(Opening::B) + Expr::one())
            - derived(Derived::Offset) * opening(Opening::A)
    }
}

fn resolve(expr: &Expr<Fr, Opening, Derived, Challenge>) -> Fr {
    expr.evaluate(
        |id| match id {
            Opening::A => Fr::from_u64(3),
            Opening::B => Fr::from_u64(5),
        },
        |id| match id {
            Challenge::Gamma => Fr::from_u64(7),
        },
        |id| match id {
            Derived::Offset => Fr::from_u64(11),
        },
    )
}

#[test]
fn nested_output_expression_evaluates_to_hand_computed_value() {
    let output = resolve(&Mock::new(()).output_expression::<Fr>());
    assert_eq!(output, Fr::from_u64(45));
}

#[test]
fn expected_output_openings_deduplicate_and_skip_non_opening_leaves() {
    let openings = Mock::new(()).expected_output_openings::<Fr>();
    let expected: BTreeSet<Opening> = [Opening::A, Opening::B].into_iter().collect();
    assert_eq!(openings, expected);
}

#[test]
fn try_evaluate_propagates_resolver_errors_and_agrees_with_evaluate() {
    let expr = Mock::new(()).output_expression::<Fr>();

    let failed: Result<Fr, &str> = expr.try_evaluate(
        |id| match id {
            Opening::A => Ok(Fr::from_u64(3)),
            Opening::B => Err("missing opening B"),
        },
        |id| match id {
            Challenge::Gamma => Ok(Fr::from_u64(7)),
        },
        |id| match id {
            Derived::Offset => Ok(Fr::from_u64(11)),
        },
    );
    assert_eq!(failed, Err("missing opening B"));

    let succeeded: Result<Fr, &str> = expr.try_evaluate(
        |id| match id {
            Opening::A => Ok(Fr::from_u64(3)),
            Opening::B => Ok(Fr::from_u64(5)),
        },
        |id| match id {
            Challenge::Gamma => Ok(Fr::from_u64(7)),
        },
        |id| match id {
            Derived::Offset => Ok(Fr::from_u64(11)),
        },
    );
    assert_eq!(succeeded, Ok(resolve(&expr)));
}

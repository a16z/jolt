use jolt_field::RingCore;

use crate::{Expr, Source, SumcheckDomain};

/// Pure symbolic description of one sumcheck relation: its id, sumcheck spec, and
/// input/output algebra over the relation's id types. The expression methods are
/// generic over the field `F`, so the implementing object is field-independent (it
/// holds only its [`Shape`](Self::Shape)) and emits expressions for any field on
/// demand. See `specs/symbolic-sumcheck.md`.
pub trait SymbolicSumcheck {
    type RelationId;
    type OpeningId;
    type DerivedId;
    type ChallengeId;

    /// The construction input that fully determines this relation's structure (its
    /// expressions and sumcheck spec). Field-independent for every relation.
    type Shape;

    /// The relation's drawn Fiat-Shamir challenges, parameterized by the field
    /// (challenges carry no opening point). A relation with no challenges uses
    /// [`NoChallenges`](crate::NoChallenges).
    type Challenges<F>;

    /// The relation's consumed-claim struct, generic over the opening *cell*,
    /// instantiated at `F` (the serialized wire value) or `Vec<F>` (the derived
    /// opening point). A symbolic-only relation uses [`NoInputs`](crate::NoInputs).
    type Inputs<C>;
    /// The relation's produced-claim struct, generic over the opening *cell*
    /// (`F` value | `Vec<F>` point). A symbolic-only relation uses
    /// [`NoOutputs`](crate::NoOutputs).
    type Outputs<C>;

    fn new(shape: Self::Shape) -> Self;

    /// The relation this sumcheck belongs to. A type-level constant; NOT a unique
    /// key — several sumchecks can share one relation id (address/cycle-phase
    /// splits, the full/committed bytecode modes, the Spartan uni-skip/remainder
    /// pairs).
    fn id() -> Self::RelationId;

    /// The domain this sumcheck runs over. Defaults to the Boolean hypercube;
    /// only the univariate-skip relations override it (centered-integer domain).
    /// A fixed constant per relation, independent of the [`Shape`](Self::Shape);
    /// it takes `&self` only so it reads like its `rounds`/`degree` siblings at
    /// the (instance) call sites.
    fn domain(&self) -> SumcheckDomain {
        SumcheckDomain::BooleanHypercube
    }

    /// The sumcheck round count, derived from [`Shape`](Self::Shape).
    fn rounds(&self) -> usize;

    /// The per-round degree bound, derived from [`Shape`](Self::Shape).
    fn degree(&self) -> usize;

    fn input_expression<F: RingCore>(
        &self,
    ) -> Expr<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>;
    fn output_expression<F: RingCore>(
        &self,
    ) -> Expr<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>;

    /// The distinct opening ids this relation *produces*, read off its
    /// [`output_expression`](Self::output_expression) — which references every
    /// produced opening, expanded by the size parameters (it loops over indexed
    /// families like lookup tables and RA chunks). This is the relation's expected
    /// output-claim shape, derived symbolically so no relation hand-writes it; it
    /// holds because the output check constrains every produced opening (an
    /// unconstrained produced opening would be unsound). The field `F` only
    /// instantiates the expression — the ids are field-independent.
    fn expected_output_openings<F: RingCore>(&self) -> std::collections::BTreeSet<Self::OpeningId>
    where
        Self::OpeningId: Ord,
    {
        self.output_expression::<F>()
            .terms
            .into_iter()
            .flat_map(|term| term.factors)
            .filter_map(|factor| match factor {
                Source::Opening(id) => Some(id),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{challenge, constant, derived, opening, NoChallenges, NoInputs, NoOutputs};
    use jolt_field::{Fr, FromPrimitiveInt, RingCore};
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

    /// Zero-round mock: empty input sum, nested product-of-sums output mixing
    /// all three leaf kinds plus constants. `A` appears in several expanded
    /// terms so the produced-opening derivation must deduplicate.
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

        fn input_expression<F: RingCore>(&self) -> Expr<F, Opening, Derived, Challenge> {
            Expr::zero()
        }

        fn output_expression<F: RingCore>(&self) -> Expr<F, Opening, Derived, Challenge> {
            let two = constant::<F, _, _, _>(F::one() + F::one());
            // (2*A + gamma) * (B + 1) - offset * A
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

    /// (2*3 + 7) * (5 + 1) - 11*3 = 13*6 - 33 = 45, with each leaf kind
    /// resolved through its own resolver.
    #[test]
    fn nested_output_expression_evaluates_to_hand_computed_value() {
        let output = resolve(&Mock::new(()).output_expression::<Fr>());
        assert_eq!(output, Fr::from_u64(45));
    }

    /// An empty sum evaluates to zero without consulting any resolver, and
    /// derives an empty produced-opening set.
    #[test]
    fn empty_input_sum_evaluates_to_zero_and_produces_no_openings() {
        let relation = Mock::new(());
        let input = relation.input_expression::<Fr>().evaluate(
            |id| unreachable!("empty sum must not read opening {id:?}"),
            |id| unreachable!("empty sum must not read challenge {id:?}"),
            |id| unreachable!("empty sum must not read derived value {id:?}"),
        );
        assert_eq!(input, Fr::from_u64(0));

        struct EmptyOutput;
        impl SymbolicSumcheck for EmptyOutput {
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
                8
            }
            fn rounds(&self) -> usize {
                0
            }
            fn degree(&self) -> usize {
                0
            }
            fn input_expression<F: RingCore>(&self) -> Expr<F, Opening, Derived, Challenge> {
                Expr::zero()
            }
            fn output_expression<F: RingCore>(&self) -> Expr<F, Opening, Derived, Challenge> {
                Expr::zero()
            }
        }
        assert!(EmptyOutput::new(())
            .expected_output_openings::<Fr>()
            .is_empty());
    }

    /// The produced-opening derivation walks every expanded term's factors:
    /// it deduplicates the repeated `A`, keeps `B`, and never reports
    /// challenge, derived, or constant leaves as openings.
    #[test]
    fn expected_output_openings_deduplicate_and_skip_non_opening_leaves() {
        let openings = Mock::new(()).expected_output_openings::<Fr>();
        let expected: BTreeSet<Opening> = [Opening::A, Opening::B].into_iter().collect();
        assert_eq!(openings, expected);
    }

    /// `try_evaluate` surfaces the resolver's error verbatim instead of a
    /// value; a fully resolvable expression matches `evaluate` exactly.
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

    /// Relations that do not override `domain` run on the Boolean hypercube.
    #[test]
    fn default_domain_is_boolean_hypercube() {
        assert_eq!(Mock::new(()).domain(), SumcheckDomain::BooleanHypercube);
    }
}

use jolt_field::RingCore;

use crate::util::extend_unique;
use crate::{Expr, SumcheckSpec};

/// Pure symbolic description of one sumcheck relation: its id, sumcheck spec, and
/// input/output algebra over the relation's id types. The expression methods are
/// generic over the field `F`, so the implementing object is field-independent (it
/// holds only its [`Shape`](Self::Shape)) and emits expressions for any field on
/// demand. See `specs/symbolic-sumcheck.md`.
pub trait SymbolicSumcheck {
    type RelationId;
    type OpeningId;
    type PublicId;
    type ChallengeId;

    /// The construction input that fully determines this relation's structure (its
    /// expressions and sumcheck spec). Field-independent for every relation.
    type Shape;

    fn new(shape: Self::Shape) -> Self;

    /// The relation this sumcheck belongs to. A type-level constant; NOT a unique
    /// key — several sumchecks can share one relation id (address/cycle-phase
    /// splits, the full/committed bytecode modes, the Spartan uni-skip/remainder
    /// pairs).
    fn id() -> Self::RelationId;

    /// The sumcheck spec, derived from [`Shape`](Self::Shape).
    fn spec(&self) -> SumcheckSpec;

    fn input_expression<F: RingCore>(
        &self,
    ) -> Expr<F, Self::OpeningId, Self::PublicId, Self::ChallengeId>;
    fn output_expression<F: RingCore>(
        &self,
    ) -> Expr<F, Self::OpeningId, Self::PublicId, Self::ChallengeId>;

    /// Openings referenced by either expression, input first, deduplicated.
    fn required_openings<F: RingCore>(&self) -> Vec<Self::OpeningId>
    where
        Self::OpeningId: Clone + Eq,
    {
        let mut ids = self.input_expression::<F>().required_openings();
        extend_unique(&mut ids, &self.output_expression::<F>().required_openings());
        ids
    }

    /// Public values referenced by either expression, input first, deduplicated.
    fn required_publics<F: RingCore>(&self) -> Vec<Self::PublicId>
    where
        Self::PublicId: Clone + Eq,
    {
        let mut ids = self.input_expression::<F>().required_publics();
        extend_unique(&mut ids, &self.output_expression::<F>().required_publics());
        ids
    }

    /// Fiat-Shamir challenges referenced by either expression, input first,
    /// deduplicated. This is the canonical challenge set; it reproduces the
    /// transcript-sync set the removed `with_input_challenges` declared.
    fn required_challenges<F: RingCore>(&self) -> Vec<Self::ChallengeId>
    where
        Self::ChallengeId: Clone + Eq,
    {
        let mut ids = self.input_expression::<F>().required_challenges();
        extend_unique(
            &mut ids,
            &self.output_expression::<F>().required_challenges(),
        );
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{challenge, opening, public, Expr, SumcheckSpec};
    use jolt_field::Fr;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum O {
        A,
        B,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum P {
        X,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Ch {
        G,
    }

    struct Dummy;
    impl SymbolicSumcheck for Dummy {
        type RelationId = u8;
        type OpeningId = O;
        type PublicId = P;
        type ChallengeId = Ch;
        type Shape = ();
        fn new((): ()) -> Self {
            Self
        }
        fn id() -> u8 {
            7
        }
        fn spec(&self) -> SumcheckSpec {
            SumcheckSpec::boolean(3, 1)
        }
        fn input_expression<F: jolt_field::RingCore>(&self) -> Expr<F, O, P, Ch> {
            opening(O::A) + challenge(Ch::G) * opening(O::B)
        }
        fn output_expression<F: jolt_field::RingCore>(&self) -> Expr<F, O, P, Ch> {
            public(P::X) * opening(O::B)
        }
    }

    #[test]
    fn required_sets_are_input_then_output_deduped() {
        let d = Dummy;
        assert_eq!(d.required_openings::<Fr>(), vec![O::A, O::B]); // A from input, B from input
        assert_eq!(d.required_publics::<Fr>(), vec![P::X]);
        assert_eq!(d.required_challenges::<Fr>(), vec![Ch::G]);
        assert_eq!(Dummy::id(), 7);
    }
}

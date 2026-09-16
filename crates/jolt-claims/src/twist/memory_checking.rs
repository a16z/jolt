//! The Twist read/write-checking and value-evaluation identities.

use jolt_field::Ring;

use crate::{challenge, derived, opening, Expr};

/// Id supplier for the register read/write-checking shape: three consumed
/// value openings folded by `gamma`, five produced openings at the shared
/// read-write point, weighted by the `EqCycle` public.
pub trait ReadWriteCheckingIds {
    type OpeningId: Clone;
    type DerivedId: Clone;
    type ChallengeId: Clone;

    /// Consumed value openings, in `gamma` power order (γ⁰, γ¹, γ²).
    fn rd_value() -> Self::OpeningId;
    fn rs1_value() -> Self::OpeningId;
    fn rs2_value() -> Self::OpeningId;

    /// Produced openings at the shared read-write point.
    fn registers_val() -> Self::OpeningId;
    fn rs1_ra() -> Self::OpeningId;
    fn rs2_ra() -> Self::OpeningId;
    fn rd_wa() -> Self::OpeningId;
    fn rd_inc() -> Self::OpeningId;

    fn gamma() -> Self::ChallengeId;
    fn eq_cycle() -> Self::DerivedId;
}

/// Per-round degree bound of the read/write-checking relation
/// (`EqCycle · access · Val`). Rounds source: the protocol's read-write
/// dimensions (`read_write_rounds()`).
pub const READ_WRITE_CHECKING_DEGREE: usize = 3;

/// `rd + γ·rs1 + γ²·rs2` over the consumed value openings.
pub fn read_write_checking_input<F: Ring, S: ReadWriteCheckingIds>(
) -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> {
    let gamma = challenge(S::gamma());
    opening(S::rd_value())
        + gamma.clone() * opening(S::rs1_value())
        + gamma.clone().pow(2) * opening(S::rs2_value())
}

/// `EqCycle · (RdWa·Inc + RdWa·Val + γ·Rs1Ra·Val + γ²·Rs2Ra·Val)`, expanded.
pub fn read_write_checking_output<F: Ring, S: ReadWriteCheckingIds>(
) -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> {
    let gamma = challenge(S::gamma());
    let eq_cycle = derived(S::eq_cycle());
    eq_cycle.clone() * opening(S::rd_wa()) * opening(S::rd_inc())
        + eq_cycle.clone() * opening(S::rd_wa()) * opening(S::registers_val())
        + eq_cycle.clone() * gamma.clone() * opening(S::rs1_ra()) * opening(S::registers_val())
        + eq_cycle * gamma.pow(2) * opening(S::rs2_ra()) * opening(S::registers_val())
}

/// Id supplier for the register val-evaluation shape: one consumed `Val`
/// opening, two produced openings weighted by the `LtCycle` public.
pub trait ValEvaluationIds {
    type OpeningId: Clone;
    type DerivedId: Clone;
    type ChallengeId: Clone;

    /// Consumed `Val` opening (at the upstream read-write point).
    fn registers_val() -> Self::OpeningId;

    /// Produced openings at the shared val-evaluation point.
    fn rd_inc() -> Self::OpeningId;
    fn rd_wa() -> Self::OpeningId;

    fn lt_cycle() -> Self::DerivedId;
}

/// Per-round degree bound of the value-evaluation relation (`LtCycle · Inc · Wa`).
/// Rounds source: the protocol's trace dimensions (`log_t()`).
pub const VAL_EVALUATION_DEGREE: usize = 3;

/// The bare consumed `Val` opening.
pub fn val_evaluation_input<F: Ring, S: ValEvaluationIds>(
) -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> {
    opening(S::registers_val())
}

/// `LtCycle · Inc · Wa`.
pub fn val_evaluation_output<F: Ring, S: ValEvaluationIds>(
) -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> {
    derived(S::lt_cycle()) * opening(S::rd_inc()) * opening(S::rd_wa())
}

/// Binds the read/write-checking identity to one memory instance from a
/// compact id mapping table. Expands to two impls on `$relation` (which must
/// be `struct $relation { shape: $dimensions }`): the
/// [`ReadWriteCheckingIds`] supplier mapping each role to the given id
/// constructor, and [`SymbolicSumcheck`](crate::SymbolicSumcheck) delegating
/// expressions and degree to this identity, with rounds =
/// `shape.read_write_rounds()` (the dimensions' rounds source).
macro_rules! instantiate_read_write_checking {
    (
        relation = $relation:ident,
        id = $id:expr,
        ids = ($relation_id:ty, $opening_id:ty, $derived_id:ty, $challenge_id:ty),
        dimensions = $dimensions:ty,
        challenges = $challenges:ident,
        inputs = $inputs:ident,
        outputs = $outputs:ident,
        rd_value = $rd_value:expr,
        rs1_value = $rs1_value:expr,
        rs2_value = $rs2_value:expr,
        registers_val = $registers_val:expr,
        rs1_ra = $rs1_ra:expr,
        rs2_ra = $rs2_ra:expr,
        rd_wa = $rd_wa:expr,
        rd_inc = $rd_inc:expr,
        gamma = $gamma:expr,
        eq_cycle = $eq_cycle:expr,
    ) => {
        impl $crate::twist::memory_checking::ReadWriteCheckingIds for $relation {
            type OpeningId = $opening_id;
            type DerivedId = $derived_id;
            type ChallengeId = $challenge_id;

            fn rd_value() -> $opening_id {
                $rd_value
            }
            fn rs1_value() -> $opening_id {
                $rs1_value
            }
            fn rs2_value() -> $opening_id {
                $rs2_value
            }
            fn registers_val() -> $opening_id {
                $registers_val
            }
            fn rs1_ra() -> $opening_id {
                $rs1_ra
            }
            fn rs2_ra() -> $opening_id {
                $rs2_ra
            }
            fn rd_wa() -> $opening_id {
                $rd_wa
            }
            fn rd_inc() -> $opening_id {
                $rd_inc
            }
            fn gamma() -> $challenge_id {
                $gamma.into()
            }
            fn eq_cycle() -> $derived_id {
                $eq_cycle.into()
            }
        }

        impl $crate::SymbolicSumcheck for $relation {
            type RelationId = $relation_id;
            type OpeningId = $opening_id;
            type DerivedId = $derived_id;
            type ChallengeId = $challenge_id;
            type Shape = $dimensions;
            type Challenges<F> = $challenges<F>;
            type Inputs<C> = $inputs<C>;
            type Outputs<C> = $outputs<C>;

            fn new(shape: $dimensions) -> Self {
                Self { shape }
            }
            fn id() -> $relation_id {
                $id
            }
            fn rounds(&self) -> usize {
                self.shape.read_write_rounds()
            }
            fn degree(&self) -> usize {
                $crate::twist::memory_checking::READ_WRITE_CHECKING_DEGREE
            }
            fn input_expression<F: ::jolt_field::Ring>(
                &self,
            ) -> $crate::Expr<F, $opening_id, $derived_id, $challenge_id> {
                $crate::twist::memory_checking::read_write_checking_input::<F, Self>()
            }
            fn output_expression<F: ::jolt_field::Ring>(
                &self,
            ) -> $crate::Expr<F, $opening_id, $derived_id, $challenge_id> {
                $crate::twist::memory_checking::read_write_checking_output::<F, Self>()
            }
        }
    };
}
pub(crate) use instantiate_read_write_checking;

/// Binds the value-evaluation identity to one memory instance from a
/// compact id mapping table. Same expansion contract as
/// [`instantiate_read_write_checking`], with rounds = `shape.log_t()` (the
/// dimensions' rounds source).
macro_rules! instantiate_val_evaluation {
    (
        relation = $relation:ident,
        id = $id:expr,
        ids = ($relation_id:ty, $opening_id:ty, $derived_id:ty, $challenge_id:ty),
        dimensions = $dimensions:ty,
        challenges = $challenges:ident,
        inputs = $inputs:ident,
        outputs = $outputs:ident,
        registers_val = $registers_val:expr,
        rd_inc = $rd_inc:expr,
        rd_wa = $rd_wa:expr,
        lt_cycle = $lt_cycle:expr,
    ) => {
        impl $crate::twist::memory_checking::ValEvaluationIds for $relation {
            type OpeningId = $opening_id;
            type DerivedId = $derived_id;
            type ChallengeId = $challenge_id;

            fn registers_val() -> $opening_id {
                $registers_val
            }
            fn rd_inc() -> $opening_id {
                $rd_inc
            }
            fn rd_wa() -> $opening_id {
                $rd_wa
            }
            fn lt_cycle() -> $derived_id {
                $lt_cycle.into()
            }
        }

        impl $crate::SymbolicSumcheck for $relation {
            type RelationId = $relation_id;
            type OpeningId = $opening_id;
            type DerivedId = $derived_id;
            type ChallengeId = $challenge_id;
            type Shape = $dimensions;
            type Challenges<F> = $challenges<F>;
            type Inputs<C> = $inputs<C>;
            type Outputs<C> = $outputs<C>;

            fn new(shape: $dimensions) -> Self {
                Self { shape }
            }
            fn id() -> $relation_id {
                $id
            }
            fn rounds(&self) -> usize {
                self.shape.log_t()
            }
            fn degree(&self) -> usize {
                $crate::twist::memory_checking::VAL_EVALUATION_DEGREE
            }
            fn input_expression<F: ::jolt_field::Ring>(
                &self,
            ) -> $crate::Expr<F, $opening_id, $derived_id, $challenge_id> {
                $crate::twist::memory_checking::val_evaluation_input::<F, Self>()
            }
            fn output_expression<F: ::jolt_field::Ring>(
                &self,
            ) -> $crate::Expr<F, $opening_id, $derived_id, $challenge_id> {
                $crate::twist::memory_checking::val_evaluation_output::<F, Self>()
            }
        }
    };
}
pub(crate) use instantiate_val_evaluation;

#[cfg(test)]
mod tests {
    use super::super::test_ids::{Challenge, Derived, Opening};
    use super::*;
    use jolt_field::Fr;

    struct Toy;

    impl ReadWriteCheckingIds for Toy {
        type OpeningId = Opening;
        type DerivedId = Derived;
        type ChallengeId = Challenge;

        fn rd_value() -> Opening {
            Opening::In(0)
        }
        fn rs1_value() -> Opening {
            Opening::In(1)
        }
        fn rs2_value() -> Opening {
            Opening::In(2)
        }
        fn registers_val() -> Opening {
            Opening::Out(0)
        }
        fn rs1_ra() -> Opening {
            Opening::Out(1)
        }
        fn rs2_ra() -> Opening {
            Opening::Out(2)
        }
        fn rd_wa() -> Opening {
            Opening::Out(3)
        }
        fn rd_inc() -> Opening {
            Opening::Out(4)
        }
        fn gamma() -> Challenge {
            Challenge::Gamma
        }
        fn eq_cycle() -> Derived {
            Derived::Eq(0)
        }
    }

    impl ValEvaluationIds for Toy {
        type OpeningId = Opening;
        type DerivedId = Derived;
        type ChallengeId = Challenge;

        fn registers_val() -> Opening {
            Opening::In(0)
        }
        fn rd_inc() -> Opening {
            Opening::Out(0)
        }
        fn rd_wa() -> Opening {
            Opening::Out(1)
        }
        fn lt_cycle() -> Derived {
            Derived::Eq(0)
        }
    }

    /// Structural pin: the builders must reproduce the previously hand-written
    /// term sequence exactly (the BlindFold lowering consumes terms in order).
    #[test]
    fn read_write_checking_terms_match_the_hand_written_construction() {
        let gamma: Expr<Fr, Opening, Derived, Challenge> = challenge(Challenge::Gamma);
        let expected_input = opening(Opening::In(0))
            + gamma.clone() * opening(Opening::In(1))
            + gamma.clone().pow(2) * opening(Opening::In(2));
        assert_eq!(
            read_write_checking_input::<Fr, Toy>(),
            expected_input,
            "input term order drifted from the hand-written construction"
        );

        let eq_cycle: Expr<Fr, Opening, Derived, Challenge> = derived(Derived::Eq(0));
        let expected_output =
            eq_cycle.clone() * opening(Opening::Out(3)) * opening(Opening::Out(4))
                + eq_cycle.clone() * opening(Opening::Out(3)) * opening(Opening::Out(0))
                + eq_cycle.clone()
                    * gamma.clone()
                    * opening(Opening::Out(1))
                    * opening(Opening::Out(0))
                + eq_cycle * gamma.pow(2) * opening(Opening::Out(2)) * opening(Opening::Out(0));
        assert_eq!(
            read_write_checking_output::<Fr, Toy>(),
            expected_output,
            "output term order drifted from the hand-written construction"
        );
    }

    #[test]
    fn val_evaluation_terms_match_the_hand_written_construction() {
        let expected_input: Expr<Fr, Opening, Derived, Challenge> = opening(Opening::In(0));
        assert_eq!(val_evaluation_input::<Fr, Toy>(), expected_input);

        let expected_output: Expr<Fr, Opening, Derived, Challenge> =
            derived(Derived::Eq(0)) * opening(Opening::Out(0)) * opening(Opening::Out(1));
        assert_eq!(val_evaluation_output::<Fr, Toy>(), expected_output);
    }
}

//! The Twist claim-reduction identities: the value reduction (γ-fold to
//! the Spartan point) and the increment reduction (per-source pairs folded by
//! a drawn challenge under Eq-pair publics).

use jolt_field::Ring;

use crate::{challenge, derived, opening, Expr};

/// Id supplier for the value claim-reduction shape: three upstream openings
/// folded by `gamma`, reduced to three openings at the shared reduction point,
/// weighted by the `EqSpartan` public.
pub trait ValueReductionIds {
    type OpeningId: Clone;
    type DerivedId: Clone;
    type ChallengeId: Clone;

    /// Consumed upstream openings, in `gamma` power order (γ⁰, γ¹, γ²).
    fn consumed() -> [Self::OpeningId; 3];
    /// Produced reduced openings, paired index-for-index with
    /// [`consumed`](Self::consumed).
    fn reduced() -> [Self::OpeningId; 3];

    fn gamma() -> Self::ChallengeId;
    fn eq_spartan() -> Self::DerivedId;
}

/// Per-round degree bound of the value claim-reduction relation
/// (`EqSpartan · reduced`). Rounds source: the protocol's trace dimensions
/// (`log_t()`).
pub const VALUE_REDUCTION_DEGREE: usize = 2;

/// `c₀ + γ·c₁ + γ²·c₂` over the consumed openings.
pub fn value_reduction_input<F: Ring, S: ValueReductionIds>(
) -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> {
    let gamma: Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> = challenge(S::gamma());
    let [c0, c1, c2] = S::consumed();
    opening(c0) + gamma.clone() * opening(c1) + gamma.clone().pow(2) * opening(c2)
}

/// `EqSpartan · (r₀ + γ·r₁ + γ²·r₂)`, expanded.
pub fn value_reduction_output<F: Ring, S: ValueReductionIds>(
) -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> {
    let gamma: Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> = challenge(S::gamma());
    let eq_spartan = derived(S::eq_spartan());
    let [r0, r1, r2] = S::reduced();
    eq_spartan.clone() * opening(r0)
        + eq_spartan.clone() * gamma.clone() * opening(r1)
        + eq_spartan * gamma.pow(2) * opening(r2)
}

/// One increment-reduction source group: two semantic openings of the same
/// committed polynomial (read/write then val flavored), the Eq publics pairing
/// this reduction's cycle with each source's cycle, and the single reduced
/// opening they fold into.
pub struct IncrementReductionGroup<O, P> {
    /// Consumed openings, in `gamma` power order within the group
    /// (read/write at γ⁰, val at γ¹, relative to the group offset).
    pub consumed: [O; 2],
    /// Eq publics paired index-for-index with [`consumed`](Self::consumed).
    pub eq_publics: [P; 2],
    /// The produced reduced opening the group folds into.
    pub reduced: O,
}

/// Id supplier for the increment claim-reduction shape: `groups()` source
/// groups, group `g` riding the γ^(2g) offset.
pub trait IncrementReductionIds {
    type OpeningId: Clone;
    type DerivedId: Clone;
    type ChallengeId: Clone;

    /// The source groups in `gamma` offset order.
    fn groups() -> Vec<IncrementReductionGroup<Self::OpeningId, Self::DerivedId>>;

    fn gamma() -> Self::ChallengeId;
}

/// Per-round degree bound of the increment claim-reduction relation
/// (`Eq · reduced`). Rounds source: the protocol's trace dimensions
/// (`log_t()`).
pub const INCREMENT_REDUCTION_DEGREE: usize = 2;

/// `Σ_g γ^(2g) · (consumed_rw + γ·consumed_val)` over the source groups.
pub fn increment_reduction_input<F: Ring, S: IncrementReductionIds>(
) -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> {
    let gamma: Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> = challenge(S::gamma());
    let mut expr = Expr::zero();
    for (group_index, group) in S::groups().into_iter().enumerate() {
        let [read_write, val] = group.consumed;
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "group counts are tiny constants (at most two per protocol family)"
        )]
        let offset = 2 * group_index;
        expr = expr
            + gamma.clone().pow(offset) * opening(read_write)
            + gamma.clone().pow(offset + 1) * opening(val);
    }
    expr
}

/// `Σ_g γ^(2g) · (Eq_rw + γ·Eq_val) · reduced_g`, expanded.
pub fn increment_reduction_output<F: Ring, S: IncrementReductionIds>(
) -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> {
    let gamma: Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId> = challenge(S::gamma());
    let mut expr = Expr::zero();
    for (group_index, group) in S::groups().into_iter().enumerate() {
        let [eq_read_write, eq_val] = group.eq_publics;
        let coefficient = derived(eq_read_write) + gamma.clone() * derived(eq_val);
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "group counts are tiny constants (at most two per protocol family)"
        )]
        let offset = 2 * group_index;
        expr = expr + gamma.clone().pow(offset) * coefficient * opening(group.reduced);
    }
    expr
}

/// Binds the value claim-reduction identity to one memory instance from
/// a compact id mapping table. Expands to two impls on `$relation` (which must
/// be `struct $relation { shape: $dimensions }`): the
/// [`ValueReductionIds`] supplier and
/// [`SymbolicSumcheck`](crate::SymbolicSumcheck) delegating expressions and
/// degree to this shape, with rounds = `shape.log_t()` (the shape's rounds
/// source).
macro_rules! instantiate_value_reduction {
    (
        relation = $relation:ident,
        id = $id:expr,
        ids = ($relation_id:ty, $opening_id:ty, $derived_id:ty, $challenge_id:ty),
        dimensions = $dimensions:ty,
        challenges = $challenges:ident,
        inputs = $inputs:ident,
        outputs = $outputs:ident,
        consumed = $consumed:expr,
        reduced = $reduced:expr,
        gamma = $gamma:expr,
        eq_spartan = $eq_spartan:expr,
    ) => {
        impl $crate::twist::claim_reductions::ValueReductionIds for $relation {
            type OpeningId = $opening_id;
            type DerivedId = $derived_id;
            type ChallengeId = $challenge_id;

            fn consumed() -> [$opening_id; 3] {
                $consumed
            }
            fn reduced() -> [$opening_id; 3] {
                $reduced
            }
            fn gamma() -> $challenge_id {
                $gamma.into()
            }
            fn eq_spartan() -> $derived_id {
                $eq_spartan.into()
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
                $crate::twist::claim_reductions::VALUE_REDUCTION_DEGREE
            }
            fn input_expression<F: ::jolt_field::Ring>(
                &self,
            ) -> $crate::Expr<F, $opening_id, $derived_id, $challenge_id> {
                $crate::twist::claim_reductions::value_reduction_input::<F, Self>()
            }
            fn output_expression<F: ::jolt_field::Ring>(
                &self,
            ) -> $crate::Expr<F, $opening_id, $derived_id, $challenge_id> {
                $crate::twist::claim_reductions::value_reduction_output::<F, Self>()
            }
        }
    };
}
pub(crate) use instantiate_value_reduction;

/// Binds the increment claim-reduction identity to one memory instance
/// from a compact source-group table. Same expansion contract as
/// [`instantiate_value_reduction`]; `groups` is the
/// [`IncrementReductionGroup`] vector in γ offset order, with rounds =
/// `shape.log_t()` (the shape's rounds source).
macro_rules! instantiate_increment_reduction {
    (
        relation = $relation:ident,
        id = $id:expr,
        ids = ($relation_id:ty, $opening_id:ty, $derived_id:ty, $challenge_id:ty),
        dimensions = $dimensions:ty,
        challenges = $challenges:ident,
        inputs = $inputs:ident,
        outputs = $outputs:ident,
        groups = $groups:expr,
        gamma = $gamma:expr,
    ) => {
        impl $crate::twist::claim_reductions::IncrementReductionIds for $relation {
            type OpeningId = $opening_id;
            type DerivedId = $derived_id;
            type ChallengeId = $challenge_id;

            fn groups() -> Vec<
                $crate::twist::claim_reductions::IncrementReductionGroup<$opening_id, $derived_id>,
            > {
                $groups
            }
            fn gamma() -> $challenge_id {
                $gamma.into()
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
                $crate::twist::claim_reductions::INCREMENT_REDUCTION_DEGREE
            }
            fn input_expression<F: ::jolt_field::Ring>(
                &self,
            ) -> $crate::Expr<F, $opening_id, $derived_id, $challenge_id> {
                $crate::twist::claim_reductions::increment_reduction_input::<F, Self>()
            }
            fn output_expression<F: ::jolt_field::Ring>(
                &self,
            ) -> $crate::Expr<F, $opening_id, $derived_id, $challenge_id> {
                $crate::twist::claim_reductions::increment_reduction_output::<F, Self>()
            }
        }
    };
}
pub(crate) use instantiate_increment_reduction;

#[cfg(test)]
mod tests {
    use super::super::test_ids::{Challenge, Derived, Opening};
    use super::*;
    use jolt_field::Fr;

    struct Toy;

    impl ValueReductionIds for Toy {
        type OpeningId = Opening;
        type DerivedId = Derived;
        type ChallengeId = Challenge;

        fn consumed() -> [Opening; 3] {
            [Opening::In(0), Opening::In(1), Opening::In(2)]
        }
        fn reduced() -> [Opening; 3] {
            [Opening::Out(0), Opening::Out(1), Opening::Out(2)]
        }
        fn gamma() -> Challenge {
            Challenge::Gamma
        }
        fn eq_spartan() -> Derived {
            Derived::Eq(0)
        }
    }

    /// Two-group supplier shaped like the jolt increments reduction (RAM then
    /// registers); the one-group FR shape is its prefix.
    struct TwoGroups;

    impl IncrementReductionIds for TwoGroups {
        type OpeningId = Opening;
        type DerivedId = Derived;
        type ChallengeId = Challenge;

        fn groups() -> Vec<IncrementReductionGroup<Opening, Derived>> {
            vec![
                IncrementReductionGroup {
                    consumed: [Opening::In(0), Opening::In(1)],
                    eq_publics: [Derived::Eq(0), Derived::Eq(1)],
                    reduced: Opening::Out(0),
                },
                IncrementReductionGroup {
                    consumed: [Opening::In(2), Opening::In(3)],
                    eq_publics: [Derived::Eq(2), Derived::Eq(3)],
                    reduced: Opening::Out(1),
                },
            ]
        }
        fn gamma() -> Challenge {
            Challenge::Gamma
        }
    }

    /// Structural pin: the builders must reproduce the previously hand-written
    /// term sequence exactly (the BlindFold lowering consumes terms in order).
    #[test]
    fn value_reduction_terms_match_the_hand_written_construction() {
        let gamma: Expr<Fr, Opening, Derived, Challenge> = challenge(Challenge::Gamma);
        let expected_input = opening(Opening::In(0))
            + gamma.clone() * opening(Opening::In(1))
            + gamma.clone().pow(2) * opening(Opening::In(2));
        assert_eq!(value_reduction_input::<Fr, Toy>(), expected_input);

        let eq_spartan: Expr<Fr, Opening, Derived, Challenge> = derived(Derived::Eq(0));
        let expected_output = eq_spartan.clone() * opening(Opening::Out(0))
            + eq_spartan.clone() * gamma.clone() * opening(Opening::Out(1))
            + eq_spartan * gamma.pow(2) * opening(Opening::Out(2));
        assert_eq!(value_reduction_output::<Fr, Toy>(), expected_output);
    }

    /// Structural pin against the previously hand-written jolt construction (the four-way
    /// γ-fold input and the two-group output fold): group 0 carries no γ
    /// offset factors, group 1 rides γ².
    #[test]
    fn increment_reduction_terms_match_the_hand_written_construction() {
        let gamma: Expr<Fr, Opening, Derived, Challenge> = challenge(Challenge::Gamma);
        let expected_input = opening(Opening::In(0))
            + gamma.clone() * opening(Opening::In(1))
            + gamma.clone().pow(2) * opening(Opening::In(2))
            + gamma.clone().pow(3) * opening(Opening::In(3));
        assert_eq!(increment_reduction_input::<Fr, TwoGroups>(), expected_input);

        let group0: Expr<Fr, Opening, Derived, Challenge> =
            derived(Derived::Eq(0)) + gamma.clone() * derived(Derived::Eq(1));
        let group1: Expr<Fr, Opening, Derived, Challenge> =
            derived(Derived::Eq(2)) + gamma.clone() * derived(Derived::Eq(3));
        let expected_output =
            group0 * opening(Opening::Out(0)) + gamma.pow(2) * group1 * opening(Opening::Out(1));
        assert_eq!(
            increment_reduction_output::<Fr, TwoGroups>(),
            expected_output
        );
    }
}

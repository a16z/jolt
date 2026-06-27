//! Transcript-free claim data model: the opening cells, the `*Claims` resolvers,
//! and the value↔point zip.
//!
//! This is the *order + id-resolution* half of the per-relation opening-claim
//! plumbing. It is implemented by `#[derive(OutputClaims)]` /
//! `#[derive(InputClaims)]` on each relation's claim struct and makes the
//! canonical opening **order** and **count** a single-sourced consequence of a
//! struct's field declaration order, instead of the hand-written copies that
//! historically drift apart.
//!
//! Transcript I/O (`append_openings`, `draw_challenges`) deliberately lives in
//! `jolt-verifier`: `jolt-claims` stays free of any `Transcript` dependency. The
//! verifier-side `append_openings` is a thin consumer of [`OutputClaims::opening_values`],
//! so it cannot disagree with the canonical order defined here.

use jolt_field::Field;

use crate::protocols::jolt::{JoltChallengeId, JoltOpeningId};

/// Canonical encoders and the output-formula resolver for a relation's
/// *produced* opening-claim struct.
///
/// The implementor's field declaration order is the single definition of
/// canonical opening order: [`opening_values`](Self::opening_values) and
/// [`opening_count`](Self::opening_count) derive from it, so the Fiat-Shamir
/// `append_openings` (in `jolt-verifier`) that iterates `opening_values()` cannot
/// disagree.
///
/// Generic over the opening-id type `O` (defaulting to [`JoltOpeningId`]) so the
/// trait can live in the framework half and be reused by other protocol families.
pub trait OutputClaims<F: Field, O = JoltOpeningId> {
    /// Produced opening scalars in canonical (field-declaration) order.
    fn opening_values(&self) -> Vec<F>;

    /// Number of produced openings; equals `opening_values().len()` but is
    /// computed without allocating.
    fn opening_count(&self) -> usize;

    /// Resolve a produced opening's value by id, for evaluating the relation's
    /// output `Expr`. Returns `None` for ids this struct does not carry (callers
    /// turn that into a `MissingOpeningClaim` error).
    fn resolve_output(&self, id: &O) -> Option<F>;
}

/// The input-formula resolver for a relation's *consumed* opening-claim struct
/// (populated by explicit cross-stage wiring).
///
/// Generic over the opening-id type `O` (defaulting to [`JoltOpeningId`]).
pub trait InputClaims<F: Field, O = JoltOpeningId> {
    /// Resolve a consumed opening's value by id, for evaluating the relation's
    /// input `Expr`. Returns `None` for ids this struct does not carry.
    fn resolve_input(&self, id: &O) -> Option<F>;
}

/// The challenge resolver for a relation's drawn Fiat-Shamir challenges.
///
/// Unlike the opening-claim resolvers, challenges carry no opening point, so the
/// implementor is generic over the field `F` directly (not an opening cell) and
/// reads each field's value without `GetValue` indirection.
///
/// Generic over the challenge-id type `C` (defaulting to [`JoltChallengeId`]).
pub trait SumcheckChallenges<F: Field, C = JoltChallengeId> {
    /// Resolve a drawn Fiat-Shamir challenge by id, for evaluating a relation's
    /// input/output `Expr`. Returns `None` for ids this struct does not carry.
    fn resolve_challenge(&self, id: &C) -> Option<F>;
}

/// `Challenges` for a relation that draws no Fiat-Shamir challenges: resolves
/// every id to `None`. Generic over the field so it fits `type Challenges<F>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoChallenges<F>(::core::marker::PhantomData<F>);

impl<F: Field, C> SumcheckChallenges<F, C> for NoChallenges<F> {
    fn resolve_challenge(&self, _id: &C) -> Option<F> {
        None
    }
}

/// One opening-claim cell: a `(point, value)` pair. The opening point is
/// verifier-derived (from the sumcheck), so it never crosses the wire — only the
/// value is serialized into the proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningClaim<F> {
    pub point: Vec<F>,
    pub value: F,
}

/// A claim-struct cell that exposes an opening point. Implemented by the
/// point-only ZK cell (`Vec<F>`) and the clear cell (`OpeningClaim<F>`).
pub trait GetPoint<F> {
    fn point(&self) -> &[F];
}

/// A claim-struct cell that exposes an opening value. Implemented by the
/// value-only wire cell (`F`) and the clear cell (`OpeningClaim<F>`).
pub trait GetValue<F> {
    fn value(&self) -> F;
}

impl<F: Field> GetPoint<F> for Vec<F> {
    fn point(&self) -> &[F] {
        self.as_slice()
    }
}

impl<F: Field> GetValue<F> for F {
    fn value(&self) -> F {
        *self
    }
}

impl<F: Field> GetPoint<F> for OpeningClaim<F> {
    fn point(&self) -> &[F] {
        &self.point
    }
}

impl<F: Field> GetValue<F> for OpeningClaim<F> {
    fn value(&self) -> F {
        self.value
    }
}

/// A produced-claim struct in its clear `OpeningClaim<F>` (point + value) cell
/// form, reconstructible by pairing the value-only (`F`) and point-only (`Vec<F>`)
/// cell forms of the same struct field-by-field. `#[derive(OutputClaims)]` emits
/// the implementation (one `OpeningClaim` per leaf, element-wise for `Vec`
/// families, value-driven for `Option` leaves), so callers reach it through the
/// free [`zip_openings`] function instead of hand-writing the pairing.
pub trait ZipOpenings<F: Field>: Sized {
    /// The value-only (`F`-cell) form — the serialized wire claims.
    type Values;
    /// The point-only (`Vec<F>`-cell) form — the derived opening points.
    type Points;
    /// Pair each opening's value with its derived point.
    fn zip_openings(values: &Self::Values, points: &Self::Points) -> Self;
}

/// Pair a relation's value-only claims with its point-only claims into the clear
/// `OpeningClaim<F>` form, single-sourcing the per-field `(point, value)` pairing
/// that each stage's `*_output_claims_with_points` helper used to hand-write.
pub fn zip_openings<F: Field, T: ZipOpenings<F>>(values: &T::Values, points: &T::Points) -> T {
    T::zip_openings(values, points)
}

#[cfg(test)]
mod sumcheck_challenges_tests {
    use crate::protocols::jolt::{
        BooleanityChallenge, BytecodeClaimReductionChallenge, JoltChallengeId,
        RamReadWriteChallenge,
    };
    // The `SumcheckChallenges` re-export from the crate root covers both the trait
    // (type namespace) and the derive macro (macro namespace).
    use crate::SumcheckChallenges;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    #[derive(SumcheckChallenges)]
    struct ScalarChallenge<F> {
        #[challenge(RamReadWriteChallenge::Gamma)]
        gamma: F,
    }

    #[test]
    fn scalar_resolves_matching_id_and_misses_others() {
        let challenges = ScalarChallenge { gamma: fr(7) };

        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(RamReadWriteChallenge::Gamma)),
            Some(fr(7)),
        );
        // An unrelated id (different sub-enum) resolves to `None`.
        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(BooleanityChallenge::Gamma)),
            None,
        );
    }

    #[derive(SumcheckChallenges)]
    struct OptionChallenge<F> {
        #[challenge(BytecodeClaimReductionChallenge::Eta)]
        eta: Option<F>,
    }

    #[test]
    fn option_resolves_only_when_present() {
        let id = JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta);

        let present = OptionChallenge { eta: Some(fr(9)) };
        assert_eq!(present.resolve_challenge(&id), Some(fr(9)));

        let absent: OptionChallenge<Fr> = OptionChallenge { eta: None };
        assert_eq!(absent.resolve_challenge(&id), None);
    }

    #[derive(SumcheckChallenges)]
    struct MultiChallenge<F> {
        #[challenge(RamReadWriteChallenge::Gamma)]
        ram: F,
        #[challenge(BooleanityChallenge::Gamma)]
        booleanity: F,
    }

    #[test]
    fn multi_field_disambiguates_by_full_path() {
        let challenges = MultiChallenge {
            ram: fr(1),
            booleanity: fr(2),
        };

        // Same `Gamma` leaf name, different sub-enum → resolves to the right field.
        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(RamReadWriteChallenge::Gamma)),
            Some(fr(1)),
        );
        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(BooleanityChallenge::Gamma)),
            Some(fr(2)),
        );
    }
}

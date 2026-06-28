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
use thiserror::Error;

use crate::protocols::jolt::{JoltChallengeId, JoltOpeningId};

/// A `Challenges` struct could not be built from a drawn-value stream because the
/// stream ran dry before every required (scalar) field was populated. Surfaced by
/// [`SumcheckChallenges::from_transcript_values`]; a relation that draws challenges
/// but does not override `ConcreteSumcheck::draw_challenges` (so the draw-nothing
/// default feeds it an empty stream) is the typical cause.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
#[error("challenge value stream exhausted: only {populated} of {required} required challenge field(s) populated")]
pub struct ChallengeDrawError {
    pub required: usize,
    pub populated: usize,
}

/// Canonical encoders and the output-formula resolver for a relation's
/// *produced* opening-claim struct.
///
/// The implementor's field declaration order is the single definition of
/// canonical opening order: [`opening_values`](Self::opening_values) and
/// [`canonical_order`](Self::canonical_order) derive from it, so the Fiat-Shamir
/// `append_openings` (in `jolt-verifier`) that iterates `opening_values()` cannot
/// disagree.
///
/// Generic over the opening-id type `O` (defaulting to [`JoltOpeningId`]) so the
/// trait can live in the framework half and be reused by other protocol families.
pub trait OutputClaims<F: Field, O = JoltOpeningId> {
    /// Produced opening scalars in canonical (field-declaration) order. Built from
    /// [`canonical_order`](Self::canonical_order) + [`resolve_output`](Self::resolve_output),
    /// both derived from the same fields, so `opening_values()[k]` is exactly
    /// `resolve_output(canonical_order()[k])`. The ids within one `OutputClaims`
    /// struct are distinct, so each canonical-order id resolves to its own value.
    #[expect(
        clippy::expect_used,
        reason = "every canonical_order id is emitted from the same field as resolve_output, so resolution is infallible by construction"
    )]
    fn opening_values(&self) -> Vec<F> {
        self.canonical_order()
            .iter()
            .map(|id| {
                self.resolve_output(id)
                    .expect("every canonical_order id resolves via resolve_output by construction")
            })
            .collect()
    }

    /// The produced opening ids in canonical (field-declaration) order. Mirrors
    /// [`opening_values`](Self::opening_values) one-for-one: same length, and
    /// `canonical_order()[k]` is the id of `opening_values()[k]`. Takes `&self`
    /// because `Vec`/`Option` fields make the length and presence
    /// instance-dependent.
    fn canonical_order(&self) -> Vec<O>;

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
    /// The consumed opening ids in canonical (field-declaration) order. Takes
    /// `&self` because `Vec`/`Option` fields make the length and presence
    /// instance-dependent.
    fn canonical_order(&self) -> Vec<O>;

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
pub trait SumcheckChallenges<F: Field, C = JoltChallengeId>: Sized {
    /// Build this `Challenges` struct from already-drawn Fiat-Shamir scalars,
    /// consuming one value per field in canonical (field-declaration) order.
    ///
    /// This lets a transcript-side caller draw challenges and populate the struct
    /// without restating the field order. The verifier's
    /// `ConcreteSumcheck::draw_challenges` uses it for its draw-nothing default
    /// (passing an empty iterator), so the bound that enables the default is
    /// `SumcheckChallenges` — which every `Challenges` already implements — rather
    /// than `Default`.
    ///
    /// Each scalar field consumes one value; a scalar field with no value left
    /// returns [`ChallengeDrawError`] (a relation that draws challenges must
    /// override the draw, not rely on the empty-stream default). An `Option` field
    /// is conditional: it takes the next value if present and stays `None`
    /// otherwise.
    fn from_transcript_values<I: Iterator<Item = F>>(values: I)
        -> Result<Self, ChallengeDrawError>;

    /// Resolve a drawn Fiat-Shamir challenge by id, for evaluating a relation's
    /// input/output `Expr`. Returns `None` for ids this struct does not carry.
    fn resolve_challenge(&self, id: &C) -> Option<F>;
}

/// `Challenges` for a relation that draws no Fiat-Shamir challenges: resolves
/// every id to `None`. Generic over the field so it fits `type Challenges<F>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoChallenges<F>(::core::marker::PhantomData<F>);

impl<F: Field, C> SumcheckChallenges<F, C> for NoChallenges<F> {
    fn from_transcript_values<I: Iterator<Item = F>>(
        _values: I,
    ) -> Result<Self, ChallengeDrawError> {
        Ok(Self(::core::marker::PhantomData))
    }

    fn resolve_challenge(&self, _id: &C) -> Option<F> {
        None
    }
}

/// Placeholder `Inputs`/`Outputs` for a symbolic-only relation (no concrete
/// resolution): it declares the GATs `SymbolicSumcheck` requires without owning
/// real claim structs. Never resolved (no `ConcreteSumcheck` impl references it).
pub struct NoInputs<C>(::core::marker::PhantomData<C>);
pub struct NoOutputs<C>(::core::marker::PhantomData<C>);

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
#[expect(clippy::unwrap_used)]
mod sumcheck_challenges_tests {
    use crate::protocols::jolt::{BooleanityChallenge, JoltChallengeId, RamReadWriteChallenge};
    // The `SumcheckChallenges` re-export from the crate root covers both the trait
    // (type namespace) and the derive macro (macro namespace).
    use crate::{ChallengeDrawError, SumcheckChallenges};
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

    #[test]
    fn from_transcript_values_fills_fields_in_declaration_order() {
        // Two values populate `ram` then `booleanity` (declaration order); extra
        // stream values are ignored.
        let challenges: MultiChallenge<Fr> =
            MultiChallenge::from_transcript_values([fr(1), fr(2), fr(3)].into_iter()).unwrap();
        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(RamReadWriteChallenge::Gamma)),
            Some(fr(1)),
        );
        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(BooleanityChallenge::Gamma)),
            Some(fr(2)),
        );
    }

    #[test]
    fn from_transcript_values_errors_when_stream_runs_dry() {
        // One value cannot fill the two scalar fields; the error reports progress.
        let result = MultiChallenge::<Fr>::from_transcript_values([fr(1)].into_iter());
        assert_eq!(
            result.err(),
            Some(ChallengeDrawError {
                required: 2,
                populated: 1,
            }),
        );
    }
}

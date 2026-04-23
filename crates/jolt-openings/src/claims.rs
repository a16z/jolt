//! Stateless claim types for PCS operations.

use jolt_field::Field;
use jolt_poly::Polynomial;
use jolt_transcript::AppendToTranscript;

use crate::backend::CommitmentBackend;
use crate::schemes::CommitmentScheme;

/// Prover-side opening claim: polynomial, evaluation point, and claimed value.
///
/// TODO(prover-claim): Consider giving this struct a `commitment: PCS::Output`
/// field for symmetry with [`OpeningClaim`], so packed-commitment schemes
/// (e.g. Hachi mega-poly) can recover slot/group membership on the prover
/// side. Today the orchestrator threads the per-poly hint via a parallel
/// `Vec<PCS::OpeningHint>` and groups by point inside the helper, which
/// is enough for unpacked schemes (Mock/HyperKZG/Dory). When Hachi lands
/// we'll either: (a) introduce a `PackedProverClaim<PCS>` carrying the
/// commitment + slot, or (b) generalise `OpeningHint` to include
/// per-batch context. Defer until Hachi is wired.
#[derive(Clone, Debug)]
pub struct ProverClaim<F: Field> {
    pub polynomial: Polynomial<F>,
    pub point: Vec<F>,
    pub eval: F,
}

/// Verifier-side opening claim: commitment, point, and claimed value.
///
/// Pre-backend variant kept for downstream consumers (e.g. the
/// hash-jolt follow-up) and as a reference shape. The backend-aware
/// verifier (`verify_with_backend`) uses [`OpeningClaim`] instead.
#[derive(Clone, Debug)]
pub struct VerifierClaim<F: Field, C> {
    pub commitment: C,
    pub point: Vec<F>,
    pub eval: F,
}

/// Backend-aware opening claim: `(commitment_handle, point, eval)`,
/// where commitment / point / eval all live in a [`CommitmentBackend`]'s
/// associated handle space.
///
/// Replaces the prior `BackendVerifierClaim<B, PCS>` tuple alias; making
/// it a named struct lets us add fields later (e.g. `slot: u32` for
/// packed-commitment PCSes) without breaking call sites.
///
/// Claims are passed in the order they were accumulated by the
/// orchestrator. PCS impls that pack multiple polynomials under one
/// commitment (e.g. Hachi mega-poly) MAY treat that order as a slot
/// index: the i-th claim with `commitment == C` is the i-th polynomial
/// under `C`. Unpacked schemes (Mock/HyperKZG/Dory) treat order as
/// immaterial — they only group by `point`.
#[derive(Clone, Debug)]
pub struct OpeningClaim<B, PCS>
where
    B: CommitmentBackend<PCS>,
    PCS: CommitmentScheme<Field = B::F>,
    PCS::Output: AppendToTranscript,
{
    pub commitment: B::Commitment,
    pub point: Vec<B::Scalar>,
    pub eval: B::Scalar,
}

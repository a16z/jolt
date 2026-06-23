//! Prover-side commitment-scheme trait surface for Akita protocol code.

use crate::compute::{CommitmentComputeBackend, ProverComputeBackend};
use crate::ProverTranscriptGrind;
use crate::{AkitaPolyOps, ProverClaims};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;
use akita_types::{BasisMode, SetupContributionMode};

/// Prover-side commitment-scheme interface used by Akita protocol code.
///
/// Generic over base field `F` and cyclotomic ring degree `D`.
/// Caller-provided root polynomials are provided as `impl AkitaPolyOps<F, D>`.
/// Recursive `w` witnesses are internal to the protocol and no longer modelled
/// through this trait.
pub trait CommitmentProver<F, const D: usize>
where
    F: FieldCore + CanonicalField,
{
    /// Prover setup parameters.
    type ProverSetup: Clone + Send + Sync;
    /// Verifier setup derived from prover setup.
    type VerifierSetup: Clone + Send + Sync;
    /// Commitment object produced by the scheme.
    type Commitment: Clone + Send + Sync;
    /// Public opening point, claimed-evaluation, and proof scalar field.
    type ExtField: ExtField<F>;
    /// Prover-side hint produced for one opening-point commitment.
    type CommitHint: Clone + Send + Sync;
    /// Batched proof object produced by the scheme.
    type BatchedProof: Clone + Send + Sync;
    /// Build prover setup for maximum polynomial dimension, batch capacity,
    /// and distinct opening-point count.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested capacity, field tower, or generated
    /// setup is invalid.
    fn setup_prover(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<Self::ProverSetup, AkitaError>;

    /// Build prover setup for recursive setup-contribution mode.
    ///
    /// # Errors
    ///
    /// Returns an error if base setup construction or recursive setup-prefix
    /// population fails.
    fn setup_prover_recursion(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<Self::ProverSetup, AkitaError>;

    /// Derive verifier setup from prover setup.
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commit a single opening-point bundle.
    ///
    /// All polynomials in `polys` are aggregated into one commitment using a
    /// layout derived from the single shared opening-batch shape.
    ///
    /// # Errors
    ///
    /// Returns an error when setup/parameter constraints are not satisfied.
    fn commit<P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys: &[P],
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        P: AkitaPolyOps<F, D>,
        B: CommitmentComputeBackend<F>;

    /// Commit the polynomial bundles used by a batched prove.
    ///
    /// Each input bundle produces one commitment. All bundles share one public
    /// opening point in the subsequent [`Self::batched_prove`] call.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation, layout selection, or any
    /// per-point commitment fails.
    #[allow(clippy::type_complexity)]
    fn batched_commit<P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys_per_commitment_group: &[&[P]],
    ) -> Result<Vec<(Self::Commitment, Self::CommitHint)>, AkitaError>
    where
        P: AkitaPolyOps<F, D>,
        B: CommitmentComputeBackend<F>;

    /// Produce a fused batched opening proof for one shared opening point.
    ///
    /// A singleton opening is the 1x1 special case (one polynomial, one
    /// commitment, one opening point).
    ///
    /// # Errors
    ///
    /// Returns an error if any opening point is invalid or proof generation
    /// fails.
    #[allow(clippy::too_many_arguments)]
    fn batched_prove<'a, T, P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        claims: ProverClaims<'a, Self::ExtField, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F> + ProverTranscriptGrind<F>,
        P: AkitaPolyOps<F, D>,
        B: ProverComputeBackend<F>;
}

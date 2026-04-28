//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! - [`CommitmentScheme`] — commit, open, verify for multilinear polynomials.
//! - [`AdditivelyHomomorphic`] — linear combination of commitments.
//! - [`StreamingCommitment`] — chunked commitment without full materialization.
//! - [`ZkOpeningScheme`] — zero-knowledge opening proofs with hiding commitments.

use std::fmt::Debug;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::error::OpeningsError;

/// Commit to f: F^n -> F, then prove f(r) = v for verifier-chosen r.
pub trait CommitmentScheme: Commitment + Clone + Send + Sync + 'static {
    type Field: Field;
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type ProverSetup: Clone + Send + Sync;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    type Polynomial: MultilinearPoly<Self::Field> + From<Vec<Self::Field>>;

    /// Auxiliary data from commit reused during opening (e.g. Dory row commitments).
    type OpeningHint: Clone + Send + Sync + Default;

    type SetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup);

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup;

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof;

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Post-proof transcript binding: absorb the opening point and evaluation
    /// into the Fiat-Shamir transcript. Called after `open`/`verify` to bind
    /// the opening inputs for subsequent protocol stages.
    ///
    /// Default: no-op (PCS that don't require post-proof binding).
    fn bind_opening_inputs(
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
        _point: &[Self::Field],
        _eval: &Self::Field,
    ) {
    }
}

/// C = Σ s_i · C_i.
pub trait AdditivelyHomomorphic: CommitmentScheme
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;

    fn combine_hints(
        _hints: Vec<Self::OpeningHint>,
        _scalars: &[Self::Field],
    ) -> Self::OpeningHint {
        Self::OpeningHint::default()
    }
}

/// Incremental commitment without full materialization.
pub trait StreamingCommitment: CommitmentScheme {
    type PartialCommitment: Clone + Send + Sync;

    fn begin(setup: &Self::ProverSetup) -> Self::PartialCommitment;

    fn feed(
        partial: &mut Self::PartialCommitment,
        chunk: &[Self::Field],
        setup: &Self::ProverSetup,
    );

    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output;
}

/// Opening proofs that hide the evaluation behind a commitment.
pub trait ZkOpeningScheme: CommitmentScheme {
    type HidingCommitment: Clone
        + Debug
        + Eq
        + Send
        + Sync
        + 'static
        + Serialize
        + DeserializeOwned
        + AppendToTranscript;

    type Blind: Clone + Send + Sync;

    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind);

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;
}

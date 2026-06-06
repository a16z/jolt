//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! - [`CommitmentScheme`] — commit, open, verify for multilinear polynomials.
//! - [`AdditivelyHomomorphic`] — linear combination of commitments.
//! - [`StreamingCommitment`] — chunked commitment without full materialization.
//! - [`ZkOpeningScheme`] — zero-knowledge commitments and opening proofs.
//! - [`ZkStreamingCommitment`] — chunked zero-knowledge commitments.

use std::fmt::Debug;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::{Field, FromPrimitiveInt};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::error::OpeningsError;

/// Commit to f: F^n -> F, then prove f(r) = v for verifier-chosen r.
pub trait CommitmentScheme: Commitment {
    type Field: Field;
    type Proof: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
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

    fn open_poly<P: MultilinearPoly<Self::Field>>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let mut evals = Vec::with_capacity(1usize << poly.num_vars());
        poly.for_each_row(poly.num_vars(), &mut |_, row| evals.extend_from_slice(row));
        let dense = Self::Polynomial::from(evals);
        Self::open(&dense, point, eval, setup, hint, transcript)
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    );
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
    type OneHotChunkCommitment: Clone + Send + Sync;
    type OneHotStreamContext: Send + Sync;

    fn begin(setup: &Self::ProverSetup) -> Self::PartialCommitment;

    fn feed(
        partial: &mut Self::PartialCommitment,
        chunk: &[Self::Field],
        setup: &Self::ProverSetup,
    );

    fn feed_zeros(
        partial: &mut Self::PartialCommitment,
        row_width: usize,
        rows: usize,
        setup: &Self::ProverSetup,
    ) {
        if rows == 0 {
            return;
        }
        let row = vec![Self::Field::from_u64(0); row_width];
        for _ in 0..rows {
            Self::feed(partial, &row, setup);
        }
    }

    fn feed_u64(partial: &mut Self::PartialCommitment, chunk: &[u64], setup: &Self::ProverSetup) {
        let values: Vec<Self::Field> = chunk
            .iter()
            .copied()
            .map(<Self::Field as FromPrimitiveInt>::from_u64)
            .collect();
        Self::feed(partial, &values, setup);
    }

    fn feed_i128(partial: &mut Self::PartialCommitment, chunk: &[i128], setup: &Self::ProverSetup) {
        let values: Vec<Self::Field> = chunk
            .iter()
            .copied()
            .map(<Self::Field as FromPrimitiveInt>::from_i128)
            .collect();
        Self::feed(partial, &values, setup);
    }

    fn begin_one_hot_column_major_stream(
        setup: &Self::ProverSetup,
        row_width: usize,
    ) -> Self::OneHotStreamContext;

    fn process_one_hot_chunk(
        context: &mut Self::OneHotStreamContext,
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::OneHotChunkCommitment;

    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output;

    fn finish_with_hint(
        partial: Self::PartialCommitment,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        (Self::finish(partial, setup), Self::OpeningHint::default())
    }

    fn finish_one_hot_column_major_chunks(
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunks: &[Self::OneHotChunkCommitment],
    ) -> (Self::Output, Self::OpeningHint);
}

/// Incremental commitment support for schemes whose hiding/ZK commitment mode
/// is distinct from the transparent streaming path.
pub trait ZkStreamingCommitment: StreamingCommitment + ZkOpeningScheme {
    fn finish_zk_with_hint(
        partial: Self::PartialCommitment,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    fn finish_zk_one_hot_column_major_chunks(
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunks: &[Self::OneHotChunkCommitment],
    ) -> (Self::Output, Self::OpeningHint);
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

    /// Commit in the scheme's ZK/hiding mode.
    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    /// Open a ZK/hiding commitment using the opening hint returned by
    /// [`commit_zk`](Self::commit_zk).
    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind);

    fn open_zk_poly<P: MultilinearPoly<Self::Field>>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let mut evals = Vec::with_capacity(1usize << poly.num_vars());
        poly.for_each_row(poly.num_vars(), &mut |_, row| evals.extend_from_slice(row));
        let dense = Self::Polynomial::from(evals);
        Self::open_zk(&dense, point, eval, setup, hint, transcript)
    }

    /// Verify a ZK opening proof and return the hiding commitment to the
    /// evaluation that the proof binds internally.
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError>;

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    );
}

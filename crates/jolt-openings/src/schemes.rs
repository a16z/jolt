//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! - [`CommitmentScheme`] — commit, open, verify for multilinear polynomials.
//! - [`AdditivelyHomomorphic`] — linear combination of commitments.
//! - [`StreamingCommitment`] — chunked commitment without full materialization.
//! - [`ZkOpeningScheme`] — zero-knowledge commitments and opening proofs.

use std::fmt::Debug;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::error::OpeningsError;

/// Optional metadata exposed by commitment backends that bind a layout digest
/// into the commitment.
pub trait CommitmentLayoutDigest {
    fn layout_digest(&self) -> Option<[u8; 32]>;
}

impl CommitmentLayoutDigest for u64 {
    fn layout_digest(&self) -> Option<[u8; 32]> {
        None
    }
}

/// Verifier statement for a same-point batch opening.
///
/// `logical_point` is the protocol point where the listed claims originated.
/// This point is transcript-bound even when the physical PCS proof opens a
/// different point.
///
/// `pcs_point` is the point used by direct/native batch openings. It is usually
/// equal to `logical_point`, but may differ when a protocol embeds a logical
/// claim into a different PCS coordinate system. Packed reductions derive their
/// final native PCS point from the packing reduction proof; for those statements
/// `pcs_point` is retained as statement metadata rather than the final opened
/// point.
///
/// `layout_digest` domain-separates the physical statement. Direct statements
/// should use the commitment's own layout digest when the backend exposes one;
/// packed statements should use the canonical packing layout digest that also
/// appears on every packed physical view. The digest binds metadata only: the
/// packed-view relation is proven by the packing reduction, not by this field
/// alone.
///
/// Each claim carries a `scale`. The batch reduction samples RLC coefficients
/// from the transcript and uses `gamma_i * scale_i` as the effective logical
/// coefficient for claim `i`; packed views then multiply that value by each
/// view term coefficient. In other words, `claim` is checked after applying
/// `scale`, while packed term coefficients describe how that scaled logical
/// claim is read from the packed witness.
///
/// Direct/native batch statements may contain claims against different
/// commitments when the backend supports that reduction. A packed-linear
/// statement is stricter: all claims must be `PhysicalView::PackedLinear`, all
/// packed view digests must equal `layout_digest`, and all claims must refer to
/// one packed witness commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOpeningStatement<F, C, OpeningId = (), RelationId = (), Claim = F> {
    /// Protocol point for the logical claims and transcript binding.
    pub logical_point: Vec<F>,
    /// Direct/native PCS opening point. Packed reductions derive their native
    /// opening point separately.
    pub pcs_point: Vec<F>,
    /// Digest for the physical layout used by this statement.
    pub layout_digest: [u8; 32],
    pub claims: Vec<BatchOpeningClaim<F, C, OpeningId, RelationId, Claim>>,
}

/// One logical opening claim inside a same-point batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOpeningClaim<F, C, OpeningId = (), RelationId = (), Claim = F> {
    pub id: OpeningId,
    pub relation: RelationId,
    pub commitment: C,
    pub claim: Claim,
    pub view: PhysicalView<F>,
    /// Multiplier applied before batching. The effective RLC coefficient is
    /// `gamma_i * scale`, and packed view terms multiply this value by their
    /// own coefficients.
    pub scale: F,
}

/// Physical commitment view used to satisfy a logical opening claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalView<F> {
    Direct,
    PackedLinear {
        layout_digest: [u8; 32],
        terms: Vec<PackedLinearTerm<F>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLinearTerm<F> {
    pub coefficient: F,
    pub family: PackedFamilyRef,
    pub limb: usize,
    pub symbol: usize,
    pub row_point: Vec<F>,
}

impl<F> PackedLinearTerm<F> {
    pub fn new(coefficient: F, family: PackedFamilyRef, limb: usize, symbol: usize) -> Self {
        Self {
            coefficient,
            family,
            limb,
            symbol,
            row_point: Vec::new(),
        }
    }

    pub fn with_row_point(mut self, row_point: Vec<F>) -> Self {
        self.row_point = row_point;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackedFamilyRef {
    pub namespace: u64,
    pub id: u64,
    pub index: u64,
}

impl PackedFamilyRef {
    pub const fn new(namespace: u64, id: u64, index: u64) -> Self {
        Self {
            namespace,
            id,
            index,
        }
    }
}

/// PCS-specific reduction data produced by verifying a same-point batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOpeningResult<F, C, R = F> {
    pub coefficients: Vec<F>,
    pub joint_commitment: C,
    pub reduced_opening: R,
}

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

/// Same-point batch opening extension for a PCS.
pub trait BatchOpeningScheme: CommitmentScheme {
    fn prove_batch<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<Self::Proof, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;

    fn verify_batch<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}

/// ZK same-point batch opening extension mirroring [`ZkOpeningScheme`].
pub trait ZkBatchOpeningScheme: BatchOpeningScheme + ZkOpeningScheme {
    #[expect(
        clippy::type_complexity,
        reason = "ZK batch openings mirror ZkOpeningScheme's proof, hiding commitment, and blind tuple"
    )]
    fn prove_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        evals: &[Self::Field],
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;

    #[expect(
        clippy::type_complexity,
        reason = "ZK batch verification returns the same opening result shape with a hiding commitment"
    )]
    fn verify_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
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

#[cfg(test)]
#[path = "schemes_tests.rs"]
mod tests;

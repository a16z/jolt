//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! The base verifier/prover traits expose single openings, ordinary batch
//! openings, and source-based commitment. Extension traits add homomorphic and
//! ZK operations only for schemes and protocols that need them.

use std::fmt::Debug;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::claims::{
    BatchOpeningProverResult, BatchOpeningPublic, BatchOutputExpression, BatchOutputRelation,
    BatchOutputValue, OpenedBatchOutput, OpeningClaim, ProverBatchOpeningTerm, ProverClaim,
    VerifierBatchOpeningTerm, ZkBatchOpeningProverResult,
};
use crate::error::OpeningsError;
use crate::sources::{
    BatchCommitmentSource, CommitmentSource, LinearCombinationOpeningSource, SourceId,
};

/// Verifier-side interface for a polynomial commitment scheme.
pub trait CommitmentSchemeVerifier: Commitment + Clone + Send + Sync + 'static {
    type Field: Field;
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type BatchProof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Verifies one opening proof.
    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Verifies an ordinary batch-opening proof.
    fn verify_batch(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Binds one transparent opening input to the Fiat-Shamir transcript.
    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    );
}

/// Verifier setup derivable from public parameters without prover setup.
///
/// Transparent schemes such as Dory can build verifier setup from a size
/// parameter alone. Structured-reference-string schemes such as KZG generally
/// cannot: their verifier setup contains trapdoor-derived elements generated
/// during setup, so verifier-only code should receive the verifier setup as an
/// input rather than pretending it can derive it from public generators.
///
/// This is separate from [`CommitmentSchemeVerifier`] so verifier code can stay
/// generic over schemes that require a supplied verifier key.
pub trait VerifierSetupFromPublicParams: CommitmentSchemeVerifier {
    type PublicParams;

    /// Builds verifier setup directly from public parameters.
    fn verifier_setup_from_public_params(params: Self::PublicParams) -> Self::VerifierSetup;
}

/// Prover-side interface for a polynomial commitment scheme.
pub trait CommitmentScheme: CommitmentSchemeVerifier {
    type ProverSetup: Clone + Send + Sync;
    type OpeningHint: Clone + Send + Sync + Default;
    type SetupParams;

    /// Builds prover and verifier setup.
    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup);

    /// Derives verifier setup from prover setup.
    fn prover_to_verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commits to one source.
    fn commit<S: CommitmentSource<Self::Field> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    /// Commits to a batch of sources.
    fn commit_batch<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        ids.iter()
            .map(|&id| {
                let source = batch.source(id);
                Self::commit(&source, setup)
            })
            .collect()
    }

    /// Proves one opening.
    fn open<S>(
        polynomial: &S,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof
    where
        S: CommitmentSource<Self::Field> + ?Sized;

    /// Proves an ordinary batch opening.
    fn prove_batch<S>(
        claims: Vec<ProverClaim<Self::Field, S>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::BatchProof
    where
        S: CommitmentSource<Self::Field>;
}

/// Verifier-side interface for linear source-backed batch openings.
pub trait LinearOpeningSchemeVerifier: CommitmentSchemeVerifier {
    /// Verifies a linear source-backed batch opening and returns the PCS output
    /// relation created by that verification.
    ///
    /// The default implementation verifies the raw terms through ordinary
    /// [`CommitmentSchemeVerifier::verify_batch`] and exposes one public output
    /// per raw term. Schemes with native linear-source fusion should override
    /// this method so the PCS owns its batching challenge and output relation.
    fn verify_batch_opening<ClaimId, SourceIdT>(
        terms: Vec<VerifierBatchOpeningTerm<Self::Field, Self, ClaimId, SourceIdT>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<BatchOpeningPublic<Self::Field, (), ClaimId>, OpeningsError>
    where
        Self: Sized,
        SourceIdT: SourceId,
    {
        let claims = terms
            .iter()
            .map(|term| OpeningClaim {
                commitment: term.commitment.clone(),
                point: term.point.proof.clone(),
                eval: term.eval,
            })
            .collect();
        Self::verify_batch(claims, proof, setup, transcript)?;

        let public = transparent_public_from_terms(
            terms
                .into_iter()
                .map(|term| (term.claim_id, term.point.public, term.eval, term.eval_scale)),
        );
        bind_transparent_batch_outputs::<Self, ClaimId>(&public, transcript);
        Ok(public)
    }
}

/// Prover-side interface for linear source-backed batch openings.
pub trait LinearOpeningScheme: CommitmentScheme + LinearOpeningSchemeVerifier {
    /// Proves a linear source-backed batch opening and returns the PCS output
    /// relation.
    ///
    /// The default implementation routes each raw term through ordinary
    /// [`CommitmentScheme::prove_batch`]. Production backends that can preserve
    /// streaming fusion should override this method.
    fn prove_batch_opening<B, ClaimId>(
        terms: Vec<ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>>,
        source_batch: &mut B,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> BatchOpeningProverResult<Self, ClaimId>
    where
        Self: Sized,
        B: LinearCombinationOpeningSource<Self::Field, Self::OpeningHint>,
    {
        let claims = terms
            .iter()
            .map(|term| ProverClaim {
                polynomial: source_batch.source(term.source_id),
                point: term.point.proof.clone(),
                eval: term.eval,
            })
            .collect();
        let hints = terms
            .iter()
            .map(|term| source_batch.opening_hint(term.source_id).clone())
            .collect();
        let proof = Self::prove_batch(claims, hints, setup, transcript);

        let public = transparent_public_from_terms(
            terms
                .into_iter()
                .map(|term| (term.claim_id, term.point.public, term.eval, term.eval_scale)),
        );
        bind_transparent_batch_outputs::<Self, ClaimId>(&public, transcript);
        BatchOpeningProverResult { proof, public }
    }
}

/// Verifier-side additive combination of commitments.
pub trait AdditivelyHomomorphicVerifier: CommitmentSchemeVerifier
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    /// Computes `Σ scalars[i] * commitments[i]`.
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;
}

/// Prover-side additive combination of commitment hints.
pub trait AdditivelyHomomorphic: CommitmentScheme + AdditivelyHomomorphicVerifier
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    /// Computes the hint corresponding to the same linear combination as
    /// [`AdditivelyHomomorphicVerifier::combine`].
    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint;
}

/// Verifier-side interface for openings that hide evaluations.
pub trait ZkOpeningSchemeVerifier: CommitmentSchemeVerifier {
    type HidingCommitment: Clone
        + Debug
        + Eq
        + Send
        + Sync
        + 'static
        + Serialize
        + DeserializeOwned
        + AppendToTranscript;

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Verifies a ZK batch-opening proof.
    fn verify_batch_zk(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Binds one ZK opening input to the Fiat-Shamir transcript.
    ///
    /// The evaluation is hidden, so the transcript receives the opening point
    /// and the hiding commitment to the evaluation instead of the scalar value.
    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    );
}

/// Prover-side interface for openings that hide evaluations.
pub trait ZkOpeningScheme: CommitmentScheme + ZkOpeningSchemeVerifier {
    type Blind: Clone + Send + Sync;

    /// Commits in the scheme's ZK/hiding mode.
    fn commit_zk<S: CommitmentSource<Self::Field> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    /// Commits to a batch of sources in the scheme's ZK/hiding mode.
    fn commit_batch_zk<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        ids.iter()
            .map(|&id| {
                let source = batch.source(id);
                Self::commit_zk(&source, setup)
            })
            .collect()
    }

    /// Opens a ZK/hiding commitment using the hint returned by
    /// [`commit_zk`](Self::commit_zk).
    fn open_zk<S>(
        polynomial: &S,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field> + ?Sized;

    /// Proves a ZK batch opening.
    fn prove_batch_zk<S>(
        claims: Vec<ProverClaim<Self::Field, S>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::BatchProof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field>;
}

/// Verifier-side interface for ZK linear source-backed batch openings.
pub trait ZkLinearOpeningSchemeVerifier: ZkOpeningSchemeVerifier {
    /// Verifies a ZK linear source-backed batch opening and returns the public
    /// output relation produced by the PCS.
    fn verify_batch_opening_zk<ClaimId, SourceIdT>(
        terms: Vec<VerifierBatchOpeningTerm<Self::Field, Self, ClaimId, SourceIdT>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<BatchOpeningPublic<Self::Field, Self::HidingCommitment, ClaimId>, OpeningsError>
    where
        Self: Sized,
        SourceIdT: SourceId;
}

/// Prover-side interface for ZK linear source-backed batch openings.
pub trait ZkLinearOpeningScheme: ZkOpeningScheme + ZkLinearOpeningSchemeVerifier {
    /// Proves a ZK linear source-backed batch opening and returns both public
    /// output metadata and prover-only hidden-output witnesses.
    fn prove_batch_opening_zk<B, ClaimId>(
        terms: Vec<ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>>,
        source_batch: &mut B,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> ZkBatchOpeningProverResult<Self, ClaimId>
    where
        Self: Sized,
        B: LinearCombinationOpeningSource<Self::Field, Self::OpeningHint>;
}

fn transparent_public_from_terms<F, ClaimId, I>(terms: I) -> BatchOpeningPublic<F, (), ClaimId>
where
    F: Field,
    I: IntoIterator<Item = (ClaimId, Vec<F>, F, F)>,
{
    let mut outputs = Vec::new();
    let mut relations = Vec::new();

    for (claim_id, point, eval, eval_scale) in terms {
        let output_index = outputs.len();
        outputs.push(OpenedBatchOutput {
            point,
            value: BatchOutputValue::Public(eval * eval_scale),
        });
        relations.push(BatchOutputRelation {
            output_index,
            expression: BatchOutputExpression::Linear(vec![(claim_id, eval_scale)]),
        });
    }

    BatchOpeningPublic { outputs, relations }
}

fn bind_transparent_batch_outputs<PCS, ClaimId>(
    public: &BatchOpeningPublic<PCS::Field, (), ClaimId>,
    transcript: &mut impl Transcript<Challenge = PCS::Field>,
) where
    PCS: CommitmentSchemeVerifier,
{
    for output in &public.outputs {
        if let BatchOutputValue::Public(eval) = &output.value {
            PCS::bind_opening_inputs(transcript, &output.point, eval);
        }
    }
}

/// Verifier-side hooks for schemes whose ZK openings bind a hidden evaluation.
///
/// Jolt's BlindFold integration needs to absorb the commitment to the hidden
/// evaluation and use the same commitment generators inside its verifier R1CS.
/// Schemes without this Dory-style evaluation commitment should not implement
/// this extension trait.
pub trait EvaluationCommitmentScheme<G>: ZkOpeningSchemeVerifier
where
    G: Clone + Send + Sync + 'static,
{
    /// Extracts the hidden evaluation commitment from a batch proof.
    fn batch_eval_commitment(proof: &Self::BatchProof) -> Option<G>;

    /// Returns the verifier-side generators used by the hidden evaluation
    /// commitment relation.
    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(G, G)>;
}

/// Prover-side hooks for schemes whose ZK openings bind a hidden evaluation.
pub trait EvaluationCommitmentProver<G>: EvaluationCommitmentScheme<G> + ZkOpeningScheme
where
    G: Clone + Send + Sync + 'static,
{
    /// Returns the prover-side generators used by the hidden evaluation
    /// commitment relation.
    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(G, G)>;

    /// Returns Pedersen generators derived from the PCS setup for BlindFold.
    fn zk_generators(setup: &Self::ProverSetup, count: usize) -> Option<(Vec<G>, G)>;
}

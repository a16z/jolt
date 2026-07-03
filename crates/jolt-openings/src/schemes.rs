//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! - [`CommitmentScheme`] — commit, open, verify for multilinear polynomials.
//! - [`AdditivelyHomomorphic`] — linear combination of commitments.
//! - [`StreamingCommitment`] — chunked commitment without full materialization.
//! - [`ZkOpeningScheme`] — zero-knowledge commitments and opening proofs.
//! - [`ZkStreamingCommitment`] — chunked zero-knowledge commitments.

use std::{fmt::Debug, marker::PhantomData};

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::{Field, FromPrimitiveInt};
use jolt_poly::{MultilinearPoly, Point, RlcSource, HIGH_TO_LOW};
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::claims::{EvaluationClaim, VerifierOpeningClaim, VerifierRlcClaims, ZkEvaluationClaim};
use crate::error::OpeningsError;
use crate::packing::{
    prove_reduction_sumcheck, verify_reduction_sumcheck, PackedBatchProof, PrefixPackedProverSetup,
    PrefixPackedStatement, PrefixPackedVerifierSetup,
};

/// Commit to f: F^n -> F, then prove f(r) = v for verifier-chosen r.
pub trait CommitmentScheme: Commitment {
    type Field: Field;
    type Proof: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
    type ProverSetup: Clone + Send + Sync;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Auxiliary data from commit reused during opening (e.g. Dory row commitments).
    type OpeningHint: Clone + Send + Sync + Default;

    type SetupParams;

    fn setup(
        params: Self::SetupParams,
    ) -> Result<(Self::ProverSetup, Self::VerifierSetup), OpeningsError>;

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup;

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError>;

    fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError>;

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;
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

/// Incremental commitment without full materialization, plus the one-hot
/// column-major and primitive-typed feed paths used by the streaming prover.
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

    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output;

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

/// Opening proofs that hide the evaluation behind a commitment.
///
/// Two commitments with different subjects are in play. The inherited
/// [`Commitment::Output`](jolt_crypto::Commitment) commits to the
/// *polynomial* — it is what the verifier already holds when an opening
/// starts. [`HidingCommitment`](Self::HidingCommitment) commits to the
/// *evaluation value* at the opening point (e.g. Dory's `y_com`): instead of
/// revealing the cleartext evaluation, the opening proof binds it inside this
/// commitment. That is why [`verify_zk`](Self::verify_zk) takes no `eval`
/// argument and returns the hiding commitment — downstream protocol layers
/// (e.g. BlindFold) verify relations over the still-hidden evaluation.
pub trait ZkOpeningScheme: CommitmentScheme {
    /// Commitment to the evaluation value that an opening proof binds
    /// internally, in contrast to `Output`, which commits to the polynomial.
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
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError>;

    /// Open a ZK/hiding commitment using the opening hint returned by
    /// [`commit_zk`](Self::commit_zk).
    #[expect(
        clippy::type_complexity,
        reason = "ZK openings return the native proof, hiding commitment, and blind"
    )]
    fn open_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>;

    /// Verify a ZK opening proof and return the hiding commitment to the
    /// evaluation that the proof binds internally.
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError>;
}

/// Hiding/ZK finish paths for streaming commitments whose ZK commitment mode
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

/// Batch opening protocol selected by a small marker adapter.
///
/// The prover-side inputs are deliberately split into three parameters with
/// distinct roles:
///
/// - [`Statement`](Self::Statement) is the public input both sides agree on
///   and bind to the transcript: the opening claims plus the commitments they
///   refer to. Its shape is scheme-specific — [`HomomorphicBatch`] carries
///   one commitment per claim, [`PackedBatch`] one packed commitment for all
///   claims.
/// - [`Polynomials`](Self::Polynomials) are the borrowed prover-side
///   polynomial sources backing the statement; the verifier never sees them.
/// - [`Hints`](Self::Hints) are the commit-time auxiliary data
///   ([`CommitmentScheme::OpeningHint`]) the PCS reuses when opening.
pub trait BatchOpeningScheme {
    type Field: Field;
    type ProverSetup;
    type VerifierSetup;
    /// Public opening claims plus the commitments they refer to.
    type Statement;
    /// Borrowed prover-side polynomial sources backing the statement.
    type Polynomials<'a>
    where
        Self: 'a;
    /// Commit-time auxiliary data reused by the PCS when opening.
    type Hints;
    type Proof;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        polynomials: Self::Polynomials<'a>,
        hints: Self::Hints,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>;

    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        statement: Self::Statement,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}

/// ZK same-point batching for schemes with native hiding openings.
///
/// The cleartext evaluations are not part of the public statement — the
/// proof binds them behind a hiding commitment — so the prover supplies them
/// as a separate `evaluations` argument alongside the polynomial sources and
/// hints.
pub trait ZkBatchOpeningScheme: BatchOpeningScheme {
    type Commitment;
    type HidingCommitment;
    type Blind;

    #[expect(
        clippy::type_complexity,
        reason = "ZK batch openings return the native proof, hiding commitment, and blind"
    )]
    fn prove_batch_zk<'a, T>(
        setup: &Self::ProverSetup,
        point: Point<HIGH_TO_LOW, Self::Field>,
        commitments: Vec<Self::Commitment>,
        polynomials: Self::Polynomials<'a>,
        hints: Self::Hints,
        evaluations: Vec<Self::Field>,
        transcript: &mut T,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>;

    fn verify_batch_zk<T>(
        setup: &Self::VerifierSetup,
        point: Point<HIGH_TO_LOW, Self::Field>,
        commitments: Vec<Self::Commitment>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<Self::HidingCommitment, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}

// Batching strategies are zero-sized marker types rather than impls on the
// PCS itself: one PCS can support several strategies (a homomorphic scheme
// can also be prefix-packed), and Rust cannot disambiguate two blanket
// `impl BatchOpeningScheme for PCS` impls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HomomorphicBatch<PCS>(PhantomData<PCS>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedBatch<PCS, Id = u64>(PhantomData<(PCS, Id)>);

type HomomorphicRlcSource<'a, F> = RlcSource<F, &'a (dyn MultilinearPoly<F> + 'a)>;

impl<PCS> BatchOpeningScheme for HomomorphicBatch<PCS>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
{
    type Field = PCS::Field;
    type ProverSetup = PCS::ProverSetup;
    type VerifierSetup = PCS::VerifierSetup;
    type Statement = Vec<VerifierOpeningClaim<PCS::Field, PCS::Output>>;
    type Polynomials<'a>
        = Vec<&'a (dyn MultilinearPoly<PCS::Field> + 'a)>
    where
        Self: 'a;
    type Hints = Vec<PCS::OpeningHint>;
    type Proof = PCS::Proof;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        claims: Self::Statement,
        polynomials: Self::Polynomials<'a>,
        hints: Self::Hints,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        let statement = HomomorphicBatchStatement::new(&claims)?;
        if polynomials.len() != statement.claims.len() || hints.len() != statement.claims.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "{} polynomials and {} hints do not match claim count {}",
                polynomials.len(),
                hints.len(),
                statement.claims.len()
            )));
        }

        statement.append_to_transcript(transcript);
        let scalars = transcript.challenge_scalar_powers(statement.claims.len());
        let joint_eval = statement.joint_eval(&scalars);
        let joint_polynomial =
            Self::combine_polynomials(polynomials, &scalars, statement.point.len())?;
        let combined_hint = PCS::combine_hints(hints, &scalars);
        let proof = PCS::open(
            &joint_polynomial,
            statement.point.as_slice(),
            joint_eval,
            setup,
            Some(combined_hint),
            transcript,
        )?;
        EvaluationClaim::new(statement.point.clone(), joint_eval).append_to_transcript(transcript);
        Ok(proof)
    }

    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        claims: Self::Statement,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let statement = HomomorphicBatchStatement::new(&claims)?;
        statement.append_to_transcript(transcript);
        let scalars = transcript.challenge_scalar_powers(statement.claims.len());
        let joint_eval = statement.joint_eval(&scalars);
        let commitments = statement.commitments();
        let joint_commitment = PCS::combine(&commitments, &scalars);
        PCS::verify(
            &joint_commitment,
            statement.point.as_slice(),
            joint_eval,
            proof,
            setup,
            transcript,
        )?;
        EvaluationClaim::new(statement.point.clone(), joint_eval).append_to_transcript(transcript);
        Ok(())
    }
}

impl<PCS> HomomorphicBatch<PCS>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
{
    fn combine_polynomials<'a>(
        polynomials: Vec<&'a (dyn MultilinearPoly<PCS::Field> + 'a)>,
        scalars: &[PCS::Field],
        point_len: usize,
    ) -> Result<HomomorphicRlcSource<'a, PCS::Field>, OpeningsError> {
        for polynomial in &polynomials {
            if polynomial.num_vars() != point_len {
                return Err(OpeningsError::InvalidBatch(format!(
                    "polynomial has {} variables but opening point has {point_len}",
                    polynomial.num_vars()
                )));
            }
        }
        Ok(RlcSource::new(polynomials, scalars.to_vec()))
    }
}

struct HomomorphicBatchStatement<'a, F: Field, C> {
    claims: &'a [VerifierOpeningClaim<F, C>],
    point: Point<HIGH_TO_LOW, F>,
}

impl<'a, F, C> HomomorphicBatchStatement<'a, F, C>
where
    F: Field,
    C: Clone,
{
    fn new(claims: &'a [VerifierOpeningClaim<F, C>]) -> Result<Self, OpeningsError> {
        let first = claims.first().ok_or_else(|| {
            OpeningsError::InvalidBatch("batch opening requires at least one claim".to_owned())
        })?;
        let point = first.evaluation.point.clone();
        for claim in &claims[1..] {
            if claim.evaluation.point != point {
                return Err(OpeningsError::InvalidBatch(
                    "batch opening claims must use one common point".to_owned(),
                ));
            }
        }
        Ok(Self { claims, point })
    }

    fn joint_eval(&self, scalars: &[F]) -> F {
        debug_assert_eq!(self.claims.len(), scalars.len());
        self.claims
            .iter()
            .zip(scalars)
            .fold(F::zero(), |acc, (claim, scalar)| {
                acc + *scalar * claim.evaluation.value
            })
    }

    fn commitments(&self) -> Vec<C> {
        self.claims
            .iter()
            .map(|claim| claim.commitment.clone())
            .collect()
    }
}

impl<F, C> AppendToTranscript for HomomorphicBatchStatement<'_, F, C>
where
    F: Field,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        VerifierRlcClaims(self.claims).append_to_transcript(transcript);
    }
}

impl<PCS, Id> BatchOpeningScheme for PackedBatch<PCS, Id>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    Id: Clone + Debug + Eq + Ord + Send + Sync + 'static,
{
    type Field = PCS::Field;
    type ProverSetup = PrefixPackedProverSetup<PCS, Id>;
    type VerifierSetup = PrefixPackedVerifierSetup<PCS, Id>;
    type Statement = PrefixPackedStatement<PCS::Field, Id, PCS::Output>;
    /// The single packed polynomial backing every logical claim.
    type Polynomials<'a>
        = &'a (dyn MultilinearPoly<PCS::Field> + 'a)
    where
        Self: 'a;
    type Hints = PCS::OpeningHint;
    type Proof = PackedBatchProof<PCS::Field, PCS::Proof>;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        polynomial: Self::Polynomials<'a>,
        hint: Self::Hints,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        let packing = &setup.packing;
        let statement = packing.prepare_statement(&statement)?;
        if polynomial.num_vars() != packing.packed_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "packing polynomial has {} variables but prefix packing has {}",
                polynomial.num_vars(),
                packing.packed_num_vars
            )));
        }
        statement.append_to_transcript(transcript);
        let alpha = transcript.challenge_scalar_powers(statement.num_claims());
        let selector = statement.selector_table(&alpha);
        let packed_evaluations = polynomial.to_dense().into_owned();
        let (round_polynomials, opening_point, opening_eval) =
            prove_reduction_sumcheck(selector, packed_evaluations, transcript);
        EvaluationClaim::new(opening_point.clone(), opening_eval).append_to_transcript(transcript);
        let pcs_proof = PCS::open(
            polynomial,
            &opening_point,
            opening_eval,
            &setup.pcs,
            Some(hint),
            transcript,
        )?;
        Ok(PackedBatchProof {
            round_polynomials,
            opening_eval,
            pcs_proof,
        })
    }

    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        statement: Self::Statement,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let packing = &setup.packing;
        let statement = packing.prepare_statement(&statement)?;
        statement.append_to_transcript(transcript);
        let alpha = transcript.challenge_scalar_powers(statement.num_claims());
        let input_claim = statement.batched_claim(&alpha);
        let (opening_point, final_claim) = verify_reduction_sumcheck(
            &proof.round_polynomials,
            packing.packed_num_vars,
            input_claim,
            transcript,
        )?;
        if final_claim != statement.selector_eval(&alpha, &opening_point) * proof.opening_eval {
            return Err(OpeningsError::VerificationFailed);
        }
        EvaluationClaim::new(opening_point.clone(), proof.opening_eval)
            .append_to_transcript(transcript);
        PCS::verify(
            statement.commitment,
            &opening_point,
            proof.opening_eval,
            &proof.pcs_proof,
            &setup.pcs,
            transcript,
        )
    }
}

impl<PCS> ZkBatchOpeningScheme for HomomorphicBatch<PCS>
where
    PCS: AdditivelyHomomorphic + ZkOpeningScheme,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
{
    type Commitment = PCS::Output;
    type HidingCommitment = PCS::HidingCommitment;
    type Blind = PCS::Blind;

    fn prove_batch_zk<'a, T>(
        setup: &Self::ProverSetup,
        point: Point<HIGH_TO_LOW, Self::Field>,
        commitments: Vec<Self::Commitment>,
        polynomials: Self::Polynomials<'a>,
        hints: Self::Hints,
        evaluations: Vec<Self::Field>,
        transcript: &mut T,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        if commitments.is_empty() {
            return Err(OpeningsError::InvalidBatch(
                "batch opening requires at least one commitment".to_owned(),
            ));
        }
        if polynomials.len() != commitments.len()
            || hints.len() != commitments.len()
            || evaluations.len() != commitments.len()
        {
            return Err(OpeningsError::InvalidBatch(format!(
                "{} polynomials, {} hints, and {} evaluations do not match commitment count {}",
                polynomials.len(),
                hints.len(),
                evaluations.len(),
                commitments.len()
            )));
        }
        let scalars = transcript.challenge_scalar_powers(commitments.len());

        let joint_eval = evaluations
            .iter()
            .zip(&scalars)
            .fold(PCS::Field::from_u64(0), |acc, (eval, scalar)| {
                acc + *scalar * *eval
            });
        let joint_polynomial = Self::combine_polynomials(polynomials, &scalars, point.len())?;
        let combined_hint = PCS::combine_hints(hints, &scalars);
        let (proof, hiding_commitment, blind) = PCS::open_zk(
            &joint_polynomial,
            point.as_slice(),
            joint_eval,
            setup,
            combined_hint,
            transcript,
        )?;
        ZkEvaluationClaim::new(point.as_slice(), &hiding_commitment)
            .append_to_transcript(transcript);
        Ok((proof, hiding_commitment, blind))
    }

    fn verify_batch_zk<T>(
        setup: &Self::VerifierSetup,
        point: Point<HIGH_TO_LOW, Self::Field>,
        commitments: Vec<Self::Commitment>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<Self::HidingCommitment, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        if commitments.is_empty() {
            return Err(OpeningsError::InvalidBatch(
                "batch opening requires at least one commitment".to_owned(),
            ));
        }
        let scalars = transcript.challenge_scalar_powers(commitments.len());
        let joint_commitment = PCS::combine(&commitments, &scalars);
        let hiding_commitment = PCS::verify_zk(
            &joint_commitment,
            point.as_slice(),
            proof,
            setup,
            transcript,
        )?;
        ZkEvaluationClaim::new(point.as_slice(), &hiding_commitment)
            .append_to_transcript(transcript);
        Ok(hiding_commitment)
    }
}

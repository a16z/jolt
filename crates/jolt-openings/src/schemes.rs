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
    PackedWitness, PrefixPackedProverSetup, PrefixPackedStatement, PrefixPackedVerifierSetup,
};

/// Commit to f: F^n -> F, then prove f(r) = v for verifier-chosen r.
pub trait CommitmentScheme: Commitment {
    type Field: Field;
    type Proof: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
    type ProverSetup: Clone + Send + Sync;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    type Polynomial: MultilinearPoly<Self::Field>;

    /// Auxiliary data from commit reused during opening (e.g. Dory row commitments).
    type OpeningHint: Clone + Send + Sync + Default;

    type SetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup);

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup;

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
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

    /// Commit in the scheme's ZK/hiding mode.
    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    /// Open a ZK/hiding commitment using the opening hint returned by
    /// [`commit_zk`](Self::commit_zk).
    fn open_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
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
}

/// Incremental commitment support for schemes whose hiding/ZK commitment mode
/// is distinct from the transparent streaming path, plus the one-hot
/// column-major and primitive-typed feed paths used by the streaming prover.
pub trait ZkStreamingCommitment: StreamingCommitment + ZkOpeningScheme {
    type OneHotChunkCommitment: Clone + Send + Sync;
    type OneHotStreamContext: Send + Sync;

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
pub trait BatchOpeningScheme {
    type Field: Field;
    type ProverSetup;
    type VerifierSetup;
    type Statement;
    type BatchingWitness<'a>
    where
        Self: 'a;
    type Proof;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        witness: Self::BatchingWitness<'a>,
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
pub trait ZkBatchOpeningScheme: BatchOpeningScheme {
    type Commitment;
    type HidingCommitment;
    type Blind;
    type ZkBatchingWitness<'a>
    where
        Self: 'a;

    #[expect(
        clippy::type_complexity,
        reason = "ZK batch openings return the native proof, hiding commitment, and blind"
    )]
    fn prove_batch_zk<'a, T>(
        setup: &Self::ProverSetup,
        point: Point<HIGH_TO_LOW, Self::Field>,
        commitments: Vec<Self::Commitment>,
        witness: Self::ZkBatchingWitness<'a>,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HomomorphicBatch<PCS>(PhantomData<fn() -> PCS>);

impl<PCS> HomomorphicBatch<PCS> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedBatch<PCS, Id = u64>(PhantomData<fn() -> (PCS, Id)>);

impl<PCS, Id> PackedBatch<PCS, Id> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

pub type HomomorphicBatchWitness<'a, F, H> = Vec<(&'a (dyn MultilinearPoly<F> + 'a), H)>;

pub type HomomorphicZkBatchWitness<'a, F, H> = Vec<(&'a (dyn MultilinearPoly<F> + 'a), H, F)>;

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
    type BatchingWitness<'a>
        = HomomorphicBatchWitness<'a, PCS::Field, PCS::OpeningHint>
    where
        Self: 'a;
    type Proof = PCS::Proof;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        claims: Self::Statement,
        witness: Self::BatchingWitness<'a>,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        let statement = HomomorphicBatchStatement::new(&claims)?;
        if witness.len() != statement.claims.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "witness count {} does not match claim count {}",
                witness.len(),
                statement.claims.len()
            )));
        }

        statement.append_to_transcript(transcript);
        let scalars = transcript.challenge_scalar_powers(statement.claims.len());
        let joint_eval = statement.joint_eval(&scalars);
        let (joint_polynomial, combined_hint) =
            Self::combine_witnesses(witness, &scalars, statement.point.len())?;
        let proof = PCS::open(
            &joint_polynomial,
            statement.point.as_slice(),
            joint_eval,
            setup,
            Some(combined_hint),
            transcript,
        );
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
    fn combine_witnesses<'a>(
        witness: HomomorphicBatchWitness<'a, PCS::Field, PCS::OpeningHint>,
        scalars: &[PCS::Field],
        point_len: usize,
    ) -> Result<(HomomorphicRlcSource<'a, PCS::Field>, PCS::OpeningHint), OpeningsError> {
        let mut sources = Vec::with_capacity(witness.len());
        let mut hints = Vec::with_capacity(witness.len());

        for (polynomial, hint) in witness {
            if polynomial.num_vars() != point_len {
                return Err(OpeningsError::InvalidBatch(format!(
                    "polynomial has {} variables but opening point has {point_len}",
                    polynomial.num_vars()
                )));
            }
            sources.push(polynomial);
            hints.push(hint);
        }

        Ok((
            RlcSource::new(sources, scalars.to_vec()),
            PCS::combine_hints(hints, scalars),
        ))
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
    type BatchingWitness<'a>
        = PackedWitness<'a, PCS::Field, PCS::OpeningHint>
    where
        Self: 'a;
    type Proof = PCS::Proof;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        witness: Self::BatchingWitness<'a>,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        let packing = &setup.packing;
        let statement = packing.prepare_statement(&statement)?;
        if witness.polynomial.num_vars() != packing.packed_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "packing polynomial has {} variables but prefix packing has {}",
                witness.polynomial.num_vars(),
                packing.packed_num_vars
            )));
        }
        statement.append_to_transcript(transcript);
        let opening_point = statement.opening_point(transcript)?;
        let opening_eval = statement.reduced_eval(&opening_point);
        let native = PCS::open(
            witness.polynomial,
            &opening_point,
            opening_eval,
            &setup.pcs,
            Some(witness.hint),
            transcript,
        );
        EvaluationClaim::new(opening_point, opening_eval).append_to_transcript(transcript);
        Ok(native)
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
        let opening_point = statement.opening_point(transcript)?;
        let opening_eval = statement.reduced_eval(&opening_point);
        PCS::verify(
            statement.commitment,
            &opening_point,
            opening_eval,
            proof,
            &setup.pcs,
            transcript,
        )?;
        EvaluationClaim::new(opening_point, opening_eval).append_to_transcript(transcript);
        Ok(())
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
    type ZkBatchingWitness<'a>
        = HomomorphicZkBatchWitness<'a, PCS::Field, PCS::OpeningHint>
    where
        Self: 'a;

    fn prove_batch_zk<'a, T>(
        setup: &Self::ProverSetup,
        point: Point<HIGH_TO_LOW, Self::Field>,
        commitments: Vec<Self::Commitment>,
        witness: Self::ZkBatchingWitness<'a>,
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
        if witness.len() != commitments.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "witness count {} does not match commitment count {}",
                witness.len(),
                commitments.len()
            )));
        }
        let scalars = transcript.challenge_scalar_powers(commitments.len());

        let mut joint_eval = PCS::Field::from_u64(0);
        let mut sources = Vec::with_capacity(witness.len());
        let mut hints = Vec::with_capacity(witness.len());
        for ((polynomial, hint, eval), scalar) in witness.into_iter().zip(&scalars) {
            if polynomial.num_vars() != point.len() {
                return Err(OpeningsError::InvalidBatch(format!(
                    "polynomial has {} variables but opening point has {}",
                    polynomial.num_vars(),
                    point.len()
                )));
            }
            joint_eval += *scalar * eval;
            sources.push(polynomial);
            hints.push(hint);
        }

        let joint_polynomial = RlcSource::new(sources, scalars.clone());
        let combined_hint = PCS::combine_hints(hints, &scalars);
        let (proof, hiding_commitment, blind) = PCS::open_zk(
            &joint_polynomial,
            point.as_slice(),
            joint_eval,
            setup,
            combined_hint,
            transcript,
        );
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

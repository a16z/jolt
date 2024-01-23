use std::marker::{PhantomData, Sync};

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    jolt::instruction::JoltInstruction,
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        dense_mlpoly::{DensePolynomial, PolyCommitment, PolyCommitmentGens, PolyEvalProof},
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::{
        combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
        sumcheck::SumcheckInstanceProof,
    },
    utils::{
        errors::ProofVerifyError, math::Math, mul_0_1_optimized, random::RandomTape,
        transcript::ProofTranscript,
    },
};

use super::read_write_memory::{
    BatchedMemoryPolynomials, MemoryCommitment, MemoryReadWriteOpenings, ReadWriteMemory,
};

pub struct RangeCheckPolynomials<'a, F: PrimeField, G: CurveGroup<ScalarField = F>> {
    pub memory_polynomials: &'a ReadWriteMemory<F, G>,
    pub batched_memory_polynomials: &'a BatchedMemoryPolynomials<F>,
    pub memory_commitment: &'a MemoryCommitment<G>,
    pub read_cts: DensePolynomial<F>,
    pub final_cts: DensePolynomial<F>,
}

impl<'a, F, G> RangeCheckPolynomials<'a, F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    pub fn new(
        M: usize,
        read_timestamps: &Vec<u64>,
        memory_polynomials: &'a ReadWriteMemory<F, G>,
        batched_memory_polynomials: &'a BatchedMemoryPolynomials<F>,
        memory_commitment: &'a MemoryCommitment<G>,
    ) -> Self {
        let mut read_cts = vec![0usize; memory_polynomials.t_read.len()];
        let mut final_cts = vec![0usize; M];
        for (i, read_timestamp) in read_timestamps.iter().enumerate() {
            // TODO(moodlezoup): Also range-check read_timestamp alone
            let lookup_index = i + 1 - *read_timestamp as usize;
            let count = final_cts[lookup_index];
            read_cts[i] = count;
            final_cts[lookup_index] += 1;
        }

        Self {
            read_cts: DensePolynomial::from_usize(&read_cts),
            final_cts: DensePolynomial::from_usize(&final_cts),
            memory_polynomials,
            batched_memory_polynomials,
            memory_commitment,
        }
    }
}

pub struct RangeCheckCommitment<'a, G: CurveGroup> {
    generators: RangeCheckCommitmentGenerators<G>,
    pub memory_commitment: &'a MemoryCommitment<G>,
    pub read_cts_commitment: PolyCommitment<G>,
    pub final_cts_commitment: PolyCommitment<G>,
}

/// Container for generators for polynomial commitments. These preallocate memory
/// and allow commitments to `DensePolynomials`.
pub struct RangeCheckCommitmentGenerators<G: CurveGroup> {
    pub read_commitment_gens: PolyCommitmentGens<G>,
    pub final_commitment_gens: PolyCommitmentGens<G>,
}

impl<'a, F, G> BatchablePolynomials for RangeCheckPolynomials<'a, F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = Self;
    type Commitment = RangeCheckCommitment<'a, G>;

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        unimplemented!("Nothing to batch")
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::commit")]
    fn commit(polys: &Self) -> Self::Commitment {
        let read_commitment_gens = PolyCommitmentGens::new(
            polys.read_cts.get_num_vars(),
            b"RangeCheckPolynomials.read_cts",
        );
        let final_commitment_gens = PolyCommitmentGens::new(
            polys.final_cts.get_num_vars(),
            b"RangeCheckPolynomials.final_cts",
        );

        let (read_cts_commitment, _) = polys.read_cts.commit(&read_commitment_gens, None);
        let (final_cts_commitment, _) = polys.final_cts.commit(&final_commitment_gens, None);

        let generators = RangeCheckCommitmentGenerators {
            read_commitment_gens,
            final_commitment_gens,
        };

        Self::Commitment {
            generators,
            memory_commitment: &polys.memory_commitment,
            read_cts_commitment,
            final_cts_commitment,
        }
    }
}

pub struct RangeCheckReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    read_cts_opening: F,
    read_cts_opening_proof: PolyEvalProof<G>,
    memory_poly_openings: MemoryReadWriteOpenings<F, G>,
    identity_poly_opening: Option<F>,
}

impl<F, G> StructuredOpeningProof<F, G, RangeCheckPolynomials<'_, F, G>>
    for RangeCheckReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Openings = [F; 6];

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::open")]
    fn open(polynomials: &RangeCheckPolynomials<F, G>, opening_point: &Vec<F>) -> Self::Openings {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        [
            &polynomials.read_cts,
            &polynomials.memory_polynomials.a_read_write,
            &polynomials.memory_polynomials.v_read,
            &polynomials.memory_polynomials.v_write,
            &polynomials.memory_polynomials.t_read,
            &polynomials.memory_polynomials.t_write,
        ]
        .par_iter()
        .map(|poly| poly.evaluate_at_chi(&chis))
        .collect::<Vec<F>>()
        .try_into()
        .unwrap()
    }

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &RangeCheckPolynomials<F, G>,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        openings: [F; 6],
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        let read_cts_opening = openings[0];
        let (read_cts_opening_proof, _) = PolyEvalProof::prove(
            &polynomials.read_cts,
            None,
            opening_point,
            &read_cts_opening,
            None,
            &commitment.generators.read_commitment_gens,
            transcript,
            random_tape,
        );

        let memory_poly_openings = MemoryReadWriteOpenings::prove_openings(
            polynomials.batched_memory_polynomials,
            &commitment.memory_commitment,
            opening_point,
            openings[1..].try_into().unwrap(),
            transcript,
            random_tape,
        );

        Self {
            read_cts_opening,
            read_cts_opening_proof,
            memory_poly_openings,
            identity_poly_opening: None,
        }
    }

    fn compute_verifier_openings(&mut self, opening_point: &Vec<F>) {
        self.identity_poly_opening =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        self.read_cts_opening_proof.verify_plain(
            &commitment.generators.read_commitment_gens,
            transcript,
            opening_point,
            &self.read_cts_opening,
            &commitment.read_cts_commitment,
        )?;

        self.memory_poly_openings.verify_openings(
            commitment.memory_commitment,
            opening_point,
            transcript,
        )
    }
}

pub struct RangeCheckFinalOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    identity_poly_opening: Option<F>,
    final_cts_opening: F,
    final_cts_opening_proof: PolyEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, RangeCheckPolynomials<'_, F, G>>
    for RangeCheckFinalOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Openings = F;

    #[tracing::instrument(skip_all, name = "RangeCheckFinalOpenings::open")]
    fn open(polynomials: &RangeCheckPolynomials<F, G>, opening_point: &Vec<F>) -> Self::Openings {
        polynomials.final_cts.evaluate(&opening_point)
    }

    #[tracing::instrument(skip_all, name = "RangeCheckFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &RangeCheckPolynomials<F, G>,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        final_cts_opening: F,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        let (opening_proof, _) = PolyEvalProof::prove(
            &polynomials.final_cts,
            None,
            opening_point,
            &final_cts_opening,
            None,
            &commitment.generators.read_commitment_gens,
            transcript,
            random_tape,
        );

        Self {
            identity_poly_opening: None,
            final_cts_opening,
            final_cts_opening_proof: opening_proof,
        }
    }

    fn compute_verifier_openings(&mut self, opening_point: &Vec<F>) {
        self.identity_poly_opening =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        self.final_cts_opening_proof.verify_plain(
            &commitment.generators.final_commitment_gens,
            transcript,
            opening_point,
            &self.final_cts_opening,
            &commitment.final_cts_commitment,
        )
    }
}

impl<F, G> MemoryCheckingProver<F, G, RangeCheckPolynomials<'_, F, G>>
    for TimestampValidityProof<'_, F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type ReadWriteOpenings = RangeCheckReadWriteOpenings<F, G>;
    type InitFinalOpenings = RangeCheckFinalOpenings<F, G>;

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - tau
    }

    #[tracing::instrument(skip_all, name = "TimestampValidityProof::read_leaves")]
    fn read_leaves(
        &self,
        polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        let gamma_squared = gamma.square();
        let leaf_fingerprints = (0..self.num_lookups)
            .map(|i| {
                mul_0_1_optimized(&polynomials.read_cts[i], &gamma_squared)
                    + (F::from(i as u64) - polynomials.memory_polynomials.t_read[i]) * gamma
                    + (F::from(i as u64) - polynomials.memory_polynomials.t_read[i])
                    - *tau
            })
            .collect();
        vec![DensePolynomial::new(leaf_fingerprints)]
    }

    #[tracing::instrument(skip_all, name = "TimestampValidityProof::write_leaves")]
    fn write_leaves(
        &self,
        polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        let gamma_squared = gamma.square();
        let leaf_fingerprints = (0..self.num_lookups)
            .map(|i| {
                mul_0_1_optimized(&(polynomials.read_cts[i] + F::one()), &gamma_squared)
                    + (F::from(i as u64) - polynomials.memory_polynomials.t_read[i]) * gamma
                    + (F::from(i as u64) - polynomials.memory_polynomials.t_read[i])
                    - *tau
            })
            .collect();
        vec![DensePolynomial::new(leaf_fingerprints)]
    }

    #[tracing::instrument(skip_all, name = "TimestampValidityProof::init_leaves")]
    fn init_leaves(
        &self,
        _polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        let leaf_fingerprints = (0..self.M)
            .map(|i| {
                // 0 * gamma^2 +
                F::from(i as u64) * gamma + F::from(i as u64) - *tau
            })
            .collect();
        vec![DensePolynomial::new(leaf_fingerprints)]
    }

    #[tracing::instrument(skip_all, name = "TimestampValidityProof::final_leaves")]
    fn final_leaves(
        &self,
        polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        let gamma_squared = gamma.square();
        let leaf_fingerprints = (0..self.M)
            .map(|i| {
                polynomials.final_cts[i] * gamma_squared
                    + F::from(i as u64) * gamma
                    + F::from(i as u64)
                    - *tau
            })
            .collect();
        vec![DensePolynomial::new(leaf_fingerprints)]
    }

    fn protocol_name() -> &'static [u8] {
        b"Timestamp validity proof memory checking"
    }
}

impl<F, G> MemoryCheckingVerifier<F, G, RangeCheckPolynomials<'_, F, G>>
    for TimestampValidityProof<'_, F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.identity_poly_opening.unwrap() - openings.memory_poly_openings.t_read_opening,
            openings.identity_poly_opening.unwrap() - openings.memory_poly_openings.t_read_opening,
            openings.read_cts_opening,
        )]
    }

    fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.identity_poly_opening.unwrap() - openings.memory_poly_openings.t_read_opening,
            openings.identity_poly_opening.unwrap() - openings.memory_poly_openings.t_read_opening,
            openings.read_cts_opening + F::one(),
        )]
    }

    fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.identity_poly_opening.unwrap(),
            openings.identity_poly_opening.unwrap(),
            F::zero(),
        )]
    }

    fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.identity_poly_opening.unwrap(),
            openings.identity_poly_opening.unwrap(),
            openings.final_cts_opening,
        )]
    }
}

pub struct TimestampValidityProof<'a, F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    M: usize,
    num_lookups: usize,
    /// Commitments to all polynomials
    commitment: RangeCheckCommitment<'a, G>,

    memory_checking: MemoryCheckingProof<
        G,
        RangeCheckPolynomials<'a, F, G>,
        RangeCheckReadWriteOpenings<F, G>,
        RangeCheckFinalOpenings<F, G>,
    >,
}

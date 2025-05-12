use crate::{
    field::JoltField,
    jolt::vm::{JoltCommitments, JoltPolynomials, ProverDebugInfo},
    poly::{
        compact_polynomial::{CompactPolynomial, SmallScalar},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    subprotocols::grand_product::BatchedDenseGrandProduct,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::marker::{PhantomData, Sync};

use super::memory_checking::{
    Initializable, NoExogenousOpenings, StructuredPolynomialData, VerifierComputedOpening,
};
use crate::{
    jolt::instruction::JoltInstruction,
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        commitment::commitment_scheme::CommitmentScheme, dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    /// C-sized vector of `dim_i` polynomials/commitments/openings
    pub(crate) dim: Vec<T>,
    /// C-sized vector of `read_cts_i` polynomials/commitments/openings
    pub(crate) read_cts: Vec<T>,
    /// C-sized vector of `E_i` polynomials/commitments/openings
    pub(crate) E_polys: Vec<T>,
    /// `num_memories`-sized vector of `final_cts_i` polynomials/commitments/openings
    pub(crate) final_cts: Vec<T>,

    a_init_final: VerifierComputedOpening<T>,
    v_init_final: VerifierComputedOpening<Vec<T>>,
}

pub type SurgePolynomials<F: JoltField> = SurgeStuff<MultilinearPolynomial<F>>;
pub type SurgeOpenings<F: JoltField> = SurgeStuff<F>;
pub type SurgeCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    SurgeStuff<PCS::Commitment>;

impl<const C: usize, const M: usize, F, T, Instruction>
    Initializable<T, SurgePreprocessing<F, Instruction, C, M>> for SurgeStuff<T>
where
    F: JoltField,
    T: CanonicalSerialize + CanonicalDeserialize + Default,
    Instruction: JoltInstruction + Default,
{
    fn initialize(_preprocessing: &SurgePreprocessing<F, Instruction, C, M>) -> Self {
        let num_memories = C * Instruction::default().subtables::<F>(C, M).len();
        Self {
            dim: std::iter::repeat_with(|| T::default()).take(C).collect(),
            read_cts: std::iter::repeat_with(|| T::default()).take(C).collect(),
            final_cts: std::iter::repeat_with(|| T::default()).take(C).collect(),
            E_polys: std::iter::repeat_with(|| T::default())
                .take(num_memories)
                .collect(),
            a_init_final: None,
            v_init_final: None,
        }
    }
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T> for SurgeStuff<T> {
    fn read_write_values(&self) -> Vec<&T> {
        self.dim
            .iter()
            .chain(self.read_cts.iter())
            .chain(self.E_polys.iter())
            .collect()
    }

    fn init_final_values(&self) -> Vec<&T> {
        self.final_cts.iter().collect()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        self.dim
            .iter_mut()
            .chain(self.read_cts.iter_mut())
            .chain(self.E_polys.iter_mut())
            .collect()
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        self.final_cts.iter_mut().collect()
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize, ProofTranscript: Transcript>
    MemoryCheckingProver<F, PCS, ProofTranscript>
    for SurgeProof<F, PCS, Instruction, C, M, ProofTranscript>
where
    F: JoltField,
    Instruction: JoltInstruction + Default + Sync,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
{
    type ReadWriteGrandProduct = BatchedDenseGrandProduct<F>;
    type InitFinalGrandProduct = BatchedDenseGrandProduct<F>;

    type Polynomials = SurgePolynomials<F>;
    type Openings = SurgeOpenings<F>;
    type Commitments = SurgeCommitments<PCS, ProofTranscript>;
    type ExogenousOpenings = NoExogenousOpenings;

    type Preprocessing = SurgePreprocessing<F, Instruction, C, M>;

    /// The data associated with each memory slot. A triple (a, v, t) by default.
    type MemoryTuple = (F, F, F); // (a, v, t)

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_leaves")]
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> ((Vec<F>, usize), (Vec<F>, usize)) {
        let gamma_squared = gamma.square();

        let num_lookups = polynomials.dim[0].len();

        let read_write_leaves: Vec<_> = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let read_cts: &CompactPolynomial<u32, F> =
                    (&polynomials.read_cts[dim_index]).try_into().unwrap();
                let E_poly: &CompactPolynomial<u32, F> =
                    (&polynomials.E_polys[memory_index]).try_into().unwrap();
                let dim: &CompactPolynomial<u16, F> =
                    (&polynomials.dim[dim_index]).try_into().unwrap();
                let read_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| {
                        let a = dim[i];
                        let v = E_poly[i];
                        let t = read_cts[i];
                        t.field_mul(gamma_squared) + v.field_mul(*gamma) + F::from_u16(a) - *tau
                    })
                    .collect();
                let t_adjustment = 1u64.field_mul(gamma_squared);
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + t_adjustment)
                    .collect();

                vec![read_fingerprints, write_fingerprints]
            })
            .collect();

        let init_final_leaves: Vec<_> = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let subtable_index = Self::memory_to_subtable_index(memory_index);
                // TODO(moodlezoup): Only need one init polynomial per subtable
                let init_fingerprints: Vec<F> = (0..M)
                    .map(|i| {
                        // 0 * gamma^2 +
                        preprocessing.materialized_subtables[subtable_index][i].field_mul(*gamma)
                            + F::from_u64(i as u64)
                            - *tau
                    })
                    .collect();
                let final_fingerprints = init_fingerprints
                    .iter()
                    .enumerate()
                    .map(|(i, init_fingerprint)| {
                        let final_cts: &CompactPolynomial<u32, F> =
                            (&polynomials.final_cts[dim_index]).try_into().unwrap();
                        *init_fingerprint + final_cts[i].field_mul(gamma_squared)
                    })
                    .collect();

                vec![init_fingerprints, final_fingerprints]
            })
            .collect();

        // TODO(moodlezoup): avoid concat
        (
            (read_write_leaves.concat(), 2 * Self::num_memories()),
            (init_final_leaves.concat(), 2 * Self::num_memories()),
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"SurgeMemCheck"
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize, ProofTranscript>
    MemoryCheckingVerifier<F, PCS, ProofTranscript>
    for SurgeProof<F, PCS, Instruction, C, M, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Instruction: JoltInstruction + Default + Sync,
    ProofTranscript: Transcript,
{
    fn compute_verifier_openings(
        openings: &mut Self::Openings,
        _preprocessing: &Self::Preprocessing,
        _r_read_write: &[F],
        r_init_final: &[F],
    ) {
        openings.a_init_final =
            Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));
        openings.v_init_final = Some(
            Instruction::default()
                .subtables(C, M)
                .iter()
                .map(|(subtable, _)| subtable.evaluate_mle(r_init_final))
                .collect(),
        );
    }

    fn read_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim[dim_index],
                    openings.E_polys[memory_index],
                    openings.read_cts[dim_index],
                )
            })
            .collect()
    }
    fn write_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim[dim_index],
                    openings.E_polys[memory_index],
                    openings.read_cts[dim_index] + F::one(),
                )
            })
            .collect()
    }
    fn init_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::num_memories())
            .map(|memory_index| {
                (
                    a_init,
                    v_init[Self::memory_to_subtable_index(memory_index)],
                    F::zero(),
                )
            })
            .collect()
    }
    fn final_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    a_init,
                    v_init[Self::memory_to_subtable_index(memory_index)],
                    openings.final_cts[dim_index],
                )
            })
            .collect()
    }
}

pub struct SurgePrimarySumcheck<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    num_rounds: usize,
    claimed_evaluation: F,
    E_poly_openings: Vec<F>,
    _marker: PhantomData<ProofTranscript>,
}

pub struct SurgePreprocessing<F, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,
    _field: PhantomData<F>,
    materialized_subtables: Vec<Vec<u32>>,
}

#[allow(clippy::type_complexity)]
pub struct SurgeProof<F, PCS, Instruction, const C: usize, const M: usize, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Instruction: JoltInstruction + Default,
    ProofTranscript: Transcript,
{
    _instruction: PhantomData<Instruction>,
    /// Commitments to all polynomials
    commitments: SurgeCommitments<PCS, ProofTranscript>,

    /// Primary collation sumcheck proof
    primary_sumcheck: SurgePrimarySumcheck<F, ProofTranscript>,

    memory_checking:
        MemoryCheckingProof<F, PCS, SurgeOpenings<F>, NoExogenousOpenings, ProofTranscript>,
}

impl<F, Instruction, const C: usize, const M: usize> SurgePreprocessing<F, Instruction, C, M>
where
    F: JoltField,
    Instruction: JoltInstruction + Default + Sync,
{
    #[tracing::instrument(skip_all, name = "Surge::preprocess")]
    pub fn preprocess() -> Self {
        let instruction = Instruction::default();

        let materialized_subtables = instruction
            .subtables::<F>(C, M)
            .par_iter()
            .map(|(subtable, _)| subtable.materialize(M))
            .collect();

        // TODO(moodlezoup): do PCS setup here

        Self {
            _instruction: PhantomData,
            _field: PhantomData,
            materialized_subtables,
        }
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize, ProofTranscript>
    SurgeProof<F, PCS, Instruction, C, M, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Instruction: JoltInstruction + Default + Sync,
    ProofTranscript: Transcript,
{
    // TODO(moodlezoup): We can be more efficient (use fewer memories) if we use subtable_indices
    fn num_memories() -> usize {
        C * Instruction::default().subtables::<F>(C, M).len()
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, NUM_SUBTABLES)
    fn memory_to_subtable_index(i: usize) -> usize {
        i / C
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, C)
    fn memory_to_dimension_index(i: usize) -> usize {
        i % C
    }

    fn protocol_name() -> &'static [u8] {
        b"Surge"
    }

    /// Computes the maximum number of group generators needed to commit to Surge polynomials
    /// using Hyrax, given `M` and the maximum number of lookups.
    pub fn num_generators(max_num_lookups: usize) -> usize {
        std::cmp::max(
            max_num_lookups.next_power_of_two(),
            (M * Self::num_memories()).next_power_of_two(),
        )
    }

    #[tracing::instrument(skip_all, name = "Surge::prove")]
    pub fn prove(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        ops: Vec<Instruction>,
    ) -> (Self, Option<ProverDebugInfo<F, ProofTranscript>>) {
        let mut transcript = ProofTranscript::new(b"Surge transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let num_lookups = ops.len().next_power_of_two();
        let polynomials = Self::generate_witness(preprocessing, &ops);

        let mut commitments = SurgeCommitments::<PCS, ProofTranscript>::initialize(preprocessing);
        let trace_polys = polynomials.read_write_values();
        let trace_comitments = PCS::batch_commit(&trace_polys, generators);
        commitments
            .read_write_values_mut()
            .into_iter()
            .zip(trace_comitments.into_iter())
            .for_each(|(dest, src)| *dest = src);
        commitments.final_cts = PCS::batch_commit(&polynomials.final_cts, generators);

        let num_rounds = num_lookups.log_2();
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let r_primary_sumcheck: Vec<F> = transcript.challenge_vector(num_rounds);
        let eq = MultilinearPolynomial::from(EqPolynomial::evals(&r_primary_sumcheck));
        let sumcheck_claim: F = Self::compute_primary_sumcheck_claim(&polynomials, &eq);

        transcript.append_scalar(&sumcheck_claim);
        let mut combined_sumcheck_polys = polynomials.E_polys.clone();
        combined_sumcheck_polys.push(eq);

        let combine_lookups_eq = |vals: &[F]| -> F {
            let vals_no_eq: &[F] = &vals[0..(vals.len() - 1)];
            let eq = vals[vals.len() - 1];
            instruction.combine_lookups(vals_no_eq, C, M) * eq
        };

        let (primary_sumcheck_proof, r_z, mut sumcheck_openings) =
            SumcheckInstanceProof::<F, ProofTranscript>::prove_arbitrary::<_>(
                &sumcheck_claim,
                num_rounds,
                &mut combined_sumcheck_polys,
                combine_lookups_eq,
                instruction.g_poly_degree(C) + 1, // combined degree + eq term
                &mut transcript,
            );

        // Remove EQ
        let _ = combined_sumcheck_polys.pop();
        let _ = sumcheck_openings.pop();
        opening_accumulator.append(
            &polynomials.E_polys.iter().collect::<Vec<_>>(),
            DensePolynomial::new(EqPolynomial::evals(&r_z)),
            r_z.clone(),
            &sumcheck_openings,
            &mut transcript,
        );

        let primary_sumcheck = SurgePrimarySumcheck {
            claimed_evaluation: sumcheck_claim,
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            E_poly_openings: sumcheck_openings,
            _marker: PhantomData,
        };

        let memory_checking = SurgeProof::prove_memory_checking(
            generators,
            preprocessing,
            &polynomials,
            &JoltPolynomials::default(), // Hack: required by the memory-checking trait, but unused in Surge
            &mut opening_accumulator,
            &mut transcript,
        );

        let proof = SurgeProof {
            _instruction: PhantomData,
            commitments,
            primary_sumcheck,
            memory_checking,
        };
        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript,
            opening_accumulator,
        });
        #[cfg(not(test))]
        let debug_info = None;

        (proof, debug_info)
    }

    pub fn verify(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        proof: SurgeProof<F, PCS, Instruction, C, M, ProofTranscript>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript>>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Surge transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();
        #[cfg(test)]
        if let Some(debug_info) = _debug_info {
            transcript.compare_to(debug_info.transcript);
            opening_accumulator.compare_to(debug_info.opening_accumulator, generators);
        }

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);
        let instruction = Instruction::default();

        let r_primary_sumcheck = transcript.challenge_vector(proof.primary_sumcheck.num_rounds);

        transcript.append_scalar(&proof.primary_sumcheck.claimed_evaluation);
        let primary_sumcheck_poly_degree = instruction.g_poly_degree(C) + 1;
        let (claim_last, r_z) = proof.primary_sumcheck.sumcheck_proof.verify(
            proof.primary_sumcheck.claimed_evaluation,
            proof.primary_sumcheck.num_rounds,
            primary_sumcheck_poly_degree,
            &mut transcript,
        )?;

        let eq_eval = EqPolynomial::new(r_primary_sumcheck.to_vec()).evaluate(&r_z);
        assert_eq!(
            eq_eval * instruction.combine_lookups(&proof.primary_sumcheck.E_poly_openings, C, M),
            claim_last,
            "Primary sumcheck check failed."
        );

        opening_accumulator.append(
            &proof.commitments.E_polys.iter().collect::<Vec<_>>(),
            r_z.clone(),
            &proof
                .primary_sumcheck
                .E_poly_openings
                .iter()
                .collect::<Vec<_>>(),
            &mut transcript,
        );

        Self::verify_memory_checking(
            preprocessing,
            generators,
            proof.memory_checking,
            &proof.commitments,
            &JoltCommitments::<PCS, ProofTranscript>::default(),
            &mut opening_accumulator,
            &mut transcript,
        )
    }

    #[tracing::instrument(skip_all, name = "Surge::construct_polys")]
    fn generate_witness(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        ops: &[Instruction],
    ) -> SurgePolynomials<F> {
        let num_lookups = ops.len().next_power_of_two();
        let mut dim: Vec<Vec<u16>> = vec![vec![0; num_lookups]; C];

        let mut read_cts = vec![vec![0u32; num_lookups]; C];
        let mut final_cts = vec![vec![0u32; M]; C];
        let log_M = ark_std::log2(M) as usize;

        for (op_index, op) in ops.iter().enumerate() {
            let access_sequence = op.to_indices(C, log_M);
            assert_eq!(access_sequence.len(), C);

            for dimension_index in 0..C {
                let memory_address = access_sequence[dimension_index];
                debug_assert!(memory_address < M);

                dim[dimension_index][op_index] = memory_address as u16;

                let ts = final_cts[dimension_index][memory_address];
                read_cts[dimension_index][op_index] = ts;
                let write_timestamp = ts + 1;
                final_cts[dimension_index][memory_address] = write_timestamp;
            }
        }

        // num_ops is padded to the nearest power of 2 for the usage of DensePolynomial. We cannot just fill
        // in zeros for read_cts and final_cts as this implicitly specifies a read at address 0. The prover
        // and verifier plumbing assume write_ts(r) = read_ts(r) + 1. This will not hold unless we update
        // the final_cts for these phantom reads.
        for fake_ops_index in ops.len()..num_lookups {
            for dimension_index in 0..C {
                let memory_address = 0;
                let ts = final_cts[dimension_index][memory_address];
                read_cts[dimension_index][fake_ops_index] = ts;
                let write_timestamp = ts + 1;
                final_cts[dimension_index][memory_address] = write_timestamp;
            }
        }

        // Construct E
        let mut E_i_evals = Vec::with_capacity(Self::num_memories());
        for E_index in 0..Self::num_memories() {
            let mut E_evals = Vec::with_capacity(num_lookups);
            for op_index in 0..num_lookups {
                let dimension_index = Self::memory_to_dimension_index(E_index);
                let subtable_index = Self::memory_to_subtable_index(E_index);

                let eval_index = dim[dimension_index][op_index];
                let eval =
                    preprocessing.materialized_subtables[subtable_index][eval_index as usize];
                E_evals.push(eval);
            }
            E_i_evals.push(E_evals);
        }

        let E_polys: Vec<MultilinearPolynomial<F>> = E_i_evals
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        let dim: Vec<MultilinearPolynomial<F>> =
            dim.into_iter().map(MultilinearPolynomial::from).collect();
        let read_cts: Vec<MultilinearPolynomial<F>> = read_cts
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        let final_cts: Vec<MultilinearPolynomial<F>> = final_cts
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        SurgePolynomials {
            dim,
            read_cts,
            final_cts,
            E_polys,
            a_init_final: None,
            v_init_final: None,
        }
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_primary_sumcheck_claim")]
    fn compute_primary_sumcheck_claim(
        polys: &SurgePolynomials<F>,
        eq: &MultilinearPolynomial<F>,
    ) -> F {
        let g_operands = &polys.E_polys;
        let hypercube_size = g_operands[0].len();
        g_operands
            .iter()
            .for_each(|operand| assert_eq!(operand.len(), hypercube_size));

        let instruction = Instruction::default();

        (0..hypercube_size)
            .into_par_iter()
            .map(|eval_index| {
                let g_operands: Vec<F> = (0..Self::num_memories())
                    .map(|memory_index| g_operands[memory_index].get_coeff(eval_index))
                    .collect();
                eq.get_coeff(eval_index) * instruction.combine_lookups(&g_operands, C, M)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::SurgePreprocessing;
    use crate::utils::transcript::KeccakTranscript;
    use crate::{
        jolt::instruction::xor::XORInstruction,
        lasso::surge::SurgeProof,
        poly::commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
    };
    use ark_bn254::{Bn254, Fr};
    use ark_std::test_rng;
    use rand_core::RngCore;

    #[test]
    fn surge_32_e2e() {
        let mut rng = test_rng();
        const WORD_SIZE: usize = 32;
        const C: usize = 4;
        const M: usize = 1 << 16;
        const NUM_OPS: usize = 1024;

        let ops = std::iter::repeat_with(|| {
            XORInstruction::<WORD_SIZE>(rng.next_u32() as u64, rng.next_u32() as u64)
        })
        .take(NUM_OPS)
        .collect();

        let preprocessing = SurgePreprocessing::preprocess();
        let generators = HyperKZG::<_, KeccakTranscript>::setup(M);
        let (proof, debug_info) = SurgeProof::<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            XORInstruction<WORD_SIZE>,
            C,
            M,
            KeccakTranscript,
        >::prove(&preprocessing, &generators, ops);

        SurgeProof::verify(&preprocessing, &generators, proof, debug_info).expect("should work");
    }

    #[test]
    fn surge_32_e2e_non_pow_2() {
        let mut rng = test_rng();
        const WORD_SIZE: usize = 32;
        const C: usize = 4;
        const M: usize = 1 << 16;

        const NUM_OPS: usize = 1000;

        let ops = std::iter::repeat_with(|| {
            XORInstruction::<WORD_SIZE>(rng.next_u32() as u64, rng.next_u32() as u64)
        })
        .take(NUM_OPS)
        .collect();

        let preprocessing = SurgePreprocessing::preprocess();
        let generators = HyperKZG::<_, KeccakTranscript>::setup(M);
        let (proof, debug_info) = SurgeProof::<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            XORInstruction<WORD_SIZE>,
            C,
            M,
            KeccakTranscript,
        >::prove(&preprocessing, &generators, ops);

        SurgeProof::verify(&preprocessing, &generators, proof, debug_info).expect("should work");
    }
}

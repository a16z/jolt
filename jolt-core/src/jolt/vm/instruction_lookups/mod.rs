use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::marker::PhantomData;
use tracer::instruction::RV32IMCycle;

use crate::{
    dag::{
        stage::{StagedSumcheck, SumcheckStages},
        state_manager::StateManager,
    },
    field::JoltField,
    jolt::{
        instruction::LookupQuery,
        lookup_table::LookupTables,
        vm::{
            instruction_lookups::{
                booleanity::{BooleanityProof, BooleanitySumcheck},
                hamming_weight::{HammingWeightProof, HammingWeightSumcheck},
                read_raf_checking::{ReadCheckingProof, ReadRafSumcheck},
            },
            JoltCommitments, JoltProverPreprocessing,
        },
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        opening_proof::{OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    r1cs::inputs::JoltR1CSInputs,
    utils::{errors::ProofVerifyError, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};

pub mod booleanity;
pub mod hamming_weight;
pub mod read_raf_checking;

pub const WORD_SIZE: usize = 32;
const LOG_K: usize = WORD_SIZE * 2;
const PHASES: usize = 4;
const LOG_M: usize = LOG_K / PHASES;
const M: usize = 1 << LOG_M;
pub const D: usize = 8;
pub const LOG_K_CHUNK: usize = LOG_K / D;
pub const K_CHUNK: usize = 1 << LOG_K_CHUNK;
const RA_PER_LOG_M: usize = LOG_M / LOG_K_CHUNK;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct LookupsProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    read_checking_proof: ReadCheckingProof<F, ProofTranscript>,
    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    log_T: usize,
    _marker: PhantomData<PCS>,
}

#[derive(Default)]
pub struct LookupsDag {}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStages<F, T, PCS>
    for LookupsDag
{
    fn stage3_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let (preprocessing, trace, _, _) = sm.get_prover_data();
        let unbound_ra_polys = (0..D)
            .map(|i| CommittedPolynomials::InstructionRa(i).generate_witness(preprocessing, trace))
            .collect::<Vec<_>>();
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        let eq_r_cycle = EqPolynomial::evals(&r_cycle);
        let F = compute_ra_evals(trace, &eq_r_cycle);

        let read_raf =
            ReadRafSumcheck::new_prover(sm, eq_r_cycle.clone(), unbound_ra_polys.clone());

        let booleanity =
            BooleanitySumcheck::new_prover(sm, eq_r_cycle, F.clone(), unbound_ra_polys.clone());

        let hamming_weight = HammingWeightSumcheck::new_prover(sm, F, unbound_ra_polys.clone());

        vec![
            Box::new(read_raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }

    fn stage3_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let read_raf = ReadRafSumcheck::new_verifier(sm);
        let booleanity = BooleanitySumcheck::new_verifier(sm);
        let hamming_weight = HammingWeightSumcheck::new_verifier(sm);

        vec![
            Box::new(read_raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }
}

impl<const WORD_SIZE: usize, F, PCS, ProofTranscript>
    LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    pub fn generate_witness(_preprocessing: (), _lookups: &[LookupTables<WORD_SIZE>]) {}

    #[tracing::instrument(skip_all, name = "LookupsProof::prove")]
    pub fn prove(
        _preprocessing: &JoltProverPreprocessing<F, PCS>,
        _trace: &[RV32IMCycle],
        _opening_accumulator: &mut ProverOpeningAccumulator<F, PCS>,
        _transcript: &mut ProofTranscript,
    ) -> Self {
        todo!();
    }

    pub fn verify(
        &self,
        _commitments: &JoltCommitments<F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        todo!()
    }
}

#[inline(always)]
fn compute_ra_evals<F: JoltField>(trace: &[RV32IMCycle], eq_r_cycle: &[F]) -> [Vec<F>; D] {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: [Vec<F>; D] =
                std::array::from_fn(|_| unsafe_allocate_zero_vec(K_CHUNK));
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut lookup_index = LookupQuery::<WORD_SIZE>::to_lookup_index(cycle);
                for i in (0..D).rev() {
                    let k = lookup_index % K_CHUNK as u64;
                    result[i][k as usize] += eq_r_cycle[j];
                    lookup_index >>= LOG_K_CHUNK;
                }
                j += 1;
            }
            result
        })
        .reduce(
            || std::array::from_fn(|_| unsafe_allocate_zero_vec(K_CHUNK)),
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running
            },
        )
}

#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId},
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
        BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
        HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        dag::{
            stage::{SumcheckStagesProver, SumcheckStagesVerifier},
            state_manager::StateManager,
        },
        instruction::LookupQuery,
        instruction_lookups::{
            ra_virtual::{RaSumcheckProver, RaSumcheckVerifier},
            read_raf_checking::{ReadRafSumcheckProver, ReadRafSumcheckVerifier},
        },
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use common::constants::XLEN;
use rayon::prelude::*;
use tracer::instruction::Cycle;
pub mod ra_virtual;
pub mod read_raf_checking;

const LOG_K: usize = XLEN * 2;
const PHASES: usize = 8;
pub const LOG_M: usize = LOG_K / PHASES;
const M: usize = 1 << LOG_M;
pub const D: usize = 16;
pub const LOG_K_CHUNK: usize = LOG_K / D;
pub const K_CHUNK: usize = 1 << LOG_K_CHUNK;

pub struct LookupsDagProver<F: JoltField> {
    // Generated after stage 1 (uses r_cycle from spartan sumcheck).
    ra_evals: Option<[Vec<F>; D]>,
}

impl<F: JoltField> LookupsDagProver<F> {
    pub fn new() -> Self {
        Self { ra_evals: None }
    }
}

impl<F: JoltField> LookupsDagProver<F> {
    fn get_or_compute_ra_evals(
        &mut self,
        sm: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> &[Vec<F>; D] {
        &*self.ra_evals.get_or_insert_with(|| compute_ra_evals(sm))
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStagesProver<F, T, PCS>
    for LookupsDagProver<F>
{
    fn stage3_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let ra_evals = self.get_or_compute_ra_evals(sm);
        let hamming_weight = gen_ra_hamming_weight_prover(sm, ra_evals);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "Instruction execution HammingWeightSumcheck",
                &hamming_weight,
            );
        }

        vec![Box::new(hamming_weight)]
    }

    fn stage5_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let read_raf = ReadRafSumcheckProver::gen(sm);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Instruction execution ReadRafSumcheck", &read_raf);
        }

        vec![Box::new(read_raf)]
    }

    fn stage6_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let ra_virtual = RaSumcheckProver::gen(sm);

        let ra_evals = self.get_or_compute_ra_evals(sm);
        let booleanity = gen_ra_booleanity_prover(sm, ra_evals);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "Instruction execution RAVirtual sumcheck",
                &ra_virtual,
            );
            print_data_structure_heap_usage(
                "Instruction execution BooleanitySumcheck",
                &booleanity,
            );
        }

        vec![Box::new(ra_virtual), Box::new(booleanity)]
    }
}

pub struct LookupsDagVerifier;

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript>
    SumcheckStagesVerifier<F, T, PCS> for LookupsDagVerifier
{
    fn stage3_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let hamming_weight = new_ra_hamming_weight_verifier(sm);
        vec![Box::new(hamming_weight)]
    }

    fn stage5_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let read_raf = ReadRafSumcheckVerifier::new(sm);
        vec![Box::new(read_raf)]
    }

    fn stage6_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let ra_virtual = RaSumcheckVerifier::new(sm);
        let booleanity = new_ra_booleanity_verifier(sm);
        vec![Box::new(ra_virtual), Box::new(booleanity)]
    }
}

fn gen_ra_booleanity_prover<F: JoltField>(
    state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ra_evals: &[Vec<F>; D],
) -> BooleanitySumcheckProver<F> {
    let (_, _, trace, _, _) = state_manager.get_prover_data();
    let (r_cycle, _) = state_manager
        .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter);
    let H_indices = compute_instruction_h_indices(trace);

    let log_t = trace.len().log_2();

    let gammas = state_manager
        .transcript
        .borrow_mut()
        .challenge_vector_optimized::<F>(D);

    let r_address: Vec<F::Challenge> = state_manager
        .transcript
        .borrow_mut()
        .challenge_vector_optimized::<F>(LOG_K_CHUNK);
    let polynomial_types: Vec<CommittedPolynomial> =
        (0..D).map(CommittedPolynomial::InstructionRa).collect();

    let params = BooleanitySumcheckParams {
        d: D,
        log_k_chunk: LOG_K_CHUNK,
        log_t,
        r_cycle: r_cycle.r.clone(),
        r_address,
        gammas,
        polynomial_types,
        sumcheck_id: SumcheckId::InstructionBooleanity,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
    };

    BooleanitySumcheckProver::gen(params, ra_evals.to_vec(), H_indices)
}

fn gen_ra_hamming_weight_prover<F: JoltField>(
    state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ra_evals: &[Vec<F>; D],
) -> HammingWeightSumcheckProver<F> {
    let gamma_powers = state_manager
        .transcript
        .borrow_mut()
        .challenge_scalar_powers(D);

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..D).map(CommittedPolynomial::InstructionRa).collect();

    let params = HammingWeightSumcheckParams {
        d: D,
        num_rounds: LOG_K_CHUNK,
        gamma_powers,
        polynomial_types,
        sumcheck_id: SumcheckId::InstructionHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    HammingWeightSumcheckProver::gen(params, ra_evals.to_vec())
}

fn new_ra_booleanity_verifier<F: JoltField>(
    state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
) -> BooleanitySumcheckVerifier<F> {
    let (_, _, T_val) = state_manager.get_verifier_data();
    let log_t = T_val.log_2();

    let gammas: Vec<F::Challenge> = state_manager
        .transcript
        .borrow_mut()
        .challenge_vector_optimized::<F>(D);

    let r_address: Vec<F::Challenge> = state_manager
        .transcript
        .borrow_mut()
        .challenge_vector_optimized::<F>(LOG_K_CHUNK);

    let r_cycle = Vec::new();

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..D).map(CommittedPolynomial::InstructionRa).collect();

    let params = BooleanitySumcheckParams {
        d: D,
        log_k_chunk: LOG_K_CHUNK,
        log_t,
        gammas,
        r_address,
        r_cycle,
        polynomial_types,
        sumcheck_id: SumcheckId::InstructionBooleanity,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
    };

    BooleanitySumcheckVerifier::new(params)
}

fn new_ra_hamming_weight_verifier<F: JoltField>(
    state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
) -> HammingWeightSumcheckVerifier<F> {
    let gamma_powers = state_manager
        .transcript
        .borrow_mut()
        .challenge_scalar_powers(D);

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..D).map(CommittedPolynomial::InstructionRa).collect();

    let params = HammingWeightSumcheckParams {
        d: D,
        num_rounds: LOG_K_CHUNK,
        gamma_powers,
        polynomial_types,
        sumcheck_id: SumcheckId::InstructionHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    HammingWeightSumcheckVerifier::new(params)
}

#[tracing::instrument(skip_all, name = "instruction_lookups::compute_instruction_h_indices")]
fn compute_instruction_h_indices(trace: &[Cycle]) -> Vec<Vec<Option<u8>>> {
    (0..D)
        .map(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                    Some(((lookup_index >> (LOG_K_CHUNK * (D - 1 - i))) % K_CHUNK as u128) as u8)
                })
                .collect()
        })
        .collect()
}

#[inline(always)]
#[tracing::instrument(skip_all, name = "instruction_lookups::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(
    state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
) -> [Vec<F>; D] {
    let (_, _, trace, _, _) = state_manager.get_prover_data();
    let (r_cycle, _) = state_manager
        .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter);
    let eq_r_cycle = EqPolynomial::evals(&r_cycle.r);

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
                let mut lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                for i in (0..D).rev() {
                    let k = lookup_index % K_CHUNK as u128;
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

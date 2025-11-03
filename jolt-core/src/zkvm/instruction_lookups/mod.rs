use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{OpeningAccumulator, ProverOpeningAccumulator, SumcheckId},
    },
    subprotocols::{
        BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
        HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        instruction::LookupQuery,
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

pub fn gen_ra_one_hot_provers<F: JoltField>(
    trace: &[Cycle],
    opening_accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> (HammingWeightSumcheckProver<F>, BooleanitySumcheckProver<F>) {
    let ra_evals = compute_ra_evals(trace, opening_accumulator);

    let gamma_powers = transcript.challenge_scalar_powers(D);

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..D).map(CommittedPolynomial::InstructionRa).collect();

    let hamming_weight_params = HammingWeightSumcheckParams {
        d: D,
        num_rounds: LOG_K_CHUNK,
        gamma_powers,
        polynomial_types,
        sumcheck_id: SumcheckId::InstructionHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    let (r_cycle, _) = opening_accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter);
    let H_indices = compute_instruction_h_indices(trace);

    let log_t = trace.len().log_2();

    let gammas = transcript.challenge_vector_optimized::<F>(D);

    let r_address = transcript.challenge_vector_optimized::<F>(LOG_K_CHUNK);
    let polynomial_types: Vec<CommittedPolynomial> =
        (0..D).map(CommittedPolynomial::InstructionRa).collect();

    let booleanity_params = BooleanitySumcheckParams {
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

    (
        HammingWeightSumcheckProver::gen(hamming_weight_params, ra_evals.to_vec()),
        BooleanitySumcheckProver::gen(booleanity_params, ra_evals.to_vec(), H_indices),
    )
}

pub fn new_ra_one_hot_verifiers<F: JoltField>(
    n_cycle_vars: usize,
    transcript: &mut impl Transcript,
) -> (
    HammingWeightSumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let gamma_powers = transcript.challenge_scalar_powers(D);

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..D).map(CommittedPolynomial::InstructionRa).collect();

    let hamming_weight_params = HammingWeightSumcheckParams {
        d: D,
        num_rounds: LOG_K_CHUNK,
        gamma_powers,
        polynomial_types,
        sumcheck_id: SumcheckId::InstructionHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    let gammas = transcript.challenge_vector_optimized::<F>(D);
    let r_address = transcript.challenge_vector_optimized::<F>(LOG_K_CHUNK);
    let r_cycle = Vec::new();
    let polynomial_types: Vec<CommittedPolynomial> =
        (0..D).map(CommittedPolynomial::InstructionRa).collect();
    let booleanity_params = BooleanitySumcheckParams {
        d: D,
        log_k_chunk: LOG_K_CHUNK,
        log_t: n_cycle_vars,
        gammas,
        r_address,
        r_cycle,
        polynomial_types,
        sumcheck_id: SumcheckId::InstructionBooleanity,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
    };

    (
        HammingWeightSumcheckVerifier::new(hamming_weight_params),
        BooleanitySumcheckVerifier::new(booleanity_params),
    )
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

#[tracing::instrument(skip_all, name = "instruction_lookups::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(
    trace: &[Cycle],
    opening_accumulator: &ProverOpeningAccumulator<F>,
) -> [Vec<F>; D] {
    let (r_cycle, _) = opening_accumulator
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

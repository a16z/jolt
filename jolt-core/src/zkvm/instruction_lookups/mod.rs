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
        config::OneHotParams,
        instruction::LookupQuery,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use common::constants::XLEN;
use rayon::prelude::*;
use tracer::instruction::Cycle;
pub mod ra_virtual;
pub mod read_raf_checking;

pub const LOG_K: usize = XLEN * 2;

// TODO: transition read_raf_checking to use dynamic phase configuration from `config`.
// pub const PHASES: usize = 8;
// pub const LOG_M: usize = LOG_K / PHASES;
// const M: usize = 1 << LOG_M;

pub fn gen_ra_one_hot_provers<F: JoltField>(
    trace: &[Cycle],
    one_hot_params: &OneHotParams,
    opening_accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> (HammingWeightSumcheckProver<F>, BooleanitySumcheckProver<F>) {
    let ra_evals = compute_ra_evals(trace, one_hot_params, opening_accumulator);

    let gamma_powers = transcript.challenge_scalar_powers(one_hot_params.instruction_d);

    let polynomial_types: Vec<CommittedPolynomial> = (0..one_hot_params.instruction_d)
        .map(CommittedPolynomial::InstructionRa)
        .collect();

    let hamming_weight_params = HammingWeightSumcheckParams {
        d: one_hot_params.instruction_d,
        num_rounds: one_hot_params.log_k_chunk,
        gamma_powers,
        polynomial_types: polynomial_types.clone(),
        sumcheck_id: SumcheckId::InstructionHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    let (r_cycle, _) = opening_accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter);
    let H_indices = compute_instruction_h_indices(trace, one_hot_params);

    let log_t = trace.len().log_2();

    let gammas = transcript.challenge_vector_optimized::<F>(one_hot_params.instruction_d);

    let r_address = transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);

    let booleanity_params = BooleanitySumcheckParams {
        d: one_hot_params.instruction_d,
        log_k_chunk: one_hot_params.log_k_chunk,
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
    one_hot_params: &OneHotParams,
    transcript: &mut impl Transcript,
) -> (
    HammingWeightSumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let gamma_powers = transcript.challenge_scalar_powers(one_hot_params.instruction_d);

    let polynomial_types: Vec<CommittedPolynomial> = (0..one_hot_params.instruction_d)
        .map(CommittedPolynomial::InstructionRa)
        .collect();

    let hamming_weight_params = HammingWeightSumcheckParams {
        d: one_hot_params.instruction_d,
        num_rounds: one_hot_params.log_k_chunk,
        gamma_powers,
        polynomial_types: polynomial_types.clone(),
        sumcheck_id: SumcheckId::InstructionHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    let gammas = transcript.challenge_vector_optimized::<F>(one_hot_params.instruction_d);
    let r_address = transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);
    let r_cycle = Vec::new();
    let booleanity_params = BooleanitySumcheckParams {
        d: one_hot_params.instruction_d,
        log_k_chunk: one_hot_params.log_k_chunk,
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
fn compute_instruction_h_indices(
    trace: &[Cycle],
    one_hot_params: &OneHotParams,
) -> Vec<Vec<Option<u16>>> {
    (0..one_hot_params.instruction_d)
        .map(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                    Some(one_hot_params.lookup_index_chunk(lookup_index, i))
                })
                .collect()
        })
        .collect()
}

#[tracing::instrument(skip_all, name = "instruction_lookups::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(
    trace: &[Cycle],
    one_hot_params: &OneHotParams,
    opening_accumulator: &ProverOpeningAccumulator<F>,
) -> Vec<Vec<F>> {
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
            let mut result: Vec<Vec<F>> = (0..one_hot_params.instruction_d)
                .map(|_| unsafe_allocate_zero_vec(one_hot_params.k_chunk))
                .collect();
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                for i in 0..one_hot_params.instruction_d {
                    let k = one_hot_params.lookup_index_chunk(lookup_index, i);
                    result[i][k as usize] += eq_r_cycle[j];
                }
                j += 1;
            }
            result
        })
        .reduce(
            || {
                (0..one_hot_params.instruction_d)
                    .map(|_| unsafe_allocate_zero_vec(one_hot_params.k_chunk))
                    .collect()
            },
            |mut running, new| {
                running.iter_mut().zip(new.into_iter()).for_each(|(x, y)| {
                    x.par_iter_mut()
                        .zip(y.into_par_iter())
                        .for_each(|(x, y)| *x += y)
                });
                running
            },
        )
}

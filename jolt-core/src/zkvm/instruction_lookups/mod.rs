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
        config,
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

pub fn log_k() -> usize {
    LOG_K
}

pub fn phases() -> usize {
    config::params().instruction.phases
}

pub fn log_m() -> usize {
    config::params().instruction.log_m
}

pub fn m() -> usize {
    config::params().instruction.m
}

pub fn d() -> usize {
    config::params().instruction.d
}

pub fn log_k_chunk() -> usize {
    config::params().instruction.log_k_chunk
}

pub fn k_chunk() -> usize {
    config::params().instruction.k_chunk
}

// TODO: transition read_raf_checking to use dynamic phase configuration from `config`.
pub const PHASES: usize = 8;
pub const LOG_M: usize = LOG_K / PHASES;
const M: usize = 1 << LOG_M;

pub fn gen_ra_one_hot_provers<F: JoltField>(
    trace: &[Cycle],
    opening_accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> (HammingWeightSumcheckProver<F>, BooleanitySumcheckProver<F>) {
    let d = d();
    let log_k_chunk = log_k_chunk();
    let ra_evals = compute_ra_evals(trace, opening_accumulator);

    let gamma_powers = transcript.challenge_scalar_powers(d);

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..d).map(CommittedPolynomial::InstructionRa).collect();

    let hamming_weight_params = HammingWeightSumcheckParams {
        d,
        num_rounds: log_k_chunk,
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

    let gammas = transcript.challenge_vector_optimized::<F>(d);

    let r_address = transcript.challenge_vector_optimized::<F>(log_k_chunk);
    let polynomial_types: Vec<CommittedPolynomial> =
        (0..d).map(CommittedPolynomial::InstructionRa).collect();

    let booleanity_params = BooleanitySumcheckParams {
        d,
        log_k_chunk,
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
    let d = d();
    let log_k_chunk = log_k_chunk();
    let gamma_powers = transcript.challenge_scalar_powers(d);

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..d).map(CommittedPolynomial::InstructionRa).collect();

    let hamming_weight_params = HammingWeightSumcheckParams {
        d,
        num_rounds: log_k_chunk,
        gamma_powers,
        polynomial_types,
        sumcheck_id: SumcheckId::InstructionHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    let gammas = transcript.challenge_vector_optimized::<F>(d);
    let r_address = transcript.challenge_vector_optimized::<F>(log_k_chunk);
    let r_cycle = Vec::new();
    let polynomial_types: Vec<CommittedPolynomial> =
        (0..d).map(CommittedPolynomial::InstructionRa).collect();
    let booleanity_params = BooleanitySumcheckParams {
        d,
        log_k_chunk,
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
fn compute_instruction_h_indices(trace: &[Cycle]) -> Vec<Vec<Option<u16>>> {
    let d = d();
    let log_k_chunk = log_k_chunk();
    let k_chunk = k_chunk() as u128;

    (0..d)
        .map(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                    Some(((lookup_index >> (log_k_chunk * (d - 1 - i))) % k_chunk) as u16)
                })
                .collect()
        })
        .collect()
}

#[tracing::instrument(skip_all, name = "instruction_lookups::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(
    trace: &[Cycle],
    opening_accumulator: &ProverOpeningAccumulator<F>,
) -> Vec<Vec<F>> {
    let (r_cycle, _) = opening_accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter);
    let eq_r_cycle = EqPolynomial::evals(&r_cycle.r);

    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    let d = d();
    let log_k_chunk = log_k_chunk();
    let k_chunk = k_chunk();

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: Vec<Vec<F>> =
                (0..d).map(|_| unsafe_allocate_zero_vec(k_chunk)).collect();
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                for i in (0..d).rev() {
                    let k = lookup_index % k_chunk as u128;
                    result[i][k as usize] += eq_r_cycle[j];
                    lookup_index >>= log_k_chunk;
                }
                j += 1;
            }
            result
        })
        .reduce(
            || (0..d).map(|_| unsafe_allocate_zero_vec(k_chunk)).collect(),
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

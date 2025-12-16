//! RAM checking module for Jolt.
//!
//! This module implements the RAM checking protocol, which verifies that memory
//! reads and writes are consistent throughout program execution.
//!
//! # Important: Sumcheck Stage Constraints
//!
//! The RAM RA reduction sumcheck (`ra_reduction.rs`) consolidates four RA claims
//! into a single claim. For this to work correctly, certain challenge coincidences
//! must hold. These coincidences are guaranteed by the following constraints on
//! how sumchecks are batched:
//!
//! ## Required Coincidences
//!
//! The four RA claims use these opening points:
//!
//! | Sumcheck | Opening Point | Stage |
//! |----------|---------------|-------|
//! | RamReadWriteChecking | `ra(r_address_rw, r_cycle_rw)` | Stage 2 |
//! | RamRafEvaluation | `ra(r_address_raf, r_cycle_raf)` | Stage 2 |
//! | RamValEvaluation | `ra(r_address_rw, r_cycle_val)` | Stage 4 |
//! | RamValFinal | `ra(r_address_raf, r_cycle_val)` | Stage 4 |
//!
//! The following equalities must hold:
//! - `r_address_raf = r_address_val_final`
//! - `r_address_val_eval = r_address_rw`
//! - `r_cycle_val_eval = r_cycle_val_final`
//!
//! ## Constraints to Ensure Coincidences
//!
//! **These constraints MUST be maintained when modifying the prover/verifier:**
//!
//! 1. **OutputCheck and RafEvaluation** MUST be in the same batched sumcheck (currently Stage 2),
//!    and both must use challenges `[0 .. log_K]` for `r_address`.
//!
//! 2. **ValEvaluation and ValFinal** MUST be in the same batched sumcheck (currently Stage 4),
//!    and have the same `num_rounds = log_T`.
//!
//! 3. **ValEvaluation** MUST read `r_address` from RamReadWriteChecking's opening.
//!
//! 4. **ValFinal** MUST read `r_address` from OutputCheck's opening.
//!
//! Violating these constraints will cause the RA reduction sumcheck to fail with
//! mismatched challenge vectors.

#![allow(clippy::too_many_arguments)]

use crate::subprotocols::{
    BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
    HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
};
use crate::zkvm::config::OneHotParams;
use crate::{
    field::{self, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};
use std::vec;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{
    constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS},
    jolt_device::MemoryLayout,
};
use rayon::prelude::*;
use tracer::emulator::memory::Memory;
use tracer::instruction::Cycle;
use tracer::JoltDevice;

pub mod hamming_booleanity;
pub mod output_check;
pub mod ra_reduction;
pub mod ra_virtual;
pub mod raf_evaluation;
pub mod read_write_checking;
pub mod val_evaluation;
pub mod val_final;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMPreprocessing {
    pub min_bytecode_address: u64,
    pub bytecode_words: Vec<u64>,
}

impl RAMPreprocessing {
    pub fn preprocess(memory_init: Vec<(u64, u8)>) -> Self {
        let min_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1);

        let num_words = max_bytecode_address.next_multiple_of(8) / 8 - min_bytecode_address / 8 + 1;
        let mut bytecode_words = vec![0u64; num_words as usize];
        // Convert bytes into words and populate `bytecode_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 8 == address_b / 8)
        {
            let mut word = [0u8; 8];
            for (address, byte) in chunk {
                word[(address % 8) as usize] = *byte;
            }
            let word = u64::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 8 - min_bytecode_address / 8) as usize;
            bytecode_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            bytecode_words,
        }
    }
}

/// Returns Some(address) if there was read/write
/// Returns None if there was no read/write
pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> Option<u64> {
    if address == 0 {
        return None;
    }

    let lowest_address = memory_layout.get_lowest_address();
    if address >= lowest_address {
        Some((address - lowest_address) / 8)
    } else {
        panic!("Unexpected address {address}")
    }
}

/// Populate memory states
///
/// # Arguments
/// * `index` - Where to start writing
/// * `bytes` - Input bytes to be written
/// * `initial_state` - Memory state at program start
/// * `final_state` - Memory state at program end
pub fn populate_memory_states(
    mut index: usize,
    bytes: &[u8],
    mut initial_state: Option<&mut Vec<u64>>,
    mut final_state: Option<&mut Vec<u64>>,
) {
    for chunk in bytes.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        let word = u64::from_le_bytes(word);

        if let Some(ref mut initial) = initial_state {
            initial[index] = word;
        }
        if let Some(ref mut final_st) = final_state {
            final_st[index] = word;
        }
        index += 1;
    }
}

/// Accumulates advice polynomials (trusted and untrusted) into the prover's accumulator.
///
/// When `single_opening` is true (all cycle vars bound in phase 1):
/// - Only opens at `r_address_rw` (the two points are identical)
///
/// Otherwise opens at TWO points:
/// 1. `r_address_rw` from `RamVal`/`RamReadWriteChecking` - used by `ValEvaluationSumcheck`
/// 2. `r_address_raf` from `RamValFinal`/`RamOutputCheck` - used by `ValFinalSumcheck`
pub fn prover_accumulate_advice<F: JoltField>(
    untrusted_advice_polynomial: &Option<MultilinearPolynomial<F>>,
    trusted_advice_polynomial: &Option<MultilinearPolynomial<F>>,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
    single_opening: bool,
) {
    let total_variables = one_hot_params.ram_k.log_2();

    // Get r_address_rw from RamVal/RamReadWriteChecking (used by ValEvaluation)
    let (r_rw, _) = opening_accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RamVal,
        SumcheckId::RamReadWriteChecking,
    );
    let (r_address_rw, _) = r_rw.split_at(total_variables);

    let compute_advice_opening = |advice_poly: &MultilinearPolynomial<F>,
                                  r_address: &OpeningPoint<BIG_ENDIAN, F>,
                                  max_advice_size: usize| {
        let advice_variables = (max_advice_size / 8).next_power_of_two().log_2();
        let eval = advice_poly.evaluate(&r_address.r[total_variables - advice_variables..]);
        let mut advice_point = r_address.clone();
        advice_point.r = r_address.r[total_variables - advice_variables..].to_vec();
        (advice_point, eval)
    };

    if let Some(ref untrusted_advice_poly) = untrusted_advice_polynomial {
        let max_size = memory_layout.max_untrusted_advice_size as usize;

        // Opening at r_address_rw (for ValEvaluation)
        let (point_rw, eval_rw) =
            compute_advice_opening(untrusted_advice_poly, &r_address_rw, max_size);
        opening_accumulator.append_untrusted_advice(
            transcript,
            SumcheckId::RamValEvaluation,
            point_rw,
            eval_rw,
        );

        // Opening at r_address_raf (for ValFinalEvaluation) - only if points differ
        if !single_opening {
            let (r_raf, _) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            );
            let (point_raf, eval_raf) =
                compute_advice_opening(untrusted_advice_poly, &r_raf, max_size);
            opening_accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::RamValFinalEvaluation,
                point_raf,
                eval_raf,
            );
        }
    }

    if let Some(ref trusted_advice_poly) = trusted_advice_polynomial {
        let max_size = memory_layout.max_trusted_advice_size as usize;

        // Opening at r_address_rw (for ValEvaluation)
        let (point_rw, eval_rw) =
            compute_advice_opening(trusted_advice_poly, &r_address_rw, max_size);
        opening_accumulator.append_trusted_advice(
            transcript,
            SumcheckId::RamValEvaluation,
            point_rw,
            eval_rw,
        );

        // Opening at r_address_raf (for ValFinalEvaluation) - only if points differ
        if !single_opening {
            let (r_raf, _) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            );
            let (point_raf, eval_raf) =
                compute_advice_opening(trusted_advice_poly, &r_raf, max_size);
            opening_accumulator.append_trusted_advice(
                transcript,
                SumcheckId::RamValFinalEvaluation,
                point_raf,
                eval_raf,
            );
        }
    }
}

/// Accumulates advice commitments into the verifier's accumulator.
///
/// When `single_opening` is true (all cycle vars bound in phase 1):
/// - Only opens at `r_address_rw` (the two points are identical)
///
/// Otherwise opens at TWO points:
/// 1. `r_address_rw` from `RamVal`/`RamReadWriteChecking` - used by `ValEvaluationSumcheck`
/// 2. `r_address_raf` from `RamValFinal`/`RamOutputCheck` - used by `ValFinalSumcheck`
pub fn verifier_accumulate_advice<F: JoltField>(
    ram_K: usize,
    program_io: &JoltDevice,
    has_untrusted_advice_commitment: bool,
    has_trusted_advice_commitment: bool,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
    single_opening: bool,
) {
    let total_vars = ram_K.log_2();

    // Get r_address_rw from RamVal/RamReadWriteChecking (used by ValEvaluation)
    let (r_rw, _) = opening_accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RamVal,
        SumcheckId::RamReadWriteChecking,
    );
    let (r_address_rw, _) = r_rw.split_at(total_vars);

    let compute_advice_point = |r_address: &OpeningPoint<BIG_ENDIAN, F>, max_advice_size: usize| {
        let advice_variables = (max_advice_size / 8).next_power_of_two().log_2();
        let mut advice_point = r_address.clone();
        advice_point.r = r_address.r[total_vars - advice_variables..].to_vec();
        advice_point
    };

    if has_untrusted_advice_commitment {
        let max_size = program_io.memory_layout.max_untrusted_advice_size as usize;

        // Opening at r_address_rw (for ValEvaluation)
        let point_rw = compute_advice_point(&r_address_rw, max_size);
        opening_accumulator.append_untrusted_advice(
            transcript,
            SumcheckId::RamValEvaluation,
            point_rw,
        );

        // Opening at r_address_raf (for ValFinalEvaluation) - only if points differ
        if !single_opening {
            let (r_raf, _) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            );
            let point_raf = compute_advice_point(&r_raf, max_size);
            opening_accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::RamValFinalEvaluation,
                point_raf,
            );
        }
    }

    if has_trusted_advice_commitment {
        let max_size = program_io.memory_layout.max_trusted_advice_size as usize;

        // Opening at r_address_rw (for ValEvaluation)
        let point_rw = compute_advice_point(&r_address_rw, max_size);
        opening_accumulator.append_trusted_advice(
            transcript,
            SumcheckId::RamValEvaluation,
            point_rw,
        );

        // Opening at r_address_raf (for ValFinalEvaluation) - only if points differ
        if !single_opening {
            let (r_raf, _) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            );
            let point_raf = compute_advice_point(&r_raf, max_size);
            opening_accumulator.append_trusted_advice(
                transcript,
                SumcheckId::RamValFinalEvaluation,
                point_raf,
            );
        }
    }
}

/// Calculates how advice inputs contribute to the evaluation of initial_ram_state at a given random point.
///
/// ## Example with Two Commitments:
///
/// Consider an 8192-element initial_ram_state (l=13) with two advice commitments:
///
/// ### trusted_advice: Block size 1024, starting at index 0
/// - **Parameters**:
///   - l = 13 (total memory has 2^13 = 8192 elements)
///   - B1 = 1024 -> b1 = 10 (block has 2^10 elements)
///   - Selector variables: d1 = 13 - 10 = 3 (uses x1, x2, x3)
///   - Starting index: 0
///
/// - **Binary Prefix**:
///   - Index 0 in 13-bit binary: 0000000000000
///   - Prefix (first d1 = 3 bits): 000
///
/// - **Selector Polynomial**:
///   - (1 - x1)(1 - x2)(1 - x3)
///   - This evaluates to 1 when x1 = x2 = x3 = 0, selecting the region starting at 0
///
/// ### untrusted_advice: Block size 512, starting at index 1024
/// - **Parameters**:
///   - l = 13
///   - B2 = 512 -> b2 = 9 (block has 2^9 elements)
///   - Selector variables: d2 = 13 - 9 = 4 (uses x1, x2, x3, x4)
///   - Starting index: 1024
///
/// - **Binary Prefix**:
///   - Index 1024 in 13-bit binary: 0010000000000
///   - Prefix (first d2 = 4 bits): 0010
///
/// - **Selector Polynomial**:
///   - (1 - x1)(1 - x2)x3(1 - x4)
///   - This evaluates to 1 when x1 = 0, x2 = 0, x3 = 1, x4 = 0, selecting the region at 1024
///
/// # Parameters
///
/// * `advice_opening` - Optional tuple of opening point and evaluation at that point
/// * `advice_num_vars` - Number of variables in the advice polynomial (b in the explanation)
/// * `advice_start` - Starting index of the advice block in memory
/// * `memory_layout` - Memory layout for address remapping
/// * `r_address` - Challenge points from verifier (used for selector polynomial evaluation)
/// * `total_memory_vars` - Total number of variables for the entire memory space (l in the explanation)
///
/// # Returns
///
/// The scaled evaluation: `eval * scaling_factor`, where the scaling factor is the selector polynomial
/// evaluated at the challenge point. Returns zero if no advice opening is provided.
fn calculate_advice_memory_evaluation<F: JoltField>(
    advice_opening: Option<(OpeningPoint<BIG_ENDIAN, F>, F)>,
    advice_num_vars: usize,
    advice_start: u64,
    memory_layout: &MemoryLayout,
    r_address: &[<F as field::JoltField>::Challenge],
    total_memory_vars: usize,
) -> F {
    if let Some((_, eval)) = advice_opening {
        let num_missing_vars = total_memory_vars - advice_num_vars;

        let index = remap_address(advice_start, memory_layout).unwrap();
        let mut scaling_factor = F::one();

        // Convert index to binary representation with total_memory_vars bits.
        // For example, if index=5 and total_memory_vars=4, we get [0,1,0,1].
        let index_binary: Vec<bool> = (0..total_memory_vars)
            .rev()
            .map(|i| (index >> i) & 1 == 1)
            .collect();

        let selector_bits = &index_binary[0..num_missing_vars];

        // Each bit determines whether to use r[i] (bit=1) or (1-r[i]) (bit=0).
        for (i, &bit) in selector_bits.iter().enumerate() {
            scaling_factor *= if bit {
                r_address[i].into()
            } else {
                F::one() - r_address[i]
            };
        }
        eval * scaling_factor
    } else {
        F::zero()
    }
}

/// Returns `(initial_memory_state, final_memory_state)`
pub fn gen_ram_memory_states<F: JoltField>(
    ram_K: usize,
    ram_preprocessing: &RAMPreprocessing,
    program_io: &JoltDevice,
    final_memory: &Memory,
) -> (Vec<u64>, Vec<u64>) {
    let K = ram_K;

    let mut initial_memory_state: Vec<u64> = vec![0; K];
    // Copy bytecode
    let mut index = remap_address(
        ram_preprocessing.min_bytecode_address,
        &program_io.memory_layout,
    )
    .unwrap() as usize;
    for word in &ram_preprocessing.bytecode_words {
        initial_memory_state[index] = *word;
        index += 1;
    }

    let dram_start_index =
        remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap() as usize;
    let mut final_memory_state: Vec<u64> = vec![0; K];
    // Note that `final_memory` only contains memory at addresses >= `RAM_START_ADDRESS`
    // so we will still need to populate `final_memory_state` with the contents of
    // `program_io`, which lives at addresses < `RAM_START_ADDRESS`
    let final_memory_words = final_memory.data.len().min(K - dram_start_index);
    final_memory_state[dram_start_index..dram_start_index + final_memory_words]
        .par_iter_mut()
        .enumerate()
        .for_each(|(k, word)| {
            *word = final_memory.read_doubleword(8 * k as u64);
        });

    index = remap_address(
        program_io.memory_layout.trusted_advice_start,
        &program_io.memory_layout,
    )
    .unwrap() as usize;
    populate_memory_states(
        index,
        &program_io.trusted_advice,
        Some(&mut initial_memory_state),
        Some(&mut final_memory_state),
    );

    index = remap_address(
        program_io.memory_layout.untrusted_advice_start,
        &program_io.memory_layout,
    )
    .unwrap() as usize;
    populate_memory_states(
        index,
        &program_io.untrusted_advice,
        Some(&mut initial_memory_state),
        Some(&mut final_memory_state),
    );

    index = remap_address(
        program_io.memory_layout.input_start,
        &program_io.memory_layout,
    )
    .unwrap() as usize;
    populate_memory_states(
        index,
        &program_io.inputs,
        Some(&mut initial_memory_state),
        Some(&mut final_memory_state),
    );

    // Convert output bytes into words and populate
    // `final_memory_state`
    index = remap_address(
        program_io.memory_layout.output_start,
        &program_io.memory_layout,
    )
    .unwrap() as usize;
    populate_memory_states(
        index,
        &program_io.outputs,
        None,
        Some(&mut final_memory_state),
    );

    // Copy panic bit
    let panic_index =
        remap_address(program_io.memory_layout.panic, &program_io.memory_layout).unwrap() as usize;
    final_memory_state[panic_index] = program_io.panic as u64;
    if !program_io.panic {
        // Set termination bit
        let termination_index = remap_address(
            program_io.memory_layout.termination,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        final_memory_state[termination_index] = 1;
    }

    (initial_memory_state, final_memory_state)
}

pub fn gen_ram_initial_memory_state<F: JoltField>(
    ram_K: usize,
    ram_preprocessing: &RAMPreprocessing,
    program_io: &JoltDevice,
) -> Vec<u64> {
    let mut initial_memory_state = vec![0; ram_K];
    // Copy bytecode
    let mut index = remap_address(
        ram_preprocessing.min_bytecode_address,
        &program_io.memory_layout,
    )
    .unwrap() as usize;
    for word in &ram_preprocessing.bytecode_words {
        initial_memory_state[index] = *word;
        index += 1;
    }

    index = remap_address(
        program_io.memory_layout.input_start,
        &program_io.memory_layout,
    )
    .unwrap() as usize;
    // Convert input bytes into words and populate
    // `initial_memory_state`
    for chunk in program_io.inputs.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        let word = u64::from_le_bytes(word);
        initial_memory_state[index] = word;
        index += 1;
    }

    initial_memory_state
}

pub fn ra_booleanity_params<F: JoltField>(
    trace_len: usize,
    one_hot_params: &OneHotParams,
    transcript: &mut impl Transcript,
) -> BooleanitySumcheckParams<F> {
    let r_cycle = transcript.challenge_vector_optimized::<F>(trace_len.log_2());
    let r_address = transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);
    let polynomial_types: Vec<CommittedPolynomial> = (0..one_hot_params.ram_d)
        .map(CommittedPolynomial::RamRa)
        .collect();
    let gammas: Vec<F::Challenge> =
        transcript.challenge_vector_optimized::<F>(one_hot_params.ram_d);

    BooleanitySumcheckParams {
        d: one_hot_params.ram_d,
        log_k_chunk: one_hot_params.log_k_chunk,
        log_t: trace_len.log_2(),
        gammas,
        r_address,
        r_cycle,
        polynomial_types,
        sumcheck_id: SumcheckId::RamBooleanity,
    }
}

pub fn gen_ra_booleanity_prover<F: JoltField>(
    params: BooleanitySumcheckParams<F>,
    trace: &[Cycle],
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
) -> BooleanitySumcheckProver<F> {
    // Compute G and H for RAM
    let eq_r_cycle = EqPolynomial::<F>::evals(&params.r_cycle);
    let G = compute_ram_ra_evals(trace, memory_layout, &eq_r_cycle, one_hot_params);
    let H_indices = compute_ram_h_indices(trace, memory_layout, one_hot_params);
    BooleanitySumcheckProver::gen(params, G, H_indices)
}

pub fn ra_hamming_weight_params<F: JoltField>(
    one_hot_params: &OneHotParams,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> HammingWeightSumcheckParams<F> {
    let r_cycle = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        )
        .0
        .r;

    let gamma_powers = transcript.challenge_scalar_powers(one_hot_params.ram_d);

    let polynomial_types: Vec<CommittedPolynomial> = (0..one_hot_params.ram_d)
        .map(CommittedPolynomial::RamRa)
        .collect();

    HammingWeightSumcheckParams {
        d: one_hot_params.ram_d,
        num_rounds: one_hot_params.log_k_chunk,
        gamma_powers,
        polynomial_types,
        sumcheck_id: SumcheckId::RamHammingWeight,
        r_cycle,
    }
}

pub fn gen_ra_hamming_weight_prover<F: JoltField>(
    params: HammingWeightSumcheckParams<F>,
    trace: &[Cycle],
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
) -> HammingWeightSumcheckProver<F> {
    let eq_r_cycle = EqPolynomial::evals(&params.r_cycle);
    let G = compute_ram_ra_evals(trace, memory_layout, &eq_r_cycle, one_hot_params);

    HammingWeightSumcheckProver::gen(params, G)
}

pub fn new_ra_booleanity_verifier<F: JoltField>(
    trace_len: usize,
    one_hot_params: &OneHotParams,
    transcript: &mut impl Transcript,
) -> BooleanitySumcheckVerifier<F> {
    let params = ra_booleanity_params(trace_len, one_hot_params, transcript);
    BooleanitySumcheckVerifier::new(params)
}

pub fn new_ra_hamming_weight_verifier<F: JoltField>(
    one_hot_params: &OneHotParams,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> HammingWeightSumcheckVerifier<F> {
    let params = ra_hamming_weight_params(one_hot_params, opening_accumulator, transcript);
    HammingWeightSumcheckVerifier::new(params)
}

#[tracing::instrument(skip_all, name = "ram::compute_ram_ra_evals")]
fn compute_ram_ra_evals<F: JoltField>(
    trace: &[Cycle],
    memory_layout: &MemoryLayout,
    eq_r_cycle: &[F],
    one_hot_params: &OneHotParams,
) -> Vec<Vec<F>> {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let par_chunk_size = (T / num_chunks).max(1);

    let mut G_arrays = Vec::with_capacity(one_hot_params.ram_d);
    for i in 0..one_hot_params.ram_d {
        let G: Vec<F> = trace
            .par_chunks(par_chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut local_array = unsafe_allocate_zero_vec(one_hot_params.k_chunk);
                let mut j = chunk_index * par_chunk_size;
                for cycle in trace_chunk {
                    if let Some(address) =
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                    {
                        let address_i = one_hot_params.ram_address_chunk(address, i);
                        local_array[address_i as usize] += eq_r_cycle[j];
                    }
                    j += 1;
                }
                local_array
            })
            .reduce(
                || unsafe_allocate_zero_vec(one_hot_params.k_chunk),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );
        G_arrays.push(G);
    }
    G_arrays
}

#[tracing::instrument(skip_all, name = "ram::compute_ram_h_indices")]
fn compute_ram_h_indices(
    trace: &[Cycle],
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
) -> Vec<Vec<Option<u16>>> {
    let addresses: Vec<Option<u64>> = trace
        .par_iter()
        .map(|cycle| remap_address(cycle.ram_access().address() as u64, memory_layout))
        .collect();

    (0..one_hot_params.ram_d)
        .map(|i| {
            addresses
                .par_iter()
                .map(|address| address.map(|address| one_hot_params.ram_address_chunk(address, i)))
                .collect()
        })
        .collect()
}

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

use crate::zkvm::config::OneHotParams;
use crate::{
    field::{self, BarrettReduce, FMAdd, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    utils::{accumulation::Acc6U, math::Math},
    zkvm::witness::VirtualPolynomial,
};
use std::vec;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{
    constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS},
    jolt_device::MemoryLayout,
};
use rayon::prelude::*;
use tracer::emulator::memory::Memory;
use tracer::JoltDevice;

pub mod hamming_booleanity;
pub mod output_check;
pub mod ra_virtual;
pub mod raf_evaluation;
pub mod read_write_checking;
pub mod val_evaluation;
pub mod val_final;

/// RAM preprocessing metadata (shared between prover and verifier).
///
/// This struct is metadata-only and does NOT contain the full program-image words.
/// The full words are stored in `ProgramImagePreprocessing` (prover-only).
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMPreprocessing {
    /// Minimum bytecode address (word-aligned).
    pub min_bytecode_address: u64,
    /// Number of program-image words (unpadded).
    pub program_image_len_words: usize,
}

impl RAMPreprocessing {
    /// Create metadata from a `ProgramImagePreprocessing`.
    pub fn from_program_image(program_image: &ProgramImagePreprocessing) -> Self {
        Self {
            min_bytecode_address: program_image.min_bytecode_address,
            program_image_len_words: program_image.program_image_words.len(),
        }
    }
}

/// Full program-image preprocessing (prover-only and full-mode verifier).
///
/// Contains the actual u64 words that form the initial RAM program image.
/// This is O(program_size) data that the committed-mode verifier does NOT need.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProgramImagePreprocessing {
    /// Minimum bytecode address (word-aligned).
    pub min_bytecode_address: u64,
    /// Program-image words (little-endian packed u64 values).
    pub program_image_words: Vec<u64>,
}

impl ProgramImagePreprocessing {
    /// Preprocess memory_init bytes into packed u64 words.
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
        let mut program_image_words = vec![0u64; num_words as usize];
        // Convert bytes into words and populate `program_image_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 8 == address_b / 8)
        {
            let mut word = [0u8; 8];
            for (address, byte) in chunk {
                word[(address % 8) as usize] = *byte;
            }
            let word = u64::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 8 - min_bytecode_address / 8) as usize;
            program_image_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            program_image_words,
        }
    }

    /// Extract metadata-only `RAMPreprocessing` from this full preprocessing.
    pub fn meta(&self) -> RAMPreprocessing {
        RAMPreprocessing::from_program_image(self)
    }

    /// Unpadded number of words.
    pub fn unpadded_len_words(&self) -> usize {
        self.program_image_words.len()
    }

    /// Power-of-two padded length (minimum 1).
    pub fn padded_len_words_pow2(&self) -> usize {
        self.program_image_words.len().next_power_of_two().max(1)
    }
}

/// Returns Some(address) if there was read/write
/// Returns None if there was no read/write
#[inline(always)]
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
        let advice_point =
            OpeningPoint::new(r_address.r[total_variables - advice_variables..].to_vec());
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

/// Accumulates staged program-image scalar contribution claims into the prover accumulator.
///
/// These are scalar inner products:
/// - `C_rw  = Σ_j ProgramWord[j] * eq(r_address_rw, start_index + j)`
/// - `C_raf = Σ_j ProgramWord[j] * eq(r_address_raf, start_index + j)` (optional)
///
/// They are stored as *virtual* openings (not committed openings) because they are not direct
/// openings of the committed program-image polynomial.
pub fn prover_accumulate_program_image<F: JoltField>(
    ram_K: usize,
    min_bytecode_address: u64,
    program_image_words: &[u64],
    program_io: &JoltDevice,
    padded_len_words: usize,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
    single_opening: bool,
) {
    let total_vars = ram_K.log_2();
    let bytecode_start =
        remap_address(min_bytecode_address, &program_io.memory_layout).unwrap() as usize;

    // Get r_address_rw from RamVal/RamReadWriteChecking (used by ValEvaluation).
    let (r_rw, _) = opening_accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RamVal,
        SumcheckId::RamReadWriteChecking,
    );
    let (r_address_rw, _) = r_rw.split_at(total_vars);

    // Compute C_rw using the padded program-image word vector.
    let mut words = program_image_words.to_vec();
    words.resize(padded_len_words, 0u64);
    let c_rw = sparse_eval_u64_block::<F>(bytecode_start, &words, &r_address_rw.r);

    opening_accumulator.append_virtual(
        transcript,
        VirtualPolynomial::ProgramImageInitContributionRw,
        SumcheckId::RamValEvaluation,
        r_address_rw,
        c_rw,
    );

    if !single_opening {
        let (r_raf, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
        );
        let (r_address_raf, _) = r_raf.split_at(total_vars);
        let c_raf = sparse_eval_u64_block::<F>(bytecode_start, &words, &r_address_raf.r);
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ProgramImageInitContributionRaf,
            SumcheckId::RamValFinalEvaluation,
            r_address_raf,
            c_raf,
        );
    }
}

/// Mirrors [`prover_accumulate_program_image`], but only populates opening points and
/// appends the already-present scalar claims to the transcript.
pub fn verifier_accumulate_program_image<F: JoltField>(
    ram_K: usize,
    program_io: &JoltDevice,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
    single_opening: bool,
) {
    let total_vars = ram_K.log_2();
    // r_address_rw from RamVal/RamReadWriteChecking.
    let (r_rw, _) = opening_accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RamVal,
        SumcheckId::RamReadWriteChecking,
    );
    let (r_address_rw, _) = r_rw.split_at(total_vars);
    opening_accumulator.append_virtual(
        transcript,
        VirtualPolynomial::ProgramImageInitContributionRw,
        SumcheckId::RamValEvaluation,
        r_address_rw,
    );

    if !single_opening {
        let (r_raf, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
        );
        let (r_address_raf, _) = r_raf.split_at(total_vars);
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ProgramImageInitContributionRaf,
            SumcheckId::RamValFinalEvaluation,
            r_address_raf,
        );
    }
    // (program_io is unused for now; retained for symmetry and future checks)
    let _ = program_io;
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

/// Evaluate the public portion of the initial RAM state at a random address point `r_address`
/// without materializing the full length-`ram_K` initial memory vector.
///
/// Public initial memory consists of:
/// - the program image (`program_image_words`) placed at `min_bytecode_address`
/// - public inputs (`program_io.inputs`) placed at `memory_layout.input_start`
///
/// This function computes:
///   \sum_k Val_init_public[k] * eq(r_address, k)
/// but only over the (contiguous) regions that can be non-zero.
pub fn eval_initial_ram_mle<F: JoltField>(
    min_bytecode_address: u64,
    program_image_words: &[u64],
    program_io: &JoltDevice,
    r_address: &[F::Challenge],
) -> F {
    // Bytecode region
    let bytecode_start =
        remap_address(min_bytecode_address, &program_io.memory_layout).unwrap() as usize;
    let mut acc = sparse_eval_u64_block::<F>(bytecode_start, program_image_words, r_address);

    // Inputs region (packed into u64 words in little-endian)
    if !program_io.inputs.is_empty() {
        let input_start = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        let input_words: Vec<u64> = program_io
            .inputs
            .chunks(8)
            .map(|chunk| {
                let mut word = [0u8; 8];
                for (i, byte) in chunk.iter().enumerate() {
                    word[i] = *byte;
                }
                u64::from_le_bytes(word)
            })
            .collect();
        acc += sparse_eval_u64_block::<F>(input_start, &input_words, r_address);
    }

    acc
}

/// Evaluate only `program_io.inputs` as part of the initial RAM state at `r_address`.
///
/// Excludes program image, outputs, panic, and termination bits.
/// For the full IO region, see [`eval_io_mle`].
fn eval_inputs_mle<F: JoltField>(program_io: &JoltDevice, r_address: &[F::Challenge]) -> F {
    if program_io.inputs.is_empty() {
        return F::zero();
    }
    let input_start = remap_address(
        program_io.memory_layout.input_start,
        &program_io.memory_layout,
    )
    .unwrap() as usize;
    let input_words: Vec<u64> = program_io
        .inputs
        .chunks(8)
        .map(|chunk| {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            u64::from_le_bytes(word)
        })
        .collect();
    sparse_eval_u64_block::<F>(input_start, &input_words, r_address)
}

/// Evaluate a shifted slice of `u64` coefficients as a multilinear polynomial at `r`.
///
/// Conceptually computes:
/// \[
///   \sum_{j=0}^{len-1} values[j] \cdot eq(r, start_index + j)
/// \]
/// without materializing a full length-\(K\) vector or a full `eq(r, ·)` table.
///
/// Uses aligned power-of-two block decomposition with `EqPolynomial::evals_for_max_aligned_block`,
/// and accumulates using unreduced limb arithmetic via `Acc6U`.
fn sparse_eval_u64_block<F: JoltField>(
    start_index: usize,
    values: &[u64],
    r: &[F::Challenge],
) -> F {
    if values.is_empty() {
        return F::zero();
    }

    let mut acc = F::zero();
    let mut idx = start_index;
    let mut off = 0usize;
    while off < values.len() {
        let remaining = values.len() - off;
        let (block_size, block_evals) =
            EqPolynomial::<F>::evals_for_max_aligned_block(r, idx, remaining);
        debug_assert_eq!(block_evals.len(), block_size);

        // Accumulate this block in unreduced form, then reduce once.
        let mut block_acc: Acc6U<F> = Acc6U::default();
        for j in 0..block_size {
            // FMAdd implementation skips zeros internally.
            block_acc.fmadd(&block_evals[j], &values[off + j]);
        }
        acc += block_acc.barrett_reduce();

        idx += block_size;
        off += block_size;
    }
    acc
}

/// Evaluate the *public IO* polynomial at a (full-RAM) address point `r_address` without
/// materializing a dense IO-region vector.
///
/// This is the multilinear extension of the public IO words:
/// - inputs (packed into u64 words, little-endian) at `memory_layout.input_start`
/// - outputs (packed into u64 words, little-endian) at `memory_layout.output_start`
/// - panic bit at `memory_layout.panic`
/// - termination bit at `memory_layout.termination` (set to 1 only if not panicking)
/// - all other IO-region words are 0
///
/// The IO polynomial is naturally defined over the IO-region domain of size
/// `remap_address(RAM_START_ADDRESS, ..)` (in words), which is a power of two by construction.
/// When `r_address` has more variables than the IO polynomial, we embed it into the larger
/// domain by fixing the extra high-order variables to 0, which corresponds to multiplying
/// by `∏(1 - r_hi[i])`.
pub fn eval_io_mle<F: JoltField>(program_io: &JoltDevice, r_address: &[F::Challenge]) -> F {
    // IO region size in words (power of two).
    let range_end_words =
        remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap() as usize;
    let io_len_words = range_end_words.next_power_of_two().max(1);
    debug_assert!(io_len_words.is_power_of_two());

    let num_io_vars = io_len_words.log_2();
    let (r_hi, r_lo) = r_address.split_at(r_address.len() - num_io_vars);
    debug_assert_eq!(r_lo.len(), num_io_vars);

    // Embed the IO polynomial into the full RAM domain (if any extra high vars exist).
    let mut hi_scale = F::one();
    for r_i in r_hi.iter() {
        hi_scale *= F::one() - *r_i;
    }

    let mut acc = F::zero();

    // Inputs region
    if !program_io.inputs.is_empty() {
        let input_start = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        let input_words: Vec<u64> = program_io
            .inputs
            .chunks(8)
            .map(|chunk| {
                let mut word = [0u8; 8];
                for (i, byte) in chunk.iter().enumerate() {
                    word[i] = *byte;
                }
                u64::from_le_bytes(word)
            })
            .collect();
        acc += sparse_eval_u64_block::<F>(input_start, &input_words, r_lo);
    }

    // Outputs region
    if !program_io.outputs.is_empty() {
        let output_start = remap_address(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        let output_words: Vec<u64> = program_io
            .outputs
            .chunks(8)
            .map(|chunk| {
                let mut word = [0u8; 8];
                for (i, byte) in chunk.iter().enumerate() {
                    word[i] = *byte;
                }
                u64::from_le_bytes(word)
            })
            .collect();
        acc += sparse_eval_u64_block::<F>(output_start, &output_words, r_lo);
    }

    // Panic bit (one word)
    let panic_index =
        remap_address(program_io.memory_layout.panic, &program_io.memory_layout).unwrap() as usize;
    let panic_word = [program_io.panic as u64];
    acc += sparse_eval_u64_block::<F>(panic_index, &panic_word, r_lo);

    // Termination bit (one word), only set when not panicking.
    if !program_io.panic {
        let termination_index = remap_address(
            program_io.memory_layout.termination,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        let term_word = [1u64];
        acc += sparse_eval_u64_block::<F>(termination_index, &term_word, r_lo);
    }

    hi_scale * acc
}

/// Returns `(initial_memory_state, final_memory_state)`
pub fn gen_ram_memory_states<F: JoltField>(
    ram_K: usize,
    min_bytecode_address: u64,
    program_image_words: &[u64],
    program_io: &JoltDevice,
    final_memory: &Memory,
) -> (Vec<u64>, Vec<u64>) {
    let K = ram_K;

    let mut initial_memory_state: Vec<u64> = vec![0; K];
    // Copy bytecode
    let mut index =
        remap_address(min_bytecode_address, &program_io.memory_layout).unwrap() as usize;
    for word in program_image_words {
        initial_memory_state[index] = *word;
        index += 1;
    }

    let dram_start_index =
        remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap() as usize;
    let mut final_memory_state: Vec<u64> = vec![0; K];
    // Note that `final_memory` only contains memory at addresses >= `RAM_START_ADDRESS`
    // so we will still need to populate `final_memory_state` with the contents of
    // `program_io`, which lives at addresses < `RAM_START_ADDRESS`
    let final_memory_words = final_memory
        .data
        .get_num_doublewords()
        .min(K - dram_start_index);
    final_memory_state[dram_start_index..dram_start_index + final_memory_words]
        .par_iter_mut()
        .enumerate()
        .for_each(|(k, word)| {
            *word = final_memory.get_doubleword(8 * k as u64);
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
    min_bytecode_address: u64,
    program_image_words: &[u64],
    program_io: &JoltDevice,
) -> Vec<u64> {
    let mut initial_memory_state = vec![0; ram_K];
    // Copy bytecode
    let mut index =
        remap_address(min_bytecode_address, &program_io.memory_layout).unwrap() as usize;
    for word in program_image_words {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
    use ark_ff::UniformRand;
    use common::constants::RAM_START_ADDRESS;
    use common::jolt_device::MemoryConfig;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    #[test]
    fn public_initial_ram_eval_matches_dense_mle() {
        type F = ark_bn254::Fr;

        let mut rng = StdRng::seed_from_u64(12345);

        // Build a MemoryConfig with a fixed program size so MemoryLayout is well-defined.
        let memory_config = MemoryConfig {
            program_size: Some(4096),
            ..Default::default()
        };
        let mut program_io = JoltDevice::new(&memory_config);

        // Random public inputs (not necessarily 8-byte aligned).
        let mut inputs = vec![0u8; 37];
        rng.fill_bytes(&mut inputs);
        program_io.inputs = inputs;

        // Fake "bytecode" bytes at RAM_START_ADDRESS.
        let mut memory_init = Vec::new();
        for i in 0..73u64 {
            let b = (rng.next_u64() & 0xff) as u8;
            memory_init.push((RAM_START_ADDRESS + i, b));
        }
        let prog_pp = ProgramImagePreprocessing::preprocess(memory_init);

        // Choose ram_K large enough to cover both bytecode and inputs placements.
        let bytecode_start = remap_address(prog_pp.min_bytecode_address, &program_io.memory_layout)
            .unwrap() as usize;
        let input_start = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        let input_words_len = program_io.inputs.len().div_ceil(8);
        let needed = (bytecode_start + prog_pp.program_image_words.len())
            .max(input_start + input_words_len)
            .max(1);
        let ram_K = needed.next_power_of_two();

        let dense = gen_ram_initial_memory_state::<F>(
            ram_K,
            prog_pp.min_bytecode_address,
            &prog_pp.program_image_words,
            &program_io,
        );

        // Random evaluation point over address vars (big-endian convention).
        let n_vars = ram_K.log_2();
        let r: Vec<<F as JoltField>::Challenge> = (0..n_vars)
            .map(|_| <F as JoltField>::Challenge::rand(&mut rng))
            .collect();

        let dense_eval = MultilinearPolynomial::<F>::from(dense).evaluate(&r);
        let fast_eval = eval_initial_ram_mle::<F>(
            prog_pp.min_bytecode_address,
            &prog_pp.program_image_words,
            &program_io,
            &r,
        );

        assert_eq!(dense_eval, fast_eval);
    }
}

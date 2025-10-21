#![allow(clippy::too_many_arguments)]

use std::vec;

#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::{self, JoltField},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN},
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::math::Math,
    zkvm::{
        dag::{stage::SumcheckStages, state_manager::StateManager},
        ram::{
            booleanity::BooleanitySumcheck,
            hamming_booleanity::HammingBooleanitySumcheck,
            hamming_weight::HammingWeightSumcheck,
            output_check::{OutputSumcheck, ValFinalSumcheck},
            ra_virtual::RaSumcheck,
            raf_evaluation::RafEvaluationSumcheck,
            read_write_checking::RamReadWriteChecking,
            val_evaluation::ValEvaluationSumcheck,
        },
        witness::VirtualPolynomial,
    },
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{
    constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS},
    jolt_device::MemoryLayout,
};
use rayon::prelude::*;

pub mod booleanity;
pub mod hamming_booleanity;
pub mod hamming_weight;
pub mod output_check;
pub mod ra_virtual;
pub mod raf_evaluation;
pub mod read_write_checking;
pub mod val_evaluation;

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
        Some((address - lowest_address) / 8 + 1)
    } else {
        panic!("Unexpected address {address}")
    }
}

pub struct RamDag {
    initial_memory_state: Option<Vec<u64>>,
    final_memory_state: Option<Vec<u64>>,
}

impl RamDag {
    pub fn new_prover<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, _, program_io, final_memory) = state_manager.get_prover_data();
        let ram_preprocessing = &preprocessing.shared.ram;

        let K = state_manager.ram_K;

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
        final_memory_state[dram_start_index..]
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
        for chunk in program_io.trusted_advice.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            initial_memory_state[index] = word;
            final_memory_state[index] = word;
            index += 1;
        }

        index = remap_address(
            program_io.memory_layout.untrusted_advice_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        for chunk in program_io.untrusted_advice.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            initial_memory_state[index] = word;
            final_memory_state[index] = word;
            index += 1;
        }

        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        // Convert input bytes into words and populate
        // `initial_memory_state` and `final_memory_state`
        for chunk in program_io.inputs.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            initial_memory_state[index] = word;
            final_memory_state[index] = word;
            index += 1;
        }

        // Convert output bytes into words and populate
        // `final_memory_state`
        index = remap_address(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        for chunk in program_io.outputs.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            final_memory_state[index] = word;
            index += 1;
        }

        // Copy panic bit
        let panic_index = remap_address(program_io.memory_layout.panic, &program_io.memory_layout)
            .unwrap() as usize;
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

        #[cfg(test)]
        {
            use crate::zkvm::witness::CommittedPolynomial;

            let trace = state_manager.get_prover_data().1;

            let mut expected_final_memory_state: Vec<_> = initial_memory_state
                .iter()
                .map(|word| *word as i128)
                .collect();
            let inc = CommittedPolynomial::RamInc.generate_witness(preprocessing, trace);
            for (j, cycle) in trace.iter().enumerate() {
                use tracer::instruction::RAMAccess;

                if let RAMAccess::Write(write) = cycle.ram_access() {
                    if let Some(k) = remap_address(write.address, &program_io.memory_layout) {
                        expected_final_memory_state[k as usize] += inc.get_coeff_i128(j);
                    }
                }
            }
            let expected_final_memory_state: Vec<u64> = expected_final_memory_state
                .into_iter()
                .map(|word| word.try_into().unwrap())
                .collect();
            assert_eq!(expected_final_memory_state, final_memory_state);
        }

        Self {
            initial_memory_state: Some(initial_memory_state),
            final_memory_state: Some(final_memory_state),
        }
    }

    pub fn new_verifier<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, program_io, _) = state_manager.get_verifier_data();
        let ram_preprocessing = &preprocessing.shared.ram;

        let K = state_manager.ram_K;

        let mut initial_memory_state = vec![0; K];
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
        // `initial_memory_state` and `final_memory_state`
        for chunk in program_io.inputs.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            initial_memory_state[index] = word;
            index += 1;
        }

        Self {
            initial_memory_state: Some(initial_memory_state),
            final_memory_state: None,
        }
    }
}

/// Accumulates advice polynomials (trusted and untrusted) into the prover's accumulator.
pub fn prover_accumulate_advice<F, ProofTranscript, PCS>(
    state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
) where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let prover_state = state_manager
        .prover_state
        .as_ref()
        .expect("prover_state must be present when accumulating advice");

    let accumulate_closure = |advice_poly: &MultilinearPolynomial<F>, max_advice_size: usize| {
        let (r, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, _) = r.split_at(state_manager.ram_K.log_2());

        let total_variables = state_manager.ram_K.log_2();
        let advice_variables = (max_advice_size / 8).next_power_of_two().log_2();

        // Use the last number_of_vals elements for evaluation
        let eval = advice_poly.evaluate(&r_address.r[total_variables - advice_variables..]);

        let mut advice_point = r_address.clone();
        advice_point.r = r_address.r[total_variables - advice_variables..].to_vec();
        (advice_point, eval)
    };

    if let Some(untrusted_advice_poly) = &prover_state.untrusted_advice_polynomial {
        let (point, eval) = accumulate_closure(
            untrusted_advice_poly,
            state_manager
                .program_io
                .memory_layout
                .max_untrusted_advice_size as usize,
        );

        prover_state
            .accumulator
            .borrow_mut()
            .append_untrusted_advice(
                &mut *state_manager.get_transcript().borrow_mut(),
                point,
                eval,
            );
    }

    if let Some(trusted_advice_poly) = &prover_state.trusted_advice_polynomial {
        let (point, eval) = accumulate_closure(
            trusted_advice_poly,
            state_manager
                .program_io
                .memory_layout
                .max_trusted_advice_size as usize,
        );

        prover_state.accumulator.borrow_mut().append_trusted_advice(
            &mut *state_manager.get_transcript().borrow_mut(),
            point,
            eval,
        );
    }
}

/// Accumulates advice commitments into the verifier's accumulator.
pub fn verifier_accumulate_advice<F, ProofTranscript, PCS>(
    state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
) where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let verifier_state = state_manager
        .verifier_state
        .as_ref()
        .expect("verifier_state must be present when accumulating advice");

    let get_advice_point = |max_advice_size: usize| {
        let (r, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, _) = r.split_at(state_manager.ram_K.log_2());

        let total_vars = r_address.r.len();
        let advice_variables = (max_advice_size / 8).next_power_of_two().log_2();

        let mut advice_point = r_address.clone();
        advice_point.r = r_address.r[total_vars - advice_variables..].to_vec();
        advice_point
    };

    if state_manager.untrusted_advice_commitment.is_some() {
        let point = get_advice_point(
            state_manager
                .program_io
                .memory_layout
                .max_untrusted_advice_size as usize,
        );

        verifier_state
            .accumulator
            .borrow_mut()
            .append_untrusted_advice(&mut *state_manager.transcript.borrow_mut(), point);
    }

    if state_manager.trusted_advice_commitment.is_some() {
        let point = get_advice_point(
            state_manager
                .program_io
                .memory_layout
                .max_trusted_advice_size as usize,
        );

        verifier_state
            .accumulator
            .borrow_mut()
            .append_trusted_advice(&mut *state_manager.transcript.borrow_mut(), point);
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
pub fn calculate_advice_memory_evaluation<F: JoltField>(
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

impl<F, ProofTranscript, PCS> SumcheckStages<F, ProofTranscript, PCS> for RamDag
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        let raf_evaluation = RafEvaluationSumcheck::new_prover(state_manager);

        let read_write_checking = RamReadWriteChecking::new_prover(
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );

        let output_check = OutputSumcheck::new_prover(
            self.initial_memory_state.as_ref().unwrap().clone(),
            self.final_memory_state.as_ref().unwrap().clone(),
            state_manager,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("RAM RafEvaluationSumcheck", &raf_evaluation);
            print_data_structure_heap_usage("RAM RamReadWriteChecking", &read_write_checking);
            print_data_structure_heap_usage("RAM OutputSumcheck", &output_check);
        }

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        let raf_evaluation = RafEvaluationSumcheck::new_verifier(state_manager);
        let read_write_checking = RamReadWriteChecking::new_verifier(state_manager);
        let output_check = OutputSumcheck::new_verifier(state_manager);

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage4_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Accumulate advice polynomials if present
        prover_accumulate_advice(state_manager);
        let booleanity = BooleanitySumcheck::new_prover(state_manager);
        let val_evaluation = ValEvaluationSumcheck::new_prover(
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );
        let val_final_evaluation = ValFinalSumcheck::new_prover(state_manager);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("RAM BooleanitySumcheck", &booleanity);
            print_data_structure_heap_usage("RAM ValEvaluationSumcheck", &val_evaluation);
            print_data_structure_heap_usage("RAM ValFinalSumcheck", &val_final_evaluation);
        }

        vec![
            Box::new(booleanity),
            Box::new(val_evaluation),
            Box::new(val_final_evaluation),
        ]
    }

    fn stage4_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Accumulate advice commitments if present
        verifier_accumulate_advice(state_manager);
        let booleanity = BooleanitySumcheck::new_verifier(state_manager);
        let val_evaluation = ValEvaluationSumcheck::new_verifier(
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );
        let val_final_evaluation = ValFinalSumcheck::new_verifier(
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );

        vec![
            Box::new(booleanity),
            Box::new(val_evaluation),
            Box::new(val_final_evaluation),
        ]
    }

    fn stage5_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        let hamming_booleanity = HammingBooleanitySumcheck::new_prover(state_manager);
        let ra_virtual = RaSumcheck::new_prover(state_manager);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("RAM HammingBooleanitySumcheck", &hamming_booleanity);
            print_data_structure_heap_usage("RAM RASumcheck", &ra_virtual);
        }

        vec![Box::new(hamming_booleanity), Box::new(ra_virtual)]
    }

    fn stage5_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        let hamming_booleanity = HammingBooleanitySumcheck::new_verifier(state_manager);
        let ra_virtual = RaSumcheck::new_verifier(state_manager);

        vec![Box::new(hamming_booleanity), Box::new(ra_virtual)]
    }

    fn stage6_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        let hamming_weight = HammingWeightSumcheck::new_prover(state_manager);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("RAM HammingWeightSumcheck", &hamming_weight);
        }

        vec![Box::new(hamming_weight)]
    }

    fn stage6_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        let hamming_weight = HammingWeightSumcheck::new_verifier(state_manager);

        vec![Box::new(hamming_weight)]
    }
}

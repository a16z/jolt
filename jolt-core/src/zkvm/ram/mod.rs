#![allow(clippy::too_many_arguments)]

use std::vec;

#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    zkvm::dag::{stage::SumcheckStages, state_manager::StateManager},
    zkvm::ram::{
        booleanity::BooleanitySumcheck,
        hamming_booleanity::HammingBooleanitySumcheck,
        hamming_weight::HammingWeightSumcheck,
        output_check::{OutputSumcheck, ValFinalSumcheck},
        ra_virtual::RaSumcheck,
        raf_evaluation::RafEvaluationSumcheck,
        read_write_checking::RamReadWriteChecking,
        val_evaluation::ValEvaluationSumcheck,
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
    if address >= memory_layout.input_start {
        Some((address - memory_layout.input_start) / 8 + 1)
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
        let (preprocessing, _, _, program_io, final_memory) = state_manager.get_prover_data();
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

            let trace = state_manager.get_prover_data().2;

            let mut expected_final_memory_state: Vec<_> = initial_memory_state
                .iter()
                .map(|word| *word as i128)
                .collect();
            let ram_d = crate::zkvm::witness::AllCommittedPolynomials::ram_d();
            let inc = CommittedPolynomial::RamInc.generate_witness(preprocessing, trace, ram_d);
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

impl<F, ProofTranscript, PCS> SumcheckStages<F, ProofTranscript, PCS> for RamDag
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
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
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let raf_evaluation = RafEvaluationSumcheck::new_verifier(state_manager);
        let read_write_checking = RamReadWriteChecking::new_verifier(state_manager);
        let output_check = OutputSumcheck::new_verifier(state_manager);

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage3_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let val_evaluation = ValEvaluationSumcheck::new_prover(
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );
        let val_final_evaluation = ValFinalSumcheck::new_prover(state_manager);
        let hamming_booleanity = HammingBooleanitySumcheck::new_prover(state_manager);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("RAM ValEvaluationSumcheck", &val_evaluation);
            print_data_structure_heap_usage("RAM ValFinalSumcheck", &val_final_evaluation);
            print_data_structure_heap_usage("RAM HammingBooleanitySumcheck", &hamming_booleanity);
        }

        vec![
            Box::new(val_evaluation),
            Box::new(val_final_evaluation),
            Box::new(hamming_booleanity),
        ]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let val_evaluation = ValEvaluationSumcheck::new_verifier(
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );
        let val_final_evaluation = ValFinalSumcheck::new_verifier(
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );
        let hamming_booleanity = HammingBooleanitySumcheck::new_verifier(state_manager);

        vec![
            Box::new(val_evaluation),
            Box::new(val_final_evaluation),
            Box::new(hamming_booleanity),
        ]
    }

    fn stage4_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let hamming_weight = HammingWeightSumcheck::new_prover(state_manager);
        let booleanity = BooleanitySumcheck::new_prover(state_manager);
        let ra_virtual = RaSumcheck::new_prover(state_manager);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("RAM HammingWeightSumcheck", &hamming_weight);
            print_data_structure_heap_usage("RAM BooleanitySumcheck", &booleanity);
            print_data_structure_heap_usage("RAM RASumcheck", &ra_virtual);
        }

        vec![
            Box::new(hamming_weight),
            Box::new(booleanity),
            Box::new(ra_virtual),
        ]
    }

    fn stage4_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let hamming_weight = HammingWeightSumcheck::new_verifier(state_manager);
        let booleanity = BooleanitySumcheck::new_verifier(state_manager);
        let ra_virtual = RaSumcheck::new_verifier(state_manager);

        vec![
            Box::new(hamming_weight),
            Box::new(booleanity),
            Box::new(ra_virtual),
        ]
    }
}

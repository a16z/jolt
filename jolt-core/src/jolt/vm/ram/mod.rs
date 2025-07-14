#![allow(clippy::too_many_arguments)]

use std::vec;

use crate::{
    dag::{
        stage::{StagedSumcheck, SumcheckStages},
        state_manager::{ProofData, ProofKeys, StateManager},
    },
    field::JoltField,
    jolt::vm::ram::{
        booleanity::{BooleanityProof, BooleanitySumcheck},
        hamming_weight::{HammingWeightProof, HammingWeightSumcheck},
        output_check::{OutputProof, OutputSumcheck, ValFinalSumcheck},
        raf_evaluation::{RafEvaluationProof, RafEvaluationSumcheck},
        read_write_checking::{RamReadWriteChecking, RamReadWriteCheckingProof},
        val_evaluation::{ValEvaluationProof, ValEvaluationSumcheck},
    },
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::ra_virtual::{RAProof, RASumcheck},
    utils::transcript::Transcript,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{
    constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS},
    jolt_device::MemoryLayout,
};
use rayon::prelude::*;

pub mod booleanity;
pub mod hamming_weight;
pub mod output_check;
pub mod raf_evaluation;
pub mod read_write_checking;
pub mod val_evaluation;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMPreprocessing {
    min_bytecode_address: u64,
    bytecode_words: Vec<u32>,
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
            + (BYTES_PER_INSTRUCTION as u64 - 1); // For RV32IM, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3

        let num_words = max_bytecode_address.next_multiple_of(4) / 4 - min_bytecode_address / 4 + 1;
        let mut bytecode_words = vec![0u32; num_words as usize];
        // Convert bytes into words and populate `bytecode_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 4 == address_b / 4)
        {
            let mut word = [0u8; 4];
            for (address, byte) in chunk {
                word[(address % 4) as usize] = *byte;
            }
            let word = u32::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 4 - min_bytecode_address / 4) as usize;
            bytecode_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            bytecode_words,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RAMTwistProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) K: usize,
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: RamReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,

    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    ra_proof: RAProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    raf_evaluation_proof: RafEvaluationProof<F, ProofTranscript>,
    output_proof: OutputProof<F, ProofTranscript>,
}

pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> u64 {
    if address == 0 {
        return 0; // [JOLT-135]: Better handling for no-ops
    }
    if address >= memory_layout.input_start {
        (address - memory_layout.input_start) / 4 + 1
    } else {
        panic!("Unexpected address {address}")
    }
}

pub struct RamDag {
    K: usize,
    T: usize,
    initial_memory_state: Option<Vec<u32>>,
    final_memory_state: Option<Vec<u32>>,
}

impl RamDag {
    pub fn new_prover<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, program_io, final_memory) = state_manager.get_prover_data();
        let ram_preprocessing = &preprocessing.shared.ram;

        let K = trace
            .par_iter()
            .map(|cycle| {
                remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.shared.memory_layout,
                ) as usize
            })
            .max()
            .unwrap()
            .next_power_of_two();

        let T = trace.len();

        let mut initial_memory_state = vec![0; K];
        // Copy bytecode
        let mut index = remap_address(
            ram_preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in ram_preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word;
            index += 1;
        }

        let dram_start_index = remap_address(RAM_START_ADDRESS, &program_io.memory_layout) as usize;
        let mut final_memory_state = vec![0; K];
        // Note that `final_memory` only contains memory at addresses >= `RAM_START_ADDRESS`
        // so we will still need to populate `final_memory_state` with the contents of
        // `program_io`, which lives at addresses < `RAM_START_ADDRESS`
        final_memory_state[dram_start_index..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(k, word)| {
                *word = final_memory.read_word(4 * k as u64);
            });

        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate
        // `initial_memory_state` and `final_memory_state`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word;
            final_memory_state[index] = word;
            index += 1;
        }

        // Convert output bytes into words and populate
        // `final_memory_state`
        index = remap_address(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        ) as usize;
        for chunk in program_io.outputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            final_memory_state[index] = word;
            index += 1;
        }

        // Copy panic bit
        let panic_index =
            remap_address(program_io.memory_layout.panic, &program_io.memory_layout) as usize;
        final_memory_state[panic_index] = program_io.panic as u32;
        if !program_io.panic {
            // Set termination bit
            let termination_index = remap_address(
                program_io.memory_layout.termination,
                &program_io.memory_layout,
            ) as usize;
            final_memory_state[termination_index] = 1;
        }

        #[cfg(test)]
        {
            use crate::jolt::witness::CommittedPolynomials;

            let mut expected_final_memory_state: Vec<_> = initial_memory_state
                .iter()
                .map(|word| *word as i64)
                .collect();
            let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);
            for (j, cycle) in trace.iter().enumerate() {
                use tracer::instruction::RAMAccess;

                if let RAMAccess::Write(write) = cycle.ram_access() {
                    let k = remap_address(write.address, &program_io.memory_layout) as usize;
                    expected_final_memory_state[k] += inc.get_coeff_i64(j);
                }
            }
            let expected_final_memory_state: Vec<u32> = expected_final_memory_state
                .into_iter()
                .map(|word| word.try_into().unwrap())
                .collect();
            assert_eq!(expected_final_memory_state, final_memory_state);
        }

        Self {
            K,
            T,
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
        let (preprocessing, program_io, T) = state_manager.get_verifier_data();
        let ram_preprocessing = &preprocessing.shared.ram;

        let K = match state_manager.proofs.borrow().get(&ProofKeys::RamK) {
            Some(ProofData::RamK(K)) => *K,
            _ => panic!("RAM K not set"),
        };

        let mut initial_memory_state = vec![0; K];
        // Copy bytecode
        let mut index = remap_address(
            ram_preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in ram_preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word;
            index += 1;
        }

        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate
        // `initial_memory_state` and `final_memory_state`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word;
            index += 1;
        }

        Self {
            K,
            T,
            initial_memory_state: Some(initial_memory_state),
            final_memory_state: None,
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for RamReadWriteChecking<F>
{
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for RafEvaluationSumcheck<F>
{
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS> for OutputSumcheck<F> {}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for ValFinalSumcheck<F>
{
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for HammingWeightSumcheck<F>
{
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for BooleanitySumcheck<F>
{
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS> for RASumcheck<F> {}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStages<F, ProofTranscript, PCS> for RamDag
{
    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let raf_evaluation = RafEvaluationSumcheck::new_prover(self.K, self.T, state_manager);

        let read_write_checking = RamReadWriteChecking::new_prover(
            self.K,
            self.T,
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );

        let output_check = OutputSumcheck::new_prover(
            self.initial_memory_state.as_ref().unwrap().clone(),
            self.final_memory_state.as_ref().unwrap().clone(),
            state_manager,
        );

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let raf_evaluation = RafEvaluationSumcheck::new_verifier(self.K, state_manager);
        let read_write_checking = RamReadWriteChecking::new_verifier(self.K, state_manager);
        let output_check = OutputSumcheck::new_verifier(self.K, state_manager);

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage3_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let val_evaluation = ValEvaluationSumcheck::new_prover(
            self.K,
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );
        let val_final_evaluation = ValFinalSumcheck::new_prover(state_manager);

        vec![Box::new(val_evaluation), Box::new(val_final_evaluation)]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let val_evaluation = ValEvaluationSumcheck::new_verifier(
            self.K,
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );
        let val_final_evaluation = ValFinalSumcheck::new_verifier(
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );

        vec![Box::new(val_evaluation), Box::new(val_final_evaluation)]
    }

    fn stage4_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let hamming_weight = HammingWeightSumcheck::new_prover(self.K, state_manager);
        let booleanity = BooleanitySumcheck::new_prover(self.K, state_manager);
        let ra_virtual = RASumcheck::new_prover(self.K, state_manager);

        vec![
            Box::new(hamming_weight),
            Box::new(booleanity),
            Box::new(ra_virtual),
        ]
    }

    fn stage4_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let hamming_weight = HammingWeightSumcheck::new_verifier(self.K, state_manager);
        let booleanity = BooleanitySumcheck::new_verifier(self.K, state_manager);
        let ra_virtual = RASumcheck::new_verifier(self.K, state_manager);

        vec![
            Box::new(hamming_weight),
            Box::new(booleanity),
            Box::new(ra_virtual),
        ]
    }
}

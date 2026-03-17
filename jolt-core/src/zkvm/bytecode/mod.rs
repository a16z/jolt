use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
use rayon::prelude::*;
use tracer::instruction::{Cycle, Instruction};

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::utils::math::Math;
use crate::zkvm::bytecode::chunks::{
    build_committed_bytecode_chunk_polynomials, committed_bytecode_chunk_cycle_len,
    committed_lanes, validate_committed_bytecode_chunking_for_len,
};

pub mod chunks;
pub mod read_raf_checking;

#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedBytecodeCommitments<PCS: CommitmentScheme> {
    pub commitments: Vec<PCS::Commitment>,
    pub num_columns: usize,
    pub log_k_chunk: u8,
    pub bytecode_chunk_count: usize,
    pub bytecode_len: usize,
    pub bytecode_T: usize,
}

#[derive(Clone)]
pub struct TrustedBytecodeHints<PCS: CommitmentScheme> {
    pub hints: Vec<PCS::OpeningProofHint>,
}

impl<PCS: CommitmentScheme> TrustedBytecodeCommitments<PCS> {
    #[tracing::instrument(skip_all, name = "TrustedBytecodeCommitments::derive")]
    pub fn derive(
        bytecode: &BytecodePreprocessing,
        generators: &PCS::ProverSetup,
        log_k_chunk: usize,
        bytecode_chunk_count: usize,
    ) -> (Self, TrustedBytecodeHints<PCS>) {
        let bytecode_len = bytecode.code_size;
        validate_committed_bytecode_chunking_for_len(bytecode_len, bytecode_chunk_count);
        let bytecode_T = committed_bytecode_chunk_cycle_len(bytecode_len, bytecode_chunk_count);

        let total_vars = bytecode_T.log_2() + committed_lanes().log_2();
        let (bytecode_sigma, _) = DoryGlobals::balanced_sigma_nu(total_vars);
        let num_columns = 1usize << bytecode_sigma;

        let bytecode_chunk_polys = build_committed_bytecode_chunk_polynomials::<PCS::Field>(
            &bytecode.bytecode,
            bytecode_chunk_count,
        );
        let _bytecode_guard = DoryGlobals::initialize_context(
            committed_lanes(),
            bytecode_T,
            DoryContext::UntrustedAdvice,
            None,
        );
        let (commitments, hints): (Vec<_>, Vec<_>) = bytecode_chunk_polys
            .par_iter()
            .map(|poly| {
                let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);
                PCS::commit(poly, generators)
            })
            .unzip();

        (
            Self {
                commitments,
                num_columns,
                log_k_chunk: log_k_chunk as u8,
                bytecode_chunk_count,
                bytecode_len,
                bytecode_T,
            },
            TrustedBytecodeHints { hints },
        )
    }
}

#[derive(Default, Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<Instruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    pub pc_map: BytecodePCMapper,
    /// ELF entry point address. Used to constrain the first executed PC
    /// via the BytecodeReadRaf sumcheck (Stage 6), which adds a term forcing
    /// `ra(entry_bytecode_index, 0) = 1`.
    pub entry_address: u64,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<Instruction>, entry_address: u64) -> Self {
        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, Instruction::NoOp);
        let pc_map = BytecodePCMapper::new(&bytecode);

        let code_size = bytecode.len().next_power_of_two().max(2);

        // Bytecode: Pad to nearest power of 2
        bytecode.resize(code_size, Instruction::NoOp);

        Self {
            code_size,
            bytecode,
            pc_map,
            entry_address,
        }
    }

    /// Returns the bytecode table index for the ELF entry point.
    pub fn entry_bytecode_index(&self) -> usize {
        self.pc_map.get_pc(self.entry_address as usize, 0)
    }

    pub fn get_pc(&self, cycle: &Cycle) -> usize {
        if matches!(cycle, tracer::instruction::Cycle::NoOp) {
            return 0;
        }
        let instr = cycle.instruction().normalize();
        self.pc_map
            .get_pc(instr.address, instr.virtual_sequence_remaining.unwrap_or(0))
    }
}

#[derive(Default, Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePCMapper {
    /// Stores the mapping of the PC at the beginning of each inline sequence
    /// and the maximum number of the inline sequence
    /// Indexed by the address of instruction unmapped divided by 2
    indices: Vec<Option<(usize, u16)>>,
}

impl BytecodePCMapper {
    pub fn new(bytecode: &[Instruction]) -> Self {
        let mut indices: Vec<Option<(usize, u16)>> = {
            // For read-raf tests we simulate bytecode being empty
            #[cfg(test)]
            {
                if bytecode.len() == 1 {
                    vec![None; 1]
                } else {
                    vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
                }
            }
            #[cfg(not(test))]
            {
                vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
            }
        };
        let mut last_pc = 0;
        // Push the initial noop instruction
        indices[0] = Some((last_pc, 0));
        bytecode.iter().for_each(|instr| {
            let instr = instr.normalize();
            if instr.address == 0 {
                // ignore unimplemented instructions
                return;
            }
            last_pc += 1;
            if let Some((_, max_sequence)) = indices.get(Self::get_index(instr.address)).unwrap() {
                if instr.virtual_sequence_remaining.unwrap_or(0) >= *max_sequence {
                    panic!(
                        "Bytecode has non-decreasing inline sequences at index {}",
                        Self::get_index(instr.address)
                    );
                }
            } else {
                indices[Self::get_index(instr.address)] =
                    Some((last_pc, instr.virtual_sequence_remaining.unwrap_or(0)));
            }
        });
        Self { indices }
    }

    pub fn get_pc(&self, address: usize, virtual_sequence_remaining: u16) -> usize {
        let (base_pc, max_inline_seq) = self
            .indices
            .get(Self::get_index(address))
            .unwrap()
            .expect("PC for address not found");
        base_pc + (max_inline_seq - virtual_sequence_remaining) as usize
    }

    pub const fn get_index(address: usize) -> usize {
        assert!(address >= RAM_START_ADDRESS as usize);
        assert!(address.is_multiple_of(ALIGNMENT_FACTOR_BYTECODE));
        (address - RAM_START_ADDRESS as usize) / ALIGNMENT_FACTOR_BYTECODE + 1
    }
}

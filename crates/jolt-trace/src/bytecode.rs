//! Bytecode preprocessing and PC mapping.
//!
//! Extracts from the decoded ELF instructions two artifacts needed by the prover:
//!
//! - [`BytecodePreprocessing`] — the padded bytecode table (power-of-2 length) plus
//!   the [`BytecodePCMapper`] that maps physical instruction addresses to dense indices.
//!
//! - [`BytecodePCMapper::get_pc`] — resolves `(address, virtual_sequence_remaining)`
//!   to a dense bytecode table index, accounting for virtual instruction expansion.

use crate::JoltInstruction;
use common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
use serde::{Deserialize, Serialize};
use tracer::instruction::Instruction;

/// Preprocessed bytecode table with PC mapper.
///
/// Built once from the decoded ELF instructions. The prover uses [`get_pc`](Self::get_pc)
/// to resolve each cycle's address to a dense index, and the verifier uses the bytecode
/// table for commitment verification.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct BytecodePreprocessing {
    /// Power-of-2 padded bytecode table length.
    pub code_size: usize,
    /// Padded instruction table (index 0 is a prepended no-op).
    pub bytecode: Vec<Instruction>,
    /// Maps physical addresses to dense bytecode indices.
    pub pc_map: BytecodePCMapper,
    /// ELF entry point address.
    pub entry_address: u64,
}

impl BytecodePreprocessing {
    /// Build the bytecode table from decoded instructions.
    ///
    /// Prepends a no-op instruction at index 0 and pads to the next power of two.
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<Instruction>, entry_address: u64) -> Self {
        bytecode.insert(0, Instruction::NoOp);
        let pc_map = BytecodePCMapper::new(&bytecode);
        let code_size = bytecode.len().next_power_of_two().max(2);
        bytecode.resize(code_size, Instruction::NoOp);

        Self {
            code_size,
            bytecode,
            pc_map,
            entry_address,
        }
    }

    /// Dense bytecode table index for the ELF entry point.
    pub fn entry_bytecode_index(&self) -> usize {
        self.pc_map.get_pc(self.entry_address as usize, 0)
    }

    /// Resolve a cycle's PC to a dense bytecode table index.
    pub fn get_pc(&self, cycle: &impl JoltInstruction) -> usize {
        if cycle.is_noop() {
            return 0;
        }
        self.pc_map.get_pc(
            cycle.address() as usize,
            cycle.virtual_sequence_remaining().unwrap_or(0),
        )
    }
}

/// Maps physical instruction addresses to dense bytecode table indices.
///
/// Accounts for virtual instruction expansion: a single physical instruction
/// may expand into a sequence of virtual instructions, each at a distinct
/// dense index.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct BytecodePCMapper {
    /// `indices[(addr - RAM_START) / ALIGNMENT + 1] = Some((base_pc, max_sequence))`
    indices: Vec<Option<(usize, u16)>>,
}

impl BytecodePCMapper {
    #[expect(clippy::unwrap_used)]
    pub fn new(bytecode: &[Instruction]) -> Self {
        let capacity = if bytecode.len() <= 1 {
            1
        } else {
            Self::get_index(bytecode.last().unwrap().normalize().address) + 1
        };
        let mut indices: Vec<Option<(usize, u16)>> = vec![None; capacity];
        let mut last_pc = 0;
        indices[0] = Some((last_pc, 0));

        for instr in bytecode {
            let instr = instr.normalize();
            if instr.address == 0 {
                continue;
            }
            last_pc += 1;
            let idx = Self::get_index(instr.address);
            if let Some((_, max_sequence)) = indices[idx] {
                assert!(
                    instr.virtual_sequence_remaining.unwrap_or(0) < max_sequence,
                    "Bytecode has non-decreasing inline sequences at index {idx}"
                );
            } else {
                indices[idx] = Some((last_pc, instr.virtual_sequence_remaining.unwrap_or(0)));
            }
        }
        Self { indices }
    }

    #[expect(clippy::expect_used)]
    pub fn get_pc(&self, address: usize, virtual_sequence_remaining: u16) -> usize {
        let (base_pc, max_inline_seq) =
            self.indices[Self::get_index(address)].expect("PC for address not found");
        base_pc + (max_inline_seq - virtual_sequence_remaining) as usize
    }

    /// Convert a physical address to the internal index.
    pub const fn get_index(address: usize) -> usize {
        assert!(address >= RAM_START_ADDRESS as usize);
        assert!(address.is_multiple_of(ALIGNMENT_FACTOR_BYTECODE));
        (address - RAM_START_ADDRESS as usize) / ALIGNMENT_FACTOR_BYTECODE + 1
    }
}

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
use thiserror::Error;
use tracer::instruction::{Cycle, Instruction};

pub mod read_raf_checking;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PreprocessingError {
    #[error(
        "bytecode has non-decreasing inline sequences at index {bytecode_index} (address {address:#x}): previous max sequence {previous_max_sequence}, new sequence {new_sequence}"
    )]
    NonDecreasingInlineSequence {
        bytecode_index: usize,
        address: usize,
        previous_max_sequence: u16,
        new_sequence: u16,
    },
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
    pub fn preprocess(
        mut bytecode: Vec<Instruction>,
        entry_address: u64,
    ) -> Result<Self, PreprocessingError> {
        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, Instruction::NoOp);
        let pc_map = BytecodePCMapper::try_new(&bytecode)?;

        let code_size = bytecode.len().next_power_of_two().max(2);

        // Bytecode: Pad to nearest power of 2
        bytecode.resize(code_size, Instruction::NoOp);

        Ok(Self {
            code_size,
            bytecode,
            pc_map,
            entry_address,
        })
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
    pub fn try_new(bytecode: &[Instruction]) -> Result<Self, PreprocessingError> {
        let mut indices: Vec<Option<(usize, u16)>> = {
            // For read-raf tests we simulate bytecode being empty
            #[cfg(test)]
            if bytecode.len() == 1 {
                vec![None; 1]
            } else {
                vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
            }
            #[cfg(not(test))]
            vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
        };
        let mut last_pc = 0;
        // Push the initial noop instruction
        indices[0] = Some((last_pc, 0));
        for instr in bytecode.iter() {
            let instr = instr.normalize();
            if instr.address == 0 {
                // ignore unimplemented instructions
                continue;
            }
            last_pc += 1;
            let bytecode_index = Self::get_index(instr.address);
            let new_sequence = instr.virtual_sequence_remaining.unwrap_or(0);
            if let Some((_, max_sequence)) = indices.get(bytecode_index).unwrap() {
                if new_sequence >= *max_sequence {
                    return Err(PreprocessingError::NonDecreasingInlineSequence {
                        bytecode_index,
                        address: instr.address,
                        previous_max_sequence: *max_sequence,
                        new_sequence,
                    });
                }
            } else {
                indices[bytecode_index] = Some((last_pc, new_sequence));
            }
        }
        Ok(Self { indices })
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

#[cfg(test)]
mod tests {
    use super::{BytecodePCMapper, PreprocessingError};
    use tracer::instruction::{add::ADD, format::format_r::FormatR, Instruction};

    #[test]
    fn rejects_non_decreasing_inline_sequences() {
        let bytecode = vec![
            Instruction::NoOp,
            Instruction::ADD(ADD {
                address: 0x8000_0004,
                operands: FormatR {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                },
                virtual_sequence_remaining: Some(1),
                ..Default::default()
            }),
            Instruction::ADD(ADD {
                address: 0x8000_0004,
                operands: FormatR {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                },
                virtual_sequence_remaining: Some(1),
                ..Default::default()
            }),
        ];

        let err = BytecodePCMapper::try_new(&bytecode).unwrap_err();
        assert_eq!(
            err,
            PreprocessingError::NonDecreasingInlineSequence {
                bytecode_index: BytecodePCMapper::get_index(0x8000_0004),
                address: 0x8000_0004,
                previous_max_sequence: 1,
                new_sequence: 1,
            }
        );
    }
}

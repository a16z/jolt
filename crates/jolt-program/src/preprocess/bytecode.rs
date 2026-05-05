use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
use jolt_riscv::{InstructionKind, NormalizedInstruction};

use crate::preprocess::PreprocessingError;

#[derive(
    Default,
    Debug,
    Clone,
    PartialEq,
    Eq,
    CanonicalSerialize,
    CanonicalDeserialize,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<NormalizedInstruction>,
    /// Maps each unexpanded instruction address to its virtual bytecode index.
    pub pc_map: BytecodePCMapper,
    pub entry_address: u64,
}

impl BytecodePreprocessing {
    pub fn preprocess(
        mut bytecode: Vec<NormalizedInstruction>,
        entry_address: u64,
    ) -> Result<Self, PreprocessingError> {
        bytecode.insert(0, noop_instruction());
        let pc_map = BytecodePCMapper::try_new(&bytecode)?;

        let code_size = bytecode.len().next_power_of_two().max(2);
        bytecode.resize(code_size, noop_instruction());

        Ok(Self {
            code_size,
            bytecode,
            pc_map,
            entry_address,
        })
    }

    pub fn entry_bytecode_index(&self) -> Option<usize> {
        self.pc_map.get_pc(self.entry_address as usize, 0)
    }

    pub fn get_pc(&self, instruction: &NormalizedInstruction) -> Option<usize> {
        if instruction.instruction_kind == InstructionKind::NoOp {
            return Some(0);
        }
        self.pc_map.get_pc(
            instruction.address,
            instruction.virtual_sequence_remaining.unwrap_or(0),
        )
    }
}

#[derive(
    Default,
    Debug,
    Clone,
    PartialEq,
    Eq,
    CanonicalSerialize,
    CanonicalDeserialize,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct BytecodePCMapper {
    indices: Vec<Option<(usize, u16)>>,
}

impl BytecodePCMapper {
    pub fn try_new(bytecode: &[NormalizedInstruction]) -> Result<Self, PreprocessingError> {
        let mut indices = vec![None; Self::index_count(bytecode)];
        let mut last_pc = 0;
        indices[0] = Some((last_pc, 0));

        for instruction in bytecode {
            if instruction.address == 0 {
                continue;
            }

            last_pc += 1;
            let bytecode_index = Self::get_index(instruction.address);
            let new_sequence = instruction.virtual_sequence_remaining.unwrap_or(0);
            if let Some((_, max_sequence)) = indices[bytecode_index] {
                if new_sequence >= max_sequence {
                    return Err(PreprocessingError::NonDecreasingInlineSequence {
                        bytecode_index,
                        address: instruction.address,
                        previous_max_sequence: max_sequence,
                        new_sequence,
                    });
                }
            } else {
                indices[bytecode_index] = Some((last_pc, new_sequence));
            }
        }

        Ok(Self { indices })
    }

    pub fn get_pc(&self, address: usize, virtual_sequence_remaining: u16) -> Option<usize> {
        let index = Self::get_index(address);
        let (base_pc, max_inline_seq) = self.indices.get(index).copied().flatten()?;
        Some(base_pc + (max_inline_seq - virtual_sequence_remaining) as usize)
    }

    pub const fn get_index(address: usize) -> usize {
        assert!(address >= RAM_START_ADDRESS as usize);
        assert!(address.is_multiple_of(ALIGNMENT_FACTOR_BYTECODE));
        (address - RAM_START_ADDRESS as usize) / ALIGNMENT_FACTOR_BYTECODE + 1
    }

    fn index_count(bytecode: &[NormalizedInstruction]) -> usize {
        let max_address = bytecode
            .iter()
            .map(|instruction| instruction.address)
            .max()
            .unwrap_or(0);
        if max_address == 0 {
            1
        } else {
            Self::get_index(max_address) + 1
        }
    }
}

const fn noop_instruction() -> NormalizedInstruction {
    NormalizedInstruction {
        instruction_kind: InstructionKind::NoOp,
        address: 0,
        operands: jolt_riscv::NormalizedOperands {
            rs1: None,
            rs2: None,
            rd: None,
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_riscv::{InstructionKind, NormalizedInstruction, NormalizedOperands};

    use super::{BytecodePCMapper, BytecodePreprocessing, PreprocessingError};

    #[test]
    fn preprocess_prepends_and_pads_bytecode() {
        let bytecode = vec![instruction(0x8000_0000, None)];

        let preprocessing = BytecodePreprocessing::preprocess(bytecode, 0x8000_0000).unwrap();

        assert_eq!(preprocessing.code_size, 2);
        assert_eq!(
            preprocessing.bytecode[0].instruction_kind,
            InstructionKind::NoOp
        );
        assert_eq!(preprocessing.entry_bytecode_index(), Some(1));
    }

    #[test]
    fn maps_inline_sequence_pcs() {
        let bytecode = vec![
            instruction(0x8000_0004, Some(2)),
            instruction(0x8000_0004, Some(1)),
            instruction(0x8000_0004, Some(0)),
        ];

        let preprocessing = BytecodePreprocessing::preprocess(bytecode, 0x8000_0004).unwrap();

        assert_eq!(preprocessing.entry_bytecode_index(), Some(3));
        assert_eq!(
            preprocessing.get_pc(&instruction(0x8000_0004, Some(2))),
            Some(1)
        );
        assert_eq!(
            preprocessing.get_pc(&instruction(0x8000_0004, Some(1))),
            Some(2)
        );
        assert_eq!(
            preprocessing.get_pc(&instruction(0x8000_0004, Some(0))),
            Some(3)
        );
    }

    #[test]
    fn rejects_non_decreasing_inline_sequences() {
        let bytecode = vec![
            instruction(0x8000_0004, Some(1)),
            instruction(0x8000_0004, Some(1)),
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

    fn instruction(
        address: usize,
        virtual_sequence_remaining: Option<u16>,
    ) -> NormalizedInstruction {
        NormalizedInstruction {
            instruction_kind: InstructionKind::ADDI,
            address,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: 3,
            },
            virtual_sequence_remaining,
            is_first_in_sequence: virtual_sequence_remaining == Some(2),
            is_compressed: false,
        }
    }
}

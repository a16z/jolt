use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};

use crate::expand::{
    allocator::ExpansionAllocator, buffer::ExpansionBuffer, expand_instruction_core,
    metadata::stamp_sequence, ExpansionError,
};

pub(super) struct ExpansionState<'a> {
    allocator: &'a mut ExpansionAllocator,
}

impl<'a> ExpansionState<'a> {
    pub(super) fn new(allocator: &'a mut ExpansionAllocator) -> Self {
        Self { allocator }
    }

    pub(super) fn allocator(&mut self) -> &mut ExpansionAllocator {
        self.allocator
    }

    pub(super) fn expand_one_core(
        &mut self,
        instruction: &NormalizedInstruction,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        self.allocator.enter_expansion()?;
        let result = expand_instruction_core(instruction, self);
        self.allocator.exit_expansion();
        result
    }
}

pub(super) struct ExpansionSequence {
    address: usize,
    is_compressed: bool,
    rows: ExpansionBuffer,
}

impl ExpansionSequence {
    pub(super) fn new(source: &NormalizedInstruction) -> Self {
        Self {
            address: source.address,
            is_compressed: source.is_compressed,
            rows: ExpansionBuffer::new(),
        }
    }

    pub(super) fn emit(
        &mut self,
        instruction_kind: JoltInstructionKind,
        operands: NormalizedOperands,
    ) {
        self.rows.push(NormalizedInstruction {
            instruction_kind,
            address: self.address,
            operands,
            virtual_sequence_remaining: Some(0),
            is_first_in_sequence: false,
            is_compressed: false,
        });
    }

    pub(super) fn extend(
        &mut self,
        rows: impl IntoIterator<Item = NormalizedInstruction>,
    ) -> Result<(), ExpansionError> {
        self.rows.extend(rows)
    }

    pub(super) fn finish(self) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        self.rows.check_capacity()?;
        stamp_sequence(self.rows.into_vec(), self.is_compressed)
    }
}

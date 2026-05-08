use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};

use crate::expand::{
    allocator::ExpansionAllocator, buffer::ExpansionBuffer, expand_instruction_core,
    grammar::ExpansionOp, metadata::stamp_sequence, ExpansionError,
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

    pub(super) fn materialize_ops(
        &mut self,
        source: &NormalizedInstruction,
        ops: impl IntoIterator<Item = ExpansionOp>,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        let mut sequence = ExpansionSequence::new(source);
        self.materialize_ops_into(&mut sequence, source, ops)?;
        sequence.finish()
    }

    pub(super) fn materialize_ops_into(
        &mut self,
        sequence: &mut ExpansionSequence,
        source: &NormalizedInstruction,
        ops: impl IntoIterator<Item = ExpansionOp>,
    ) -> Result<(), ExpansionError> {
        for op in ops {
            match op {
                ExpansionOp::Row(row) => sequence.emit(row.instruction_kind, row.operands),
                ExpansionOp::Expand(row) => {
                    let instruction = row.instruction_at(source.address);
                    sequence.extend(self.expand_one_core(&instruction)?)?;
                }
                ExpansionOp::Release(register) => self.allocator.release(register)?,
            }
        }
        Ok(())
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

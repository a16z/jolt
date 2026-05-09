use jolt_riscv::{NormalizedInstruction, NormalizedOperands};

use crate::expand::{
    allocator::ExpansionAllocator,
    buffer::ExpansionBuffer,
    expand_instruction_core,
    grammar::{ExpandedInstructionSequence, ExpansionOp, RowTemplate},
    metadata::stamp_sequence,
    ExpansionError,
};

pub(super) struct ExpansionState {
    allocator: ExpansionAllocator,
    work: Vec<ExpansionOp>,
    fuel: u32,
}

impl ExpansionState {
    pub(super) fn new(allocator: ExpansionAllocator) -> Self {
        Self {
            allocator,
            work: Vec::new(),
            fuel: 0,
        }
    }

    pub(super) fn into_allocator(self) -> ExpansionAllocator {
        self.allocator
    }

    pub(super) fn allocator(&mut self) -> &mut ExpansionAllocator {
        &mut self.allocator
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

    pub(super) fn materialize(
        &mut self,
        sequence: ExpandedInstructionSequence,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        let saved_work = std::mem::replace(
            &mut self.work,
            sequence.ops.into_iter().rev().collect::<Vec<_>>(),
        );
        let saved_fuel = self.fuel;
        self.fuel = self.work.len() as u32;
        let result = self.materialize_current_work(sequence.source);
        self.work = saved_work;
        self.fuel = saved_fuel;
        result
    }

    fn materialize_current_work(
        &mut self,
        source: NormalizedInstruction,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        let mut materializer = SequenceMaterializer::new(source);
        while let Some(op) = self.work.pop() {
            self.consume_fuel()?;
            match op {
                ExpansionOp::Emit(row) => materializer.emit(row)?,
                ExpansionOp::Expand(row) => {
                    let instruction = materializer.instruction(row)?;
                    materializer.extend(self.expand_one_core(&instruction)?)?;
                }
                ExpansionOp::Allocate(register) => {
                    let allocated = self.allocator.allocate()?;
                    materializer.bind_temp(register, allocated)?;
                }
                ExpansionOp::Release(register) => {
                    let register = materializer.resolve_register_for_release(register)?;
                    self.allocator.release(register)?;
                }
            }
        }
        materializer.finish()
    }

    fn consume_fuel(&mut self) -> Result<(), ExpansionError> {
        self.fuel = self
            .fuel
            .checked_sub(1)
            .ok_or(ExpansionError::RecursionDepthExceeded { max_depth: 0 })?;
        Ok(())
    }
}

struct SequenceMaterializer {
    address: usize,
    is_compressed: bool,
    rows: ExpansionBuffer,
    temps: Vec<Option<u8>>,
}

impl SequenceMaterializer {
    fn new(source: NormalizedInstruction) -> Self {
        Self {
            address: source.address,
            is_compressed: source.is_compressed,
            rows: ExpansionBuffer::new(),
            temps: Vec::new(),
        }
    }

    fn emit(&mut self, row: RowTemplate) -> Result<(), ExpansionError> {
        let row = self.instruction(row)?;
        self.rows.extend([row])
    }

    fn extend(&mut self, rows: Vec<NormalizedInstruction>) -> Result<(), ExpansionError> {
        self.rows.extend(rows)
    }

    fn instruction(&self, row: RowTemplate) -> Result<NormalizedInstruction, ExpansionError> {
        let mut instruction = row.instruction_at(self.address);
        instruction.operands = self.resolve_operands(instruction.operands)?;
        Ok(instruction)
    }

    fn bind_temp(&mut self, register: u8, allocated: u8) -> Result<(), ExpansionError> {
        let index = RowTemplate::temporary_index(register)?;
        if self.temps.len() <= index {
            self.temps.resize(index + 1, None);
        }
        if self.temps[index].is_some() {
            return Err(ExpansionError::DuplicateTemporaryRegister { register });
        }
        self.temps[index] = Some(allocated);
        Ok(())
    }

    fn resolve_register_for_release(&mut self, register: u8) -> Result<u8, ExpansionError> {
        if !RowTemplate::is_temporary_register(register) {
            return Ok(register);
        }
        let index = RowTemplate::temporary_index(register)?;
        let resolved = self
            .temps
            .get_mut(index)
            .and_then(Option::take)
            .ok_or(ExpansionError::UnallocatedTemporaryRegister { register })?;
        Ok(resolved)
    }

    fn resolve_operands(
        &self,
        operands: NormalizedOperands,
    ) -> Result<NormalizedOperands, ExpansionError> {
        Ok(NormalizedOperands {
            rd: self.resolve_optional_register(operands.rd)?,
            rs1: self.resolve_optional_register(operands.rs1)?,
            rs2: self.resolve_optional_register(operands.rs2)?,
            imm: operands.imm,
        })
    }

    fn resolve_optional_register(
        &self,
        register: Option<u8>,
    ) -> Result<Option<u8>, ExpansionError> {
        register
            .map(|register| self.resolve_register(register))
            .transpose()
    }

    fn resolve_register(&self, register: u8) -> Result<u8, ExpansionError> {
        if !RowTemplate::is_temporary_register(register) {
            return Ok(register);
        }
        let index = RowTemplate::temporary_index(register)?;
        self.temps
            .get(index)
            .copied()
            .flatten()
            .ok_or(ExpansionError::UnallocatedTemporaryRegister { register })
    }

    fn finish(self) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        self.rows.check_capacity()?;
        stamp_sequence(self.rows.into_vec(), self.is_compressed)
    }
}

#[cfg(test)]
mod tests {
    use jolt_riscv::{JoltInstructionKind, NormalizedOperands};

    use super::*;

    fn source() -> NormalizedInstruction {
        NormalizedInstruction {
            instruction_kind: JoltInstructionKind::ADDIW,
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd: Some(3),
                rs1: Some(4),
                rs2: None,
                imm: 1,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    #[test]
    fn materializer_rejects_duplicate_temp_allocation() -> Result<(), ExpansionError> {
        let temp = RowTemplate::temporary(0)?;
        let sequence = ExpandedInstructionSequence {
            source: source(),
            ops: vec![ExpansionOp::Allocate(temp), ExpansionOp::Allocate(temp)],
        };
        let mut state = ExpansionState::new(ExpansionAllocator::new());

        assert!(matches!(
            state.materialize(sequence),
            Err(ExpansionError::DuplicateTemporaryRegister { register }) if register == temp
        ));
        Ok(())
    }

    #[test]
    fn materializer_rejects_unallocated_temp_use() -> Result<(), ExpansionError> {
        let temp = RowTemplate::temporary(0)?;
        let sequence = ExpandedInstructionSequence {
            source: source(),
            ops: vec![ExpansionOp::Emit(RowTemplate::i(
                JoltInstructionKind::ADDI,
                temp,
                1,
                0,
            ))],
        };
        let mut state = ExpansionState::new(ExpansionAllocator::new());

        assert!(matches!(
            state.materialize(sequence),
            Err(ExpansionError::UnallocatedTemporaryRegister { register }) if register == temp
        ));
        Ok(())
    }

    #[test]
    fn materializer_rejects_unallocated_temp_release() -> Result<(), ExpansionError> {
        let temp = RowTemplate::temporary(0)?;
        let sequence = ExpandedInstructionSequence {
            source: source(),
            ops: vec![ExpansionOp::Release(temp)],
        };
        let mut state = ExpansionState::new(ExpansionAllocator::new());

        assert!(matches!(
            state.materialize(sequence),
            Err(ExpansionError::UnallocatedTemporaryRegister { register }) if register == temp
        ));
        Ok(())
    }
}

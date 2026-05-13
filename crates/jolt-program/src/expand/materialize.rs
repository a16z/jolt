use jolt_riscv::{
    JoltInstructionProfile, JoltRow, NormalizedOperands, SourceInstruction, SourceRow,
};

use crate::expand::{
    allocator::{ExpansionAllocator, NUM_VIRTUAL_INSTRUCTION_REGISTERS},
    expand_source_only_instruction,
    grammar::{
        is_source_only, ExpandedInstructionSequence, ExpansionOp, RegisterOperand, RowTemplate,
        SourceRowTemplate, TempId, TemplateOperands,
    },
    metadata::stamp_instruction_sequence,
    operands::{handles_rd_zero_internally, noop_for},
    ExpansionError,
};

pub(super) const MAX_FINAL_ROWS_PER_SOURCE: usize = 64;

/// Materializes symbolic recipes into concrete instructions (phase 2).
pub(super) struct ExpansionState {
    allocator: ExpansionAllocator,
    profile: JoltInstructionProfile,
}

impl ExpansionState {
    pub(super) fn new(allocator: ExpansionAllocator, profile: JoltInstructionProfile) -> Self {
        Self { allocator, profile }
    }

    pub(super) fn into_allocator(self) -> ExpansionAllocator {
        self.allocator
    }

    pub(super) fn expand_source_recursive(
        &mut self,
        instruction: &SourceInstruction,
    ) -> Result<Vec<JoltRow>, ExpansionError> {
        self.allocator.enter_expansion()?;
        let result = self.dispatch_source(instruction);
        self.allocator.exit_expansion();
        result
    }

    /// Routes: rd=x0 rewrite → recurse, native → pass-through, source-only → build recipe + materialize.
    fn dispatch_source(
        &mut self,
        instruction: &SourceInstruction,
    ) -> Result<Vec<JoltRow>, ExpansionError> {
        let kind = instruction.kind();
        if instruction.row().operands.rd == Some(0) && !handles_rd_zero_internally(kind) {
            if kind.jolt_kind().has_side_effects() {
                let virtual_register = self.allocate_register()?;
                let rewritten = (*instruction).map_row(|mut row| {
                    row.operands.rd = Some(virtual_register);
                    row
                });
                let expanded = self.expand_source_recursive(&rewritten);
                self.release_register(virtual_register)?;
                return expanded;
            }
            return Ok(vec![noop_for(instruction.jolt_row())]);
        }

        if kind == jolt_riscv::SourceInstructionKind::Inline {
            return Err(ExpansionError::InlineProviderRequired);
        }
        if !is_source_only(kind) {
            return Ok(vec![instruction.jolt_row()]);
        }
        let sequence = expand_source_only_instruction(instruction)?;
        self.materialize(sequence)
    }

    pub(super) fn allocate_register(&mut self) -> Result<u8, ExpansionError> {
        self.allocator.allocate()
    }

    pub(super) fn release_register(&mut self, register: u8) -> Result<(), ExpansionError> {
        self.allocator.release(register)
    }

    pub(super) fn materialize(
        &mut self,
        sequence: ExpandedInstructionSequence,
    ) -> Result<Vec<JoltRow>, ExpansionError> {
        let mut materializer = SequenceMaterializer::new(sequence.source, self.profile);
        for op in sequence.ops {
            match op {
                ExpansionOp::Emit(row) => materializer.emit(row)?,
                ExpansionOp::Expand(row) => {
                    let instruction = materializer.source_instruction(row)?;
                    materializer.extend(self.expand_source_recursive(&instruction)?)?;
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
}

/// Bounded output collector — rejects sequences exceeding `MAX_FINAL_ROWS_PER_SOURCE`.
#[derive(Debug)]
struct ExpansionBuffer {
    rows: Vec<JoltRow>,
}

impl ExpansionBuffer {
    fn new() -> Self {
        Self {
            rows: Vec::with_capacity(MAX_FINAL_ROWS_PER_SOURCE),
        }
    }

    fn push(&mut self, row: JoltRow) -> Result<(), ExpansionError> {
        if self.rows.len() == MAX_FINAL_ROWS_PER_SOURCE {
            return Err(ExpansionError::CapacityExceeded {
                actual: self.rows.len() + 1,
                capacity: MAX_FINAL_ROWS_PER_SOURCE,
            });
        }
        self.rows.push(row);
        Ok(())
    }

    fn extend_vec(&mut self, rows: Vec<JoltRow>) -> Result<(), ExpansionError> {
        for row in rows {
            self.push(row)?;
        }
        Ok(())
    }

    fn check_capacity(&self) -> Result<(), ExpansionError> {
        if self.rows.len() > MAX_FINAL_ROWS_PER_SOURCE {
            return Err(ExpansionError::CapacityExceeded {
                actual: self.rows.len(),
                capacity: MAX_FINAL_ROWS_PER_SOURCE,
            });
        }
        Ok(())
    }

    fn into_vec(self) -> Vec<JoltRow> {
        self.rows
    }
}

/// Maps symbolic `TempId`s to physical virtual registers for one recipe materialization.
struct TempBindings {
    slots: [Option<u8>; NUM_VIRTUAL_INSTRUCTION_REGISTERS],
}

impl TempBindings {
    fn new() -> Self {
        Self {
            slots: [None; NUM_VIRTUAL_INSTRUCTION_REGISTERS],
        }
    }

    fn bind(&mut self, temp: TempId, allocated: u8) -> Result<(), ExpansionError> {
        let index = temp.index();
        if self.slots[index].is_some() {
            return Err(ExpansionError::DuplicateTemporaryRegister { index });
        }
        self.slots[index] = Some(allocated);
        Ok(())
    }

    fn get(&self, temp: TempId) -> Result<u8, ExpansionError> {
        let index = temp.index();
        self.slots[index].ok_or(ExpansionError::UnallocatedTemporaryRegister { index })
    }

    fn take(&mut self, temp: TempId) -> Result<u8, ExpansionError> {
        let index = temp.index();
        match self.slots[index].take() {
            Some(register) => Ok(register),
            None => Err(ExpansionError::UnallocatedTemporaryRegister { index }),
        }
    }

    fn first_leaked(&self) -> Option<usize> {
        self.slots.iter().position(Option::is_some)
    }
}

/// Executes a single recipe: resolves temps, collects output rows, checks capacity.
struct SequenceMaterializer {
    address: usize,
    is_compressed: bool,
    profile: JoltInstructionProfile,
    rows: ExpansionBuffer,
    temps: TempBindings,
}

impl SequenceMaterializer {
    fn new(source: JoltRow, profile: JoltInstructionProfile) -> Self {
        Self {
            address: source.address,
            is_compressed: source.is_compressed,
            profile,
            rows: ExpansionBuffer::new(),
            temps: TempBindings::new(),
        }
    }

    fn emit(&mut self, row: RowTemplate) -> Result<(), ExpansionError> {
        let row = self.instruction(row)?;
        self.rows.push(row)
    }

    fn extend(&mut self, rows: Vec<JoltRow>) -> Result<(), ExpansionError> {
        self.rows.extend_vec(rows)
    }

    fn instruction(&self, row: RowTemplate) -> Result<JoltRow, ExpansionError> {
        Ok(JoltRow {
            instruction_kind: row.instruction_kind,
            address: self.address,
            operands: self.resolve_operands(row.operands)?,
            virtual_sequence_remaining: Some(0),
            is_first_in_sequence: false,
            is_compressed: false,
        })
    }

    fn source_instruction(
        &self,
        row: SourceRowTemplate,
    ) -> Result<SourceInstruction, ExpansionError> {
        Ok(SourceInstruction::new(
            row.instruction_kind,
            SourceRow {
                address: self.address,
                operands: self.resolve_operands(row.operands)?,
                inline: None,
                is_compressed: false,
            },
        ))
    }

    fn bind_temp(&mut self, temp: TempId, allocated: u8) -> Result<(), ExpansionError> {
        self.temps.bind(temp, allocated)
    }

    fn resolve_register_for_release(&mut self, temp: TempId) -> Result<u8, ExpansionError> {
        self.temps.take(temp)
    }

    fn resolve_operands(
        &self,
        operands: TemplateOperands,
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
        register: Option<RegisterOperand>,
    ) -> Result<Option<u8>, ExpansionError> {
        match register {
            Some(register) => Ok(Some(self.resolve_register(register)?)),
            None => Ok(None),
        }
    }

    fn resolve_register(&self, register: RegisterOperand) -> Result<u8, ExpansionError> {
        match register {
            RegisterOperand::Register(register) => Ok(register),
            RegisterOperand::Temp(temp) => self.temps.get(temp),
        }
    }

    fn finish(self) -> Result<Vec<JoltRow>, ExpansionError> {
        if let Some(index) = self.temps.first_leaked() {
            return Err(ExpansionError::LeakedTemporaryRegister { index });
        }
        self.rows.check_capacity()?;
        stamp_instruction_sequence(self.rows.into_vec(), self.is_compressed, self.profile)
    }
}

#[cfg(test)]
mod tests {
    use jolt_riscv::{JoltInstructionKind, NormalizedOperands, RV64IMAC_JOLT};

    use crate::expand::grammar::reg;

    use super::*;

    fn source() -> JoltRow {
        JoltRow {
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
        let temp = TempId(0);
        let sequence = ExpandedInstructionSequence {
            source: source(),
            ops: vec![ExpansionOp::Allocate(temp), ExpansionOp::Allocate(temp)],
        };
        let mut state = ExpansionState::new(ExpansionAllocator::new(), RV64IMAC_JOLT);

        assert!(matches!(
            state.materialize(sequence),
            Err(ExpansionError::DuplicateTemporaryRegister { index }) if index == temp.index()
        ));
        Ok(())
    }

    #[test]
    fn materializer_rejects_unallocated_temp_use() -> Result<(), ExpansionError> {
        let temp = TempId(0);
        let sequence = ExpandedInstructionSequence {
            source: source(),
            ops: vec![ExpansionOp::Emit(RowTemplate::i(
                JoltInstructionKind::ADDI,
                temp.operand(),
                reg(1),
                0,
            ))],
        };
        let mut state = ExpansionState::new(ExpansionAllocator::new(), RV64IMAC_JOLT);

        assert!(matches!(
            state.materialize(sequence),
            Err(ExpansionError::UnallocatedTemporaryRegister { index }) if index == temp.index()
        ));
        Ok(())
    }

    #[test]
    fn materializer_rejects_unallocated_temp_release() -> Result<(), ExpansionError> {
        let temp = TempId(0);
        let sequence = ExpandedInstructionSequence {
            source: source(),
            ops: vec![ExpansionOp::Release(temp)],
        };
        let mut state = ExpansionState::new(ExpansionAllocator::new(), RV64IMAC_JOLT);

        assert!(matches!(
            state.materialize(sequence),
            Err(ExpansionError::UnallocatedTemporaryRegister { index }) if index == temp.index()
        ));
        Ok(())
    }

    #[test]
    fn materializer_rejects_leaked_temp() -> Result<(), ExpansionError> {
        let temp = TempId(0);
        let sequence = ExpandedInstructionSequence {
            source: source(),
            ops: vec![ExpansionOp::Allocate(temp)],
        };
        let mut state = ExpansionState::new(ExpansionAllocator::new(), RV64IMAC_JOLT);

        assert!(matches!(
            state.materialize(sequence),
            Err(ExpansionError::LeakedTemporaryRegister { index }) if index == temp.index()
        ));
        Ok(())
    }
}

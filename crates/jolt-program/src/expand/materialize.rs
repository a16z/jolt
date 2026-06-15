use jolt_riscv::{
    JoltInstructionProfile, JoltInstructionRow, NormalizedOperands, SourceInstruction,
    SourceInstructionRow,
};
use std::marker::PhantomData;

use crate::expand::{
    allocator::{ExpansionAllocator, NUM_INLINE_REGISTERS, NUM_VIRTUAL_INSTRUCTION_REGISTERS},
    expand_source_only_instruction,
    grammar::{
        is_source_only, ExpandedInstructionSequence, ExpansionOp, InlineTempId, RegisterOperand,
        RowTemplate, SourceInstructionRowTemplate, TempId, TemplateOperands,
    },
    metadata::{stamp_inline_sequence, stamp_instruction_sequence},
    operands::{handles_rd_zero_internally, noop_for},
    ExpansionError,
};

pub(super) const MAX_FINAL_ROWS_PER_SOURCE: usize = 64;
pub(super) const MAX_INLINE_ROWS_PER_SOURCE: usize = u16::MAX as usize + 1;

type StampSequenceFn = fn(
    Vec<JoltInstructionRow>,
    bool,
    JoltInstructionProfile,
) -> Result<Vec<JoltInstructionRow>, ExpansionError>;

struct MaterializeOptions {
    capacity: usize,
    append_inline_resets: bool,
    stamp: StampSequenceFn,
}

/// Materializes symbolic recipes into concrete final rows.
///
/// `ExpansionState` is the only component that binds symbolic temps to physical
/// virtual registers. It also owns recursive source-only expansion, so nested
/// helper recipes share allocator state with their parent and cannot collide
/// with a top-level `rd = x0` rewrite or a registered inline allocation.
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

    /// Expand a source instruction while preserving allocator recursion state.
    ///
    /// Recursive expansion is used when a recipe emits a source-only helper row.
    /// It rejects unbounded recursion and routes all rows through the same
    /// `rd = x0`, profile, and materialization rules as top-level expansion.
    pub(super) fn expand_source_recursive(
        &mut self,
        instruction: &SourceInstruction,
    ) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
        self.allocator.enter_expansion()?;
        let result = self.dispatch_source(instruction);
        self.allocator.exit_expansion();
        result
    }

    /// Route one source instruction to its final-row representation.
    ///
    /// The order matters: `rd = x0` is handled before provider dispatch or
    /// source-only lowering, native target rows pass through unchanged, and
    /// built-in source-only rows are lowered into a recipe and materialized.
    fn dispatch_source(
        &mut self,
        instruction: &SourceInstruction,
    ) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
        let kind = instruction.kind();
        if instruction.row().operands.rd == Some(0) && !handles_rd_zero_internally(kind) {
            if kind.has_side_effects() {
                let virtual_register = self.allocate_register()?;
                let mut row = *instruction.row();
                row.operands.rd = Some(virtual_register);
                let rewritten = SourceInstruction::new(kind, row);
                let expanded = self.expand_source_recursive(&rewritten);
                self.release_register(virtual_register)?;
                return expanded;
            }
            return Ok(vec![noop_for(*instruction.row())]);
        }

        if kind == jolt_riscv::SourceInstructionKind::Inline {
            return Err(ExpansionError::InlineProviderRequired);
        }
        if !is_source_only(kind) {
            return JoltInstructionRow::try_from(instruction)
                .map(|row| vec![row])
                .map_err(ExpansionError::IllegalSourceInstruction);
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

    /// Materialize a built-in source-only recipe and stamp it as one sequence.
    ///
    /// This path is bounded by `MAX_FINAL_ROWS_PER_SOURCE` because ordinary
    /// source-only instructions should expand into small fixed recipes.
    pub(super) fn materialize(
        &mut self,
        sequence: ExpandedInstructionSequence,
    ) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
        self.materialize_with_options(
            sequence,
            MaterializeOptions {
                capacity: MAX_FINAL_ROWS_PER_SOURCE,
                append_inline_resets: false,
                stamp: stamp_instruction_sequence,
            },
        )
    }

    /// Materialize a registered inline recipe, append reset rows, and stamp it.
    ///
    /// Inline recipes can be much larger than built-in instruction recipes, so
    /// they use the full `u16` virtual-sequence range. Reset rows are appended
    /// here, after all provider-owned inline registers have been released, so
    /// virtual-register cleanup is centralized and independent of tracer RAII.
    pub(super) fn materialize_inline(
        &mut self,
        sequence: ExpandedInstructionSequence,
    ) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
        self.materialize_with_options(
            sequence,
            MaterializeOptions {
                capacity: MAX_INLINE_ROWS_PER_SOURCE,
                append_inline_resets: true,
                stamp: stamp_inline_sequence,
            },
        )
    }

    fn materialize_with_options(
        &mut self,
        sequence: ExpandedInstructionSequence,
        options: MaterializeOptions,
    ) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
        let source = sequence.source;
        let mut rows = self.materialize_unstamped(sequence, options.capacity)?;
        if options.append_inline_resets {
            for register in self.allocator.take_registers_for_reset()? {
                rows.push(JoltInstructionRow {
                    instruction_kind: jolt_riscv::JoltInstructionKind::ADDI,
                    address: source.address,
                    operands: NormalizedOperands {
                        rd: Some(register),
                        rs1: Some(0),
                        rs2: None,
                        imm: 0,
                    },
                    virtual_sequence_remaining: Some(0),
                    is_first_in_sequence: false,
                    is_compressed: false,
                })?;
            }
        }
        (options.stamp)(rows.into_vec(), source.is_compressed, self.profile)
    }

    fn materialize_unstamped(
        &mut self,
        sequence: ExpandedInstructionSequence,
        capacity: usize,
    ) -> Result<ExpansionBuffer, ExpansionError> {
        let mut materializer = SequenceMaterializer::new(sequence.source, capacity);
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
                ExpansionOp::AllocateInline(register) => {
                    let allocated = self.allocator.allocate_for_inline()?;
                    materializer.bind_inline_temp(register, allocated)?;
                }
                ExpansionOp::ReleaseInline(register) => {
                    let register = materializer.resolve_inline_register_for_release(register)?;
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
    rows: Vec<JoltInstructionRow>,
    capacity: usize,
}

impl ExpansionBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            rows: Vec::new(),
            capacity,
        }
    }

    fn push(&mut self, row: JoltInstructionRow) -> Result<(), ExpansionError> {
        if self.rows.len() == self.capacity {
            return Err(ExpansionError::CapacityExceeded {
                actual: self.rows.len() + 1,
                capacity: self.capacity,
            });
        }
        self.rows.push(row);
        Ok(())
    }

    fn extend_vec(&mut self, rows: Vec<JoltInstructionRow>) -> Result<(), ExpansionError> {
        for row in rows {
            self.push(row)?;
        }
        Ok(())
    }

    fn check_capacity(&self) -> Result<(), ExpansionError> {
        if self.rows.len() > self.capacity {
            return Err(ExpansionError::CapacityExceeded {
                actual: self.rows.len(),
                capacity: self.capacity,
            });
        }
        Ok(())
    }

    fn into_vec(self) -> Vec<JoltInstructionRow> {
        self.rows
    }
}

trait TempBindingId: Copy {
    fn index(self) -> usize;
}

impl TempBindingId for TempId {
    fn index(self) -> usize {
        TempId::index(self)
    }
}

impl TempBindingId for InlineTempId {
    fn index(self) -> usize {
        InlineTempId::index(self)
    }
}

/// Maps symbolic ids to physical virtual registers for one recipe materialization.
struct TempBindings<Id, const N: usize> {
    slots: [Option<u8>; N],
    _marker: PhantomData<fn(Id)>,
}

impl<Id: TempBindingId, const N: usize> TempBindings<Id, N> {
    fn new() -> Self {
        Self {
            slots: [None; N],
            _marker: PhantomData,
        }
    }

    fn bind(&mut self, temp: Id, allocated: u8) -> Result<(), ExpansionError> {
        let index = temp.index();
        if self.slots[index].is_some() {
            return Err(ExpansionError::DuplicateTemporaryRegister { index });
        }
        self.slots[index] = Some(allocated);
        Ok(())
    }

    fn get(&self, temp: Id) -> Result<u8, ExpansionError> {
        let index = temp.index();
        self.slots[index].ok_or(ExpansionError::UnallocatedTemporaryRegister { index })
    }

    fn take(&mut self, temp: Id) -> Result<u8, ExpansionError> {
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
    rows: ExpansionBuffer,
    temps: TempBindings<TempId, NUM_VIRTUAL_INSTRUCTION_REGISTERS>,
    inline_temps: TempBindings<InlineTempId, NUM_INLINE_REGISTERS>,
}

impl SequenceMaterializer {
    fn new(source: SourceInstructionRow, capacity: usize) -> Self {
        Self {
            address: source.address,
            rows: ExpansionBuffer::new(capacity),
            temps: TempBindings::new(),
            inline_temps: TempBindings::new(),
        }
    }

    fn emit(&mut self, row: RowTemplate) -> Result<(), ExpansionError> {
        let row = self.instruction(row)?;
        self.rows.push(row)
    }

    fn extend(&mut self, rows: Vec<JoltInstructionRow>) -> Result<(), ExpansionError> {
        self.rows.extend_vec(rows)
    }

    fn instruction(&self, row: RowTemplate) -> Result<JoltInstructionRow, ExpansionError> {
        Ok(JoltInstructionRow {
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
        row: SourceInstructionRowTemplate,
    ) -> Result<SourceInstruction, ExpansionError> {
        Ok(SourceInstruction::new(
            row.instruction_kind,
            SourceInstructionRow {
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

    fn bind_inline_temp(
        &mut self,
        temp: InlineTempId,
        allocated: u8,
    ) -> Result<(), ExpansionError> {
        self.inline_temps.bind(temp, allocated)
    }

    fn resolve_register_for_release(&mut self, temp: TempId) -> Result<u8, ExpansionError> {
        self.temps.take(temp)
    }

    fn resolve_inline_register_for_release(
        &mut self,
        temp: InlineTempId,
    ) -> Result<u8, ExpansionError> {
        self.inline_temps.take(temp)
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
            RegisterOperand::InlineTemp(temp) => self.inline_temps.get(temp),
        }
    }

    fn finish(self) -> Result<ExpansionBuffer, ExpansionError> {
        if let Some(index) = self.temps.first_leaked() {
            return Err(ExpansionError::LeakedTemporaryRegister { index });
        }
        if let Some(index) = self.inline_temps.first_leaked() {
            return Err(ExpansionError::LeakedTemporaryRegister { index });
        }
        self.rows.check_capacity()?;
        Ok(self.rows)
    }
}

#[cfg(test)]
mod tests {
    use jolt_riscv::{
        JoltInstructionKind, NormalizedOperands, SourceInstructionRow, RV64IMAC_JOLT,
    };

    use crate::expand::grammar::reg;

    use super::*;

    fn source() -> SourceInstructionRow {
        SourceInstructionRow {
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd: Some(3),
                rs1: Some(4),
                rs2: None,
                imm: 1,
            },
            inline: None,
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

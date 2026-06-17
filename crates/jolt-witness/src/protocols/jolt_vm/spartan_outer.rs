//! Stage 1 Spartan outer-sumcheck rows.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVmSpartanOuterRow {
    pub left_instruction_input: u64,
    pub right_instruction_input: i128,
    pub product_magnitude: u128,
    pub product_is_positive: bool,
    pub should_branch: bool,
    pub pc: u64,
    pub unexpanded_pc: u64,
    pub imm: i128,
    pub ram_address: u64,
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_write_value: u64,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub next_unexpanded_pc: u64,
    pub next_pc: u64,
    pub next_is_virtual: bool,
    pub next_is_first_in_sequence: bool,
    pub lookup_output: u64,
    pub should_jump: bool,
    pub flag_add_operands: bool,
    pub flag_subtract_operands: bool,
    pub flag_multiply_operands: bool,
    pub flag_load: bool,
    pub flag_store: bool,
    pub flag_jump: bool,
    pub flag_write_lookup_output_to_rd: bool,
    pub flag_virtual_instruction: bool,
    pub flag_assert: bool,
    pub flag_do_not_update_unexpanded_pc: bool,
    pub flag_advice: bool,
    pub flag_is_compressed: bool,
    pub flag_is_first_in_sequence: bool,
    pub flag_is_last_in_sequence: bool,
}

pub trait JoltVmSpartanOuterRows {
    fn spartan_outer_rows(&self) -> Result<Vec<JoltVmSpartanOuterRow>, WitnessError>;
}

impl<T: TraceSource + Clone> JoltVmSpartanOuterRows for TraceBackedJoltVmWitness<'_, T> {
    fn spartan_outer_rows(&self) -> Result<Vec<JoltVmSpartanOuterRow>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        let mut pc_cache = PcLookupCache::default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            values.push(spartan_outer_row(
                &current,
                next.as_ref(),
                self.preprocessing,
                &mut pc_cache,
            )?);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }
}

pub(crate) fn spartan_outer_row(
    row: &TraceRow,
    next: Option<&TraceRow>,
    preprocessing: &JoltProgramPreprocessing,
    pc_cache: &mut PcLookupCache,
) -> Result<JoltVmSpartanOuterRow, WitnessError> {
    let instruction = JoltInstruction::try_from(row.instruction).map_err(|kind| {
        WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        }
    })?;
    let circuit_flags = instruction.circuit_flags();
    let instruction_flags = instruction.instruction_flags();
    let query = JoltLookupQuery::new(row.instruction.instruction_kind, row);

    let (left_instruction_input, right_instruction_input) =
        LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
    let product = S64::from_u64(left_instruction_input)
        .mul_trunc::<2, 2>(&S128::from_i128(right_instruction_input));
    let (left_lookup_operand, right_lookup_operand) =
        LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
    let lookup_output = LookupQuery::<RV64_XLEN>::to_lookup_output(&query);
    let rs1_value = row.registers.rs1.map_or(0, |read| read.value);
    let rs2_value = row.registers.rs2.map_or(0, |read| read.value);
    let rd_write_value = row.registers.rd.map_or(0, |write| write.post_value);
    let (ram_address, ram_read_value, ram_write_value) = match row.ram_access {
        RamAccess::Read(read) => (read.address, read.value, read.value),
        RamAccess::Write(write) => (write.address, write.pre_value, write.post_value),
        RamAccess::NoOp => (0, 0, 0),
    };
    let pc = pc_cache.pc_for_row(row, preprocessing)? as u64;
    let next_pc = next
        .map(|row| pc_cache.pc_for_row(row, preprocessing).map(|pc| pc as u64))
        .transpose()?
        .unwrap_or(0);
    let next_unexpanded_pc = next.map_or(0, |row| row.instruction.address as u64);
    let next_is_noop = next.is_some_and(row_is_noop);
    let next_is_virtual =
        next.is_some_and(|row| row.instruction.virtual_sequence_remaining.is_some());
    let next_is_first_in_sequence = next.is_some_and(|row| row.instruction.is_first_in_sequence);

    Ok(JoltVmSpartanOuterRow {
        left_instruction_input,
        right_instruction_input,
        product_magnitude: product.magnitude_as_u128(),
        product_is_positive: product.is_positive,
        should_branch: instruction_flags[InstructionFlags::Branch] && lookup_output == 1,
        pc,
        unexpanded_pc: row.instruction.address as u64,
        imm: row.instruction.operands.imm,
        ram_address,
        rs1_value,
        rs2_value,
        rd_write_value,
        ram_read_value,
        ram_write_value,
        left_lookup_operand,
        right_lookup_operand,
        next_unexpanded_pc,
        next_pc,
        next_is_virtual,
        next_is_first_in_sequence,
        lookup_output,
        should_jump: circuit_flags[CircuitFlags::Jump] && !next_is_noop,
        flag_add_operands: circuit_flags[CircuitFlags::AddOperands],
        flag_subtract_operands: circuit_flags[CircuitFlags::SubtractOperands],
        flag_multiply_operands: circuit_flags[CircuitFlags::MultiplyOperands],
        flag_load: circuit_flags[CircuitFlags::Load],
        flag_store: circuit_flags[CircuitFlags::Store],
        flag_jump: circuit_flags[CircuitFlags::Jump],
        flag_write_lookup_output_to_rd: circuit_flags[CircuitFlags::WriteLookupOutputToRD],
        flag_virtual_instruction: circuit_flags[CircuitFlags::VirtualInstruction],
        flag_assert: circuit_flags[CircuitFlags::Assert],
        flag_do_not_update_unexpanded_pc: circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC],
        flag_advice: circuit_flags[CircuitFlags::Advice],
        flag_is_compressed: circuit_flags[CircuitFlags::IsCompressed],
        flag_is_first_in_sequence: circuit_flags[CircuitFlags::IsFirstInSequence],
        flag_is_last_in_sequence: circuit_flags[CircuitFlags::IsLastInSequence],
    })
}

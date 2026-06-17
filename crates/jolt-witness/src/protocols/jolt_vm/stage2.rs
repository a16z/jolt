//! Stage 2 trace rows.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVmStage2TraceRow {
    pub remapped_ram_address: Option<usize>,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub ram_increment: i128,
    pub left_instruction_input: u64,
    pub right_instruction_input: i128,
    pub lookup_output: u64,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub branch_flag: bool,
    pub jump_flag: bool,
    pub write_lookup_output_to_rd_flag: bool,
    pub virtual_instruction_flag: bool,
    pub next_is_noop: bool,
}

pub trait JoltVmStage2Rows {
    fn stage2_rows(&self) -> Result<Vec<JoltVmStage2TraceRow>, WitnessError>;

    fn initial_ram_state_words(&self) -> Result<Vec<u64>, WitnessError>;

    fn final_ram_state_words(&self) -> Result<Vec<u64>, WitnessError>;
}

impl<T: TraceSource + Clone> JoltVmStage2Rows for TraceBackedJoltVmWitness<'_, T> {
    fn stage2_rows(&self) -> Result<Vec<JoltVmStage2TraceRow>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            let remapped_ram_address = ram_access_address(current.ram_access)
                .map(|address| self.remapped_ram_address(address))
                .transpose()?
                .flatten();
            values.push(stage2_trace_row(
                &current,
                next.as_ref(),
                remapped_ram_address,
            )?);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }

    fn initial_ram_state_words(&self) -> Result<Vec<u64>, WitnessError> {
        self.initial_ram_state()
    }

    fn final_ram_state_words(&self) -> Result<Vec<u64>, WitnessError> {
        self.final_ram_state()
    }
}

pub(crate) fn stage2_trace_row(
    row: &TraceRow,
    next: Option<&TraceRow>,
    remapped_ram_address: Option<usize>,
) -> Result<JoltVmStage2TraceRow, WitnessError> {
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
    let (left_lookup_operand, right_lookup_operand) =
        LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
    let lookup_output = LookupQuery::<RV64_XLEN>::to_lookup_output(&query);
    let (ram_read_value, ram_write_value) = match row.ram_access {
        RamAccess::Read(read) => (read.value, read.value),
        RamAccess::Write(write) => (write.pre_value, write.post_value),
        RamAccess::NoOp => (0, 0),
    };
    let next_is_noop = next.is_none_or(row_is_noop);

    Ok(JoltVmStage2TraceRow {
        remapped_ram_address,
        ram_read_value,
        ram_write_value,
        ram_increment: JoltVmIncrementStreamKind::RamInc.value_from_row(row),
        left_instruction_input,
        right_instruction_input,
        lookup_output,
        left_lookup_operand,
        right_lookup_operand,
        branch_flag: instruction_flags[InstructionFlags::Branch],
        jump_flag: circuit_flags[CircuitFlags::Jump],
        write_lookup_output_to_rd_flag: circuit_flags[CircuitFlags::WriteLookupOutputToRD],
        virtual_instruction_flag: circuit_flags[CircuitFlags::VirtualInstruction],
        next_is_noop,
    })
}

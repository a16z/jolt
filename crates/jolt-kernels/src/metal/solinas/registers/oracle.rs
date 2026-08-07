use std::collections::BTreeMap;

use super::owner::{
    RdIncrement, RegisterCsr256Parts, RegisterOwnerRead, RegisterOwnerRow,
    REGISTER_CSR_BLOCK_CYCLES, REGISTER_CSR_COLUMNS,
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct OracleColumn {
    start_value: u64,
    rs1_positions: Vec<u8>,
    rs2_positions: Vec<u8>,
    rd_positions: Vec<u8>,
    rd_post_values: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DenseBtreeRegisterOracle {
    cycles: usize,
    columns: BTreeMap<(usize, usize), OracleColumn>,
    final_values: [u64; REGISTER_CSR_COLUMNS],
    rd_increments: Vec<RdIncrement>,
}

impl DenseBtreeRegisterOracle {
    pub(super) fn build(
        rows: &[RegisterOwnerRow],
        initial_values: &[u64; REGISTER_CSR_COLUMNS],
    ) -> Result<Self, RegisterOracleError> {
        let mut state_timeline = Vec::with_capacity(rows.len() + 1);
        let mut state = *initial_values;
        state_timeline.push(state);
        let mut rd_increments = Vec::new();

        for (cycle, row) in rows.iter().enumerate() {
            if let Some(read) = row.rs1 {
                check_read(read, &state)?;
            }
            if let Some(read) = row.rs2 {
                check_read(read, &state)?;
            }
            if let Some(write) = row.rd {
                let register = register_index(write.register)?;
                if state[register] != write.pre_value {
                    return Err(RegisterOracleError::WritePreMismatch);
                }
                let increment = i128::from(write.post_value) - i128::from(write.pre_value);
                if increment != 0 {
                    let cycle = u32::try_from(cycle)
                        .map_err(|_| RegisterOracleError::CycleCountOverflow)?;
                    rd_increments.push(RdIncrement { cycle, increment });
                }
                state[register] = write.post_value;
            }
            state_timeline.push(state);
        }

        let block_count = rows.len().div_ceil(REGISTER_CSR_BLOCK_CYCLES);
        let mut columns = BTreeMap::new();
        for block in 0..block_count {
            let cycle_start = block * REGISTER_CSR_BLOCK_CYCLES;
            let cycle_end = (cycle_start + REGISTER_CSR_BLOCK_CYCLES).min(rows.len());
            for (register, start_value) in state_timeline[cycle_start].iter().copied().enumerate() {
                let mut column = OracleColumn {
                    start_value,
                    ..OracleColumn::default()
                };
                for (position, row) in rows[cycle_start..cycle_end].iter().enumerate() {
                    let position = u8::try_from(position)
                        .map_err(|_| RegisterOracleError::CycleCountOverflow)?;
                    if row
                        .rs1
                        .is_some_and(|read| usize::from(read.register) == register)
                    {
                        column.rs1_positions.push(position);
                    }
                    if row
                        .rs2
                        .is_some_and(|read| usize::from(read.register) == register)
                    {
                        column.rs2_positions.push(position);
                    }
                    if let Some(write) = row.rd {
                        if usize::from(write.register) == register {
                            column.rd_positions.push(position);
                            column.rd_post_values.push(write.post_value);
                        }
                    }
                }
                let _ = columns.insert((block, register), column);
            }
        }

        Ok(Self {
            cycles: rows.len(),
            columns,
            final_values: state,
            rd_increments,
        })
    }

    pub(super) fn to_parts(&self) -> Result<RegisterCsr256Parts, RegisterOracleError> {
        let mut start_values = Vec::with_capacity(self.columns.len());
        let mut rs1_offsets = vec![0];
        let mut rs2_offsets = vec![0];
        let mut rd_offsets = vec![0];
        let mut rs1_positions = Vec::new();
        let mut rs2_positions = Vec::new();
        let mut rd_positions = Vec::new();
        let mut rd_post_values = Vec::new();

        for column in self.columns.values() {
            start_values.push(column.start_value);
            rs1_positions.extend_from_slice(&column.rs1_positions);
            rs1_offsets.push(event_offset(rs1_positions.len())?);
            rs2_positions.extend_from_slice(&column.rs2_positions);
            rs2_offsets.push(event_offset(rs2_positions.len())?);
            rd_positions.extend_from_slice(&column.rd_positions);
            rd_post_values.extend_from_slice(&column.rd_post_values);
            rd_offsets.push(event_offset(rd_positions.len())?);
        }

        Ok(RegisterCsr256Parts {
            cycles: self.cycles,
            start_values,
            rs1_offsets,
            rs2_offsets,
            rd_offsets,
            rs1_positions,
            rs2_positions,
            rd_positions,
            rd_post_values,
        })
    }

    pub(super) const fn final_values(&self) -> &[u64; REGISTER_CSR_COLUMNS] {
        &self.final_values
    }

    pub(super) fn rd_increments(&self) -> &[RdIncrement] {
        &self.rd_increments
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RegisterOracleError {
    CycleCountOverflow,
    InvalidRegister,
    ReadMismatch,
    WritePreMismatch,
}

fn register_index(register: u8) -> Result<usize, RegisterOracleError> {
    let register = usize::from(register);
    if register >= REGISTER_CSR_COLUMNS {
        return Err(RegisterOracleError::InvalidRegister);
    }
    Ok(register)
}

fn check_read(
    read: RegisterOwnerRead,
    state: &[u64; REGISTER_CSR_COLUMNS],
) -> Result<(), RegisterOracleError> {
    let register = register_index(read.register)?;
    if state[register] != read.value {
        return Err(RegisterOracleError::ReadMismatch);
    }
    Ok(())
}

fn event_offset(events: usize) -> Result<u32, RegisterOracleError> {
    u32::try_from(events).map_err(|_| RegisterOracleError::CycleCountOverflow)
}

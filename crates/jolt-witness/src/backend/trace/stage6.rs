//! Stage 6 rows.

use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoltVmStage6Row {
    pub instruction_lookup_index: u128,
    pub bytecode_index: usize,
    pub remapped_ram_address: Option<usize>,
    pub ram_access_nonzero: bool,
    pub ram_increment: i128,
    pub rd_increment: i128,
}

pub trait JoltVmStage6Rows {
    fn stage6_rows(&self) -> Result<Vec<JoltVmStage6Row>, WitnessError>;
}

impl<T: TraceSource + Clone> JoltVmStage6Rows for TraceBackend<'_, T> {
    fn stage6_rows(&self) -> Result<Vec<JoltVmStage6Row>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut pc_cache = PcLookupCache::default();
        for _ in 0..rows {
            let Some(row) = trace.next_row() else {
                values.push(JoltVmStage6Row::default());
                continue;
            };
            let instruction_lookup_index =
                instruction_lookup_index::<RV64_XLEN>(&row).map_err(|error| {
                    WitnessError::InvalidWitnessData {
                        label: JOLT_VM_LABEL,
                        reason: error.to_string(),
                    }
                })?;
            let bytecode_index = pc_cache
                .pc_for_row_optional(&row, self.preprocessing)
                .unwrap_or(0);
            let ram_address = ram_access_address(row.ram_access);
            let remapped_ram_address = ram_address
                .map(|address| self.remapped_ram_address(address))
                .transpose()?
                .flatten();
            values.push(JoltVmStage6Row {
                instruction_lookup_index,
                bytecode_index,
                remapped_ram_address,
                ram_access_nonzero: ram_address.is_some_and(|address| address != 0),
                ram_increment: JoltVmIncrementStreamKind::RamInc.value_from_row(&row),
                rd_increment: JoltVmIncrementStreamKind::RdInc.value_from_row(&row),
            });
        }
        Ok(values)
    }
}

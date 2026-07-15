//! Cycle-domain virtual polynomial materialization over the atomic
//! extractors.

use super::extract::{cycle_witness_value, row_is_noop, supported_trace_virtual, WitnessEnv};
use super::*;

impl<T: TraceSource + Clone> TraceBackedJoltVmWitness<'_, T> {
    pub(crate) fn materialize_trace_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        if !supported_trace_virtual(id) {
            return Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            });
        }

        let rows = checked_pow2(self.config.log_t)?;
        let env = WitnessEnv {
            preprocessing: self.preprocessing,
        };
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            values.push(cycle_witness_value::<F>(id, &current, next.as_ref(), &env)?);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PcLookupCache {
    values: HashMap<(usize, u16), usize>,
}

impl PcLookupCache {
    pub(crate) fn pc_for_row_optional(
        &mut self,
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Option<usize> {
        if row_is_noop(row) {
            return Some(0);
        }
        let key = pc_lookup_key(row);
        if let Some(&pc) = self.values.get(&key) {
            return Some(pc);
        }
        let pc = preprocessing.bytecode.get_pc(&row.instruction)?;
        let _ = self.values.insert(key, pc);
        Some(pc)
    }
}

pub(crate) fn pc_lookup_key(row: &TraceRow) -> (usize, u16) {
    (
        row.instruction.address,
        row.instruction.virtual_sequence_remaining.unwrap_or(0),
    )
}

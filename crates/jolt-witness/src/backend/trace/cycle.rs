//! The sequential cycle walk driving the atomic extractors.

use super::*;
use crate::witnesses::{row_is_noop, Extract, ExtractIndexed, ToField, WitnessEnv};

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    /// Materializes one cycle-domain witness column by walking the trace
    /// once; all per-witness logic lives on `W`.
    pub(crate) fn materialize_cycle<F: Field, W: Extract + ToField>(
        &self,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, next, env| W::extract(row, next, env).map(ToField::to_field))
    }

    /// [`Self::materialize_cycle`] for indexed witness families; `index`
    /// selects the family member.
    pub(crate) fn materialize_cycle_indexed<F: Field, W: ExtractIndexed<I> + ToField, I: Copy>(
        &self,
        index: I,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, next, env| {
            W::extract_indexed(index, row, next, env).map(ToField::to_field)
        })
    }

    /// One pass over `2^log_t` cycles with the one-row lookahead window;
    /// rows beyond the trace are padding (default) rows.
    fn walk_cycles<F>(
        &self,
        mut value: impl FnMut(&TraceRow, Option<&TraceRow>, &WitnessEnv<'_>) -> Result<F, WitnessError>,
    ) -> Result<Vec<F>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let env = WitnessEnv {
            preprocessing: self.preprocessing,
        };
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            values.push(value(&current, next.as_ref(), &env)?);
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

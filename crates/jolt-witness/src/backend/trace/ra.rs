//! Virtual instruction-RA grid materialization.

use super::*;
use crate::witnesses::{Extract, LookupIndex, RaChunkSelector, WitnessEnv};

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    pub(crate) fn materialize_instruction_ra<F: Field>(
        &self,
        index: usize,
    ) -> Result<Vec<F>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let chunk_bits = self.config.one_hot.lookup_virtual_chunk_bits();
        let chunks = self.instruction_virtual_ra_count()?;
        require_index(index, chunks)?;
        let selector = RaChunkSelector::new(index, chunks, chunk_bits)?;
        let addresses = checked_pow2(chunk_bits)?;
        let env = WitnessEnv {
            preprocessing: self.preprocessing,
        };
        let mut values = vec![F::zero(); addresses * cycles];
        let mut trace = self.trace.trace.clone();

        for cycle in 0..cycles {
            let value = trace.next_row().map_or_else(
                || Ok(selector.chunk_u128(0)),
                |row| {
                    LookupIndex::extract(&row, None, &env)
                        .map(|lookup_index| selector.chunk_u128(lookup_index.0))
                },
            )?;
            values[value * cycles + cycle] = F::one();
        }

        Ok(values)
    }
}

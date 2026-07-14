//! One-hot RA chunk selection.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RaChunkSelector {
    shift: usize,
    mask: u128,
}

impl RaChunkSelector {
    pub(crate) fn new(
        index: usize,
        chunks: usize,
        chunk_bits: usize,
    ) -> Result<Self, WitnessError> {
        let remaining = chunks
            .checked_sub(index + 1)
            .ok_or(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            })?;
        let shift =
            remaining
                .checked_mul(chunk_bits)
                .ok_or_else(|| WitnessError::InvalidDimensions {
                    namespace: JOLT_VM_NAMESPACE.name,
                    reason: "RA chunk shift overflow".to_owned(),
                })?;
        let k = checked_pow2_u128(chunk_bits)?;
        Ok(Self { shift, mask: k - 1 })
    }

    pub(crate) const fn chunk_usize(self, value: usize) -> usize {
        self.chunk_u128(value as u128)
    }

    pub(crate) const fn chunk_u128(self, value: u128) -> usize {
        ((value >> self.shift) & self.mask) as usize
    }
}

impl<T: TraceSource + Clone> TraceBackedJoltVmWitness<'_, T> {
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
        let mut values = vec![F::zero(); addresses * cycles];
        let mut trace = self.trace.trace.clone();

        for cycle in 0..cycles {
            let value = trace.next_row().map_or_else(
                || Ok(selector.chunk_u128(0)),
                |row| {
                    instruction_lookup_index::<RV64_XLEN>(&row)
                        .map(|lookup_index| selector.chunk_u128(lookup_index))
                        .map_err(|error| WitnessError::InvalidWitnessData {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: error.to_string(),
                        })
                },
            )?;
            values[value * cycles + cycle] = F::one();
        }

        Ok(values)
    }
}

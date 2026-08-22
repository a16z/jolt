//! Advice-column materialization: canonical zero-padded little-endian words.

use super::*;

impl<T: TraceSource> TraceBackend<T> {
    pub(crate) fn materialize_trusted_advice<F: Field>(&self) -> Result<Vec<F>, WitnessError> {
        materialize_advice(
            "trusted",
            &self.trace.device.trusted_advice,
            self.preprocessing.memory_layout.max_trusted_advice_size as usize,
        )
    }

    pub(crate) fn materialize_untrusted_advice<F: Field>(&self) -> Result<Vec<F>, WitnessError> {
        materialize_advice(
            "untrusted",
            &self.trace.device.untrusted_advice,
            self.preprocessing.memory_layout.max_untrusted_advice_size as usize,
        )
    }
}

/// An advice column's word count: the power-of-two number of little-endian
/// words covering the configured maximum size. Single-sources the column
/// length for [`TraceBackend::shape_of`].
pub(super) fn advice_words(max_bytes: usize) -> usize {
    (max_bytes / 8).next_power_of_two().max(1)
}

fn materialize_advice<F: Field>(
    kind: &str,
    bytes: &[u8],
    max_bytes: usize,
) -> Result<Vec<F>, WitnessError> {
    common::advice::canonical_advice_words(bytes, max_bytes)
        .map(|words| words.into_iter().map(F::from_u64).collect())
        .map_err(|error| WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!("invalid {kind} advice: {error}"),
        })
}

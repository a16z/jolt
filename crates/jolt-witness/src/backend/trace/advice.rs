//! Advice-column materialization: device bytes packed as little-endian
//! words, zero-padded to the column's power-of-two word count.

use super::*;

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
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
/// length between [`TraceBackend::shape_of`] and the materialization below.
pub(super) fn advice_words(max_bytes: usize) -> usize {
    (max_bytes / 8).next_power_of_two().max(1)
}

fn materialize_advice<F: Field>(
    kind: &str,
    bytes: &[u8],
    max_bytes: usize,
) -> Result<Vec<F>, WitnessError> {
    if bytes.len() > max_bytes {
        return Err(WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!(
                "{kind} advice has {} bytes, exceeding configured max {max_bytes}",
                bytes.len()
            ),
        });
    }
    Ok((0..advice_words(max_bytes))
        .map(|word| F::from_u64(advice_word_le(bytes, word)))
        .collect())
}

fn advice_word_le(bytes: &[u8], word_index: usize) -> u64 {
    let Some(start) = word_index.checked_mul(8) else {
        return 0;
    };
    if start >= bytes.len() {
        return 0;
    }
    let end = start.saturating_add(8).min(bytes.len());
    let mut word = [0_u8; 8];
    word[..end - start].copy_from_slice(&bytes[start..end]);
    u64::from_le_bytes(word)
}

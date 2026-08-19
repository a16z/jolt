//! Advice-column materialization: device bytes packed as little-endian
//! words, zero-padded to the column's power-of-two word count.

use super::*;

impl<T: TraceSource + Clone> TraceBackend<T> {
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

    pub(crate) fn materialize_trusted_advice_bytes<F: Field>(
        &self,
    ) -> Result<Vec<F>, WitnessError> {
        materialize_advice_bytes(
            "trusted",
            &self.trace.device.trusted_advice,
            self.preprocessing.memory_layout.max_trusted_advice_size as usize,
        )
    }

    pub(crate) fn materialize_untrusted_advice_bytes<F: Field>(
        &self,
    ) -> Result<Vec<F>, WitnessError> {
        materialize_advice_bytes(
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

/// The dense byte one-hot cell table of an advice column, over the
/// `(byte ‖ place ‖ word)` cell domain: per `(place ‖ word)` the hot value is
/// the advice byte, zero-padded past the actual advice length — the same zero
/// padding the word column carries, so every `(place, word)` pair holds
/// exactly one hot byte (the hamming leg of the untrusted reconstruction).
fn materialize_advice_bytes<F: Field>(
    kind: &str,
    bytes: &[u8],
    max_bytes: usize,
) -> Result<Vec<F>, WitnessError> {
    use jolt_claims::protocols::jolt::lattice::geometry::{word_byte_num_vars, WORD_BYTES};

    if bytes.len() > max_bytes {
        return Err(WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!(
                "{kind} advice has {} bytes, exceeding configured max {max_bytes}",
                bytes.len()
            ),
        });
    }
    let words = advice_words(max_bytes);
    let word_vars = words.ilog2() as usize;
    let limb_bits = WORD_BYTES.ilog2() as usize;
    let mut values = vec![F::zero(); checked_pow2(word_byte_num_vars(word_vars))?];
    for word_index in 0..words {
        let word = advice_word_le(bytes, word_index);
        for limb in 0..WORD_BYTES {
            let byte = (word >> (8 * limb)) as u8 as usize;
            values[(((byte << limb_bits) | limb) << word_vars) | word_index] = F::one();
        }
    }
    Ok(values)
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

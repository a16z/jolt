//! Canonical trusted/untrusted advice word encoding.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::jolt_device::bytes_to_words_le;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdviceWordsError {
    AdviceTooLong { actual: usize, max: usize },
}

impl fmt::Display for AdviceWordsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdviceTooLong { actual, max } => {
                write!(
                    formatter,
                    "advice has {actual} bytes, exceeding configured max {max}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AdviceWordsError {}

/// Packs bytes into the canonical zero-padded little-endian advice word table.
pub fn canonical_advice_words(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<Vec<u64>, AdviceWordsError> {
    let word_capacity = (max_bytes / 8).next_power_of_two();
    if bytes.len() > max_bytes {
        return Err(AdviceWordsError::AdviceTooLong {
            actual: bytes.len(),
            max: max_bytes,
        });
    }
    let mut words = bytes_to_words_le(bytes);
    words.resize(word_capacity, 0);
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_words_are_little_endian_and_zero_padded() {
        assert_eq!(canonical_advice_words(&[], 0), Ok(vec![0]));
        assert_eq!(
            canonical_advice_words(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 16),
            Ok(vec![0x0807_0605_0403_0201, 9])
        );
        assert_eq!(
            canonical_advice_words(&[u8::MAX; 8], 16),
            Ok(vec![u64::MAX, 0])
        );
    }

    #[test]
    fn oversize_advice_is_rejected() {
        assert_eq!(
            canonical_advice_words(&[0; 9], 8),
            Err(AdviceWordsError::AdviceTooLong { actual: 9, max: 8 })
        );
    }

    /// `MemoryLayout::new` aligns advice capacities to 8 bytes and asserts they
    /// are powers of two, so the word domain rounds up rather than rejecting.
    #[test]
    fn unaligned_capacity_rounds_up_to_the_word_domain() {
        assert_eq!(canonical_advice_words(&[], 7), Ok(vec![0]));
        assert_eq!(canonical_advice_words(&[], 24), Ok(vec![0; 4]));
    }
}

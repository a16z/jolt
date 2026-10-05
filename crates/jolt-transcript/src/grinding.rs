//! Proof of work: squeeze-then-grind.
//!
//! A grind squeezes a seed from the live transcript, then searches nonces off
//! the transcript: a candidate passes when the first 32 bytes its
//! [`Fork`] squeezes start with `bits` zero bits, low bit first. The nonce is then sent as a `u32` prover
//! message, so the protected challenge drawn after it binds the solution.
//! Nonces are searched over `bits + GRINDING_NONCE_SLACK_BITS` bits, so an
//! honest search exhausts its range with probability about `exp(-2^7)`.

use std::num::NonZeroU8;

use crate::{Fork, Sponge};

/// Extra nonce bits searched beyond the difficulty.
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
/// Largest supported difficulty; the nonce range then fills a `u32`.
pub const MAX_GRINDING_BITS: u8 = u32::BITS as u8 - GRINDING_NONCE_SLACK_BITS;
/// Byte length of the squeezed proof-of-work predicate.
pub const GRINDING_PREDICATE_LEN: usize = 32;
/// Byte length of the seed a grind squeezes from the transcript.
pub const GRINDING_SEED_LEN: usize = crate::FORK_SEED_LEN;
/// Returns whether the low `bits` bits of `predicate`, low byte first and low
/// bit first within each byte, are all zero.
#[must_use]
pub fn grinding_predicate_accepts(
    predicate: &[u8; GRINDING_PREDICATE_LEN],
    bits: NonZeroU8,
) -> bool {
    let bits = usize::from(bits.get());
    let (whole, partial) = predicate.split_at(bits / 8);
    let remaining = bits % 8;
    whole.iter().all(|&byte| byte == 0)
        && (remaining == 0
            || partial
                .first()
                .is_some_and(|&byte| byte & ((1u8 << remaining) - 1) == 0))
}

/// Whether `nonce` passes `bits` bits of work under `seed`.
pub(crate) fn grinding_accepts<H: Sponge>(
    seed: &[u8; GRINDING_SEED_LEN],
    nonce: u32,
    bits: NonZeroU8,
) -> bool {
    grinding_predicate_accepts(&Fork::<H>::new(seed, nonce).squeeze(), bits)
}

/// The first nonce in `0..2^nonce_bits` that passes `bits` bits of work.
pub(crate) fn grind_nonce<H: Sponge>(
    seed: &[u8; GRINDING_SEED_LEN],
    bits: NonZeroU8,
    nonce_bits: u8,
) -> Option<u32> {
    (0..1u64 << nonce_bits)
        .filter_map(|candidate| u32::try_from(candidate).ok())
        .find(|&nonce| grinding_accepts::<H>(seed, nonce, bits))
}

/// Nonce search width for difficulty `bits`.
pub(crate) fn nonce_bits(bits: NonZeroU8) -> Option<u8> {
    (bits.get() <= MAX_GRINDING_BITS).then(|| bits.get() + GRINDING_NONCE_SLACK_BITS)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "test code")]
mod tests {
    use super::*;

    #[test]
    fn predicate_checks_low_bits_at_byte_boundaries() {
        for bits in [1, 7, 8, 9, 16, 17, MAX_GRINDING_BITS] {
            let nonzero = NonZeroU8::new(bits).unwrap();
            let mut predicate = [0u8; GRINDING_PREDICATE_LEN];
            assert!(grinding_predicate_accepts(&predicate, nonzero));
            let last = usize::from(bits - 1);
            predicate[last / 8] = 1 << (last % 8);
            assert!(!grinding_predicate_accepts(&predicate, nonzero));
            predicate[last / 8] = 0;
            let first_free = usize::from(bits);
            predicate[first_free / 8] |= 1 << (first_free % 8);
            assert!(grinding_predicate_accepts(&predicate, nonzero));
        }
    }
}

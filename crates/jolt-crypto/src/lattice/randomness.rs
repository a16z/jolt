//! Hiding randomness for Ajtai commitments (Whiteout §2.3, §4.1).
//!
//! Randomness is drawn uniformly from a box of balanced digits. Distinct box
//! vectors differ by a coordinate whose coefficients lie below `q^{1/16}/4`,
//! which makes it a unit (see `MAX_RANDOMNESS_LOG_BASIS`). So the
//! commitment map is a universal hash and the leftover hash lemma hides the
//! image once the box has enough entropy.

#[cfg(feature = "sample")]
use rand_core::{CryptoRng, RngCore};
#[cfg(feature = "sample")]
use zeroize::Zeroizing;

use super::ajtai::field_bits;
use crate::CryptoError;
use jolt_algebra::{CanonicalEncoding, Field};

/// The statistical security parameter λ: the hiding entropy exceeds the
/// image's bit length by `2λ`.
pub const HIDING_SECURITY_BITS: usize = 128;

/// The largest digit base `2^log_basis` whose digit differences are units in
/// `F_q[X]/(X^D + 1)` for the production field `Prime128Offset6A23801F`.
///
/// That prime satisfies `q ≡ 33 (mod 64)`, so for every `D ≥ 16`, `X^D + 1`
/// splits into 16 factors, and LS18 (Cor. 1.2) makes every nonzero element
/// with coefficients below `q^{1/16}/4` (just under 64) a unit. Balanced
/// digits of base `2^log_basis` differ by at most `2^log_basis − 1`: 63 at
/// this cap, 127 one base above it. `sample_box_digits` and
/// `required_randomness_columns` reject larger bases.
pub(super) const MAX_RANDOMNESS_LOG_BASIS: u32 = 6;

/// `len` digits drawn uniformly from the balanced alphabet
/// `[−2^{log_basis−1}, 2^{log_basis−1} − 1]`. Fails for `log_basis` outside
/// `1..=6`, the bases whose digit differences are units.
///
/// Each digit is one masked byte of `rng` output: no rejection, no branch
/// and no table, so the running time is independent of the digits. The
/// digits and the bytes they come from are zeroed when freed.
#[cfg(feature = "sample")]
pub fn sample_box_digits<R: RngCore + CryptoRng>(
    rng: &mut R,
    len: usize,
    log_basis: u32,
) -> Result<Zeroizing<Vec<i8>>, CryptoError> {
    if !(1..=MAX_RANDOMNESS_LOG_BASIS).contains(&log_basis) {
        return Err(CryptoError::InvalidInput(format!(
            "box digits need log_basis in 1..={MAX_RANDOMNESS_LOG_BASIS}, got {log_basis}"
        )));
    }
    let mask = ((1u16 << log_basis) - 1) as u8;
    let half = 1i16 << (log_basis - 1);
    let mut bytes = Zeroizing::new(vec![0u8; len]);
    rng.fill_bytes(&mut bytes);
    Ok(Zeroizing::new(
        bytes
            .iter()
            .map(|&byte| (i16::from(byte & mask) - half) as i8)
            .collect(),
    ))
}

/// The fewest whole ring columns of balanced `2^log_basis` digits whose
/// entropy is at least `(image_coeffs + extra_coeffs) · ⌈log₂ q⌉ + 2λ +
/// slack_bits`.
///
/// `image_coeffs` is the commitment image's coefficient count, and
/// `extra_coeffs` covers further outputs the same randomness must hide, such
/// as the 64 coefficients of Whiteout's row `L` on the pad commitment. Each
/// commitment slice draws its own randomness of this size. Fails for
/// `log_basis` outside `1..=6`, the bases whose digit differences are units.
pub fn required_randomness_columns<F: Field + CanonicalEncoding>(
    image_coeffs: usize,
    extra_coeffs: usize,
    ring_dim: usize,
    log_basis: u32,
    slack_bits: usize,
) -> Result<usize, CryptoError> {
    if !(1..=MAX_RANDOMNESS_LOG_BASIS).contains(&log_basis) {
        return Err(CryptoError::InvalidInput(format!(
            "randomness digits need log_basis in 1..={MAX_RANDOMNESS_LOG_BASIS}, got {log_basis}"
        )));
    }
    let overflow = || CryptoError::InvalidSetup("randomness size overflows usize".into());
    let field_bits = field_bits::<F>()? as usize;
    let needed = image_coeffs
        .checked_add(extra_coeffs)
        .and_then(|coeffs| coeffs.checked_mul(field_bits))
        .and_then(|bits| bits.checked_add(2 * HIDING_SECURITY_BITS))
        .and_then(|bits| bits.checked_add(slack_bits))
        .ok_or_else(overflow)?;
    let per_column = ring_dim
        .checked_mul(log_basis as usize)
        .filter(|&bits| bits > 0)
        .ok_or_else(|| CryptoError::InvalidInput("randomness columns need ring_dim > 0".into()))?;
    Ok(needed.div_ceil(per_column))
}

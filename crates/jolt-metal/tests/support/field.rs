//! Field inputs shared by the conformance suites: the field types under test,
//! boundary values, and fixed-seed random elements.

use std::fmt::Debug;

use jolt_field::{CanonicalEncoding, PseudoMersenne};
use jolt_metal::MetalField;

use super::support::SplitMix64;

/// The field types under test, with the CPU operations the harness needs.
pub trait TestField: MetalField + PseudoMersenne + CanonicalEncoding + Debug {}
impl<F: MetalField + PseudoMersenne + CanonicalEncoding + Debug> TestField for F {}

pub fn element<F: TestField>(value: u128) -> F {
    F::from_u128_checked(value).unwrap()
}

pub fn modulus<F: TestField>() -> u128 {
    0u128.wrapping_sub(F::OFFSET)
}

/// Canonical values at every boundary the arithmetic treats specially:
/// small values, `C` and its neighbours, word and limb boundaries, the
/// top of the field, and every value whose 32-bit words are each one of
/// `0`, `1`, `2^31`, `2^32 − 1`.
pub fn edges<F: TestField>() -> Vec<u128> {
    let p = modulus::<F>();
    let c = F::OFFSET;
    let mut values = vec![
        0,
        1,
        2,
        3,
        c - 1,
        c,
        c + 1,
        (1 << 32) - 1,
        1 << 32,
        (1 << 63) - 1,
        1 << 63,
        (1 << 64) - 1,
        1 << 64,
        (1 << 96) - 1,
        1 << 96,
        (1 << 127) - 1,
        1 << 127,
        p / 2,
        p / 2 + 1,
        p - c,
        p - 2,
        p - 1,
    ];
    let words = [0u128, 1, 1 << 31, (1 << 32) - 1];
    for pattern in 0..256u32 {
        let value = (0..4).fold(0u128, |value, i| {
            value | words[((pattern >> (2 * i)) & 3) as usize] << (32 * i)
        });
        values.push(value);
    }
    values.retain(|&v| v < p);
    values.sort_unstable();
    values.dedup();
    values
}

pub fn random_elements<F: TestField>(words: &mut SplitMix64, len: usize) -> Vec<u128> {
    let p = modulus::<F>();
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let v = u128::from(words.next().unwrap()) | u128::from(words.next().unwrap()) << 64;
        if v < p {
            values.push(v);
        }
    }
    values
}

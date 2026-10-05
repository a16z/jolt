//! `jolt_field` types with MSL counterparts in `shaders/jolt/field`.

use bytemuck::checked::CheckedBitPattern;
use bytemuck::{NoUninit, Zeroable};
use jolt_field::solinas::Fp128;
use jolt_field::Field;

use crate::runtime::MslType;

/// A `jolt_field` type whose MSL counterpart gives bit-identical results.
///
/// Values upload as their bytes ([`NoUninit`]) and read back only through a
/// canonical-form check ([`CheckedBitPattern`]), so a kernel that writes a
/// non-canonical value yields
/// [`MetalError::InvalidReadback`](crate::MetalError::InvalidReadback).
pub trait MetalField: Field + MslType + NoUninit + Zeroable + CheckedBitPattern {}

impl<const P: u128> MetalField for Fp128<P> {}

/// `Fp128<P>` is `jolt::Fp128<C>` in MSL, with `C = 2^128 − P` spelled from
/// `P` at compile time, so no offset is written by hand: `jolt::Fp128<0xffffa7f7u>`
/// with host suffix `fp128_ffffa7f7`.
impl<const P: u128> MslType for Fp128<P> {
    const MSL_NAME: &'static str = ascii(&Fp128Spelling::<P>::MSL_NAME);
    const HOST_SUFFIX: &'static str = ascii(&Fp128Spelling::<P>::HOST_SUFFIX);
}

struct Fp128Spelling<const P: u128>;

impl<const P: u128> Fp128Spelling<P> {
    /// `jolt_field` const-asserts `C < 2^32`, so the cast is exact.
    const OFFSET: u32 = Fp128::<P>::C as u32;
    const MSL_NAME: [u8; 24] = spell(b"jolt::Fp128<0x", Self::OFFSET, b"u>");
    const HOST_SUFFIX: [u8; 14] = spell(b"fp128_", Self::OFFSET, b"");
}

/// `prefix`, `value` as eight lowercase hex digits, then `suffix`.
#[expect(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    reason = "evaluated only in constants, where any failure is a build error"
)]
const fn spell<const N: usize>(prefix: &[u8], value: u32, suffix: &[u8]) -> [u8; N] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    assert!(
        prefix.len() + 8 + suffix.len() == N,
        "spelling length does not match its array"
    );
    let mut out = [0u8; N];
    let mut i = 0;
    while i < prefix.len() {
        out[i] = prefix[i];
        i += 1;
    }
    let mut digit = 0;
    while digit < 8 {
        out[i] = HEX[((value >> (28 - 4 * digit)) & 0xf) as usize];
        i += 1;
        digit += 1;
    }
    let mut j = 0;
    while j < suffix.len() {
        out[i] = suffix[j];
        i += 1;
        j += 1;
    }
    out
}

#[expect(
    clippy::panic,
    reason = "evaluated only in constants, where any failure is a build error"
)]
const fn ascii(bytes: &'static [u8]) -> &'static str {
    match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => panic!("spellings are ASCII"),
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::solinas::{Prime128Offset275, Prime128OffsetA7F7};

    use super::*;

    #[test]
    fn fp128_spellings_come_from_the_modulus() {
        assert_eq!(Prime128OffsetA7F7::MSL_NAME, "jolt::Fp128<0xffffa7f7u>");
        assert_eq!(Prime128OffsetA7F7::HOST_SUFFIX, "fp128_ffffa7f7");
        assert_eq!(Prime128Offset275::MSL_NAME, "jolt::Fp128<0x00000113u>");
        assert_eq!(Prime128Offset275::HOST_SUFFIX, "fp128_00000113");
    }
}

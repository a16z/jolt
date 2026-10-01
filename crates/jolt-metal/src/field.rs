//! `jolt_field` types with MSL counterparts in `shaders/jolt/field`.

use std::marker::PhantomData;

use bytemuck::checked::CheckedBitPattern;
use bytemuck::{NoUninit, Pod, Zeroable};
use jolt_field::solinas::{Ext2, Fp128, Fp64};
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

impl<const P: u64> MetalField for Fp64<P> {}

impl<F: MetalField> MetalField for Ext2<F> where F::Bits: Pod {}

/// `Fp128<P>` is `jolt::Fp128<C>` in MSL, with `C = 2^128 − P` spelled from
/// `P` at compile time, so no offset is written by hand: `jolt::Fp128<0xffffa7f7u>`
/// with host suffix `fp128_ffffa7f7`.
impl<const P: u128> MslType for Fp128<P> {
    const MSL_NAME: &'static str = Fp128Spelling::<P>::MSL_NAME.text();
    const HOST_SUFFIX: &'static str = Fp128Spelling::<P>::HOST_SUFFIX.text();
}

struct Fp128Spelling<const P: u128>;

impl<const P: u128> Fp128Spelling<P> {
    /// `jolt_field` const-asserts `C < 2^32`, so the cast is exact.
    const OFFSET: [u8; 8] = hex8(Fp128::<P>::C as u32);
    const MSL_NAME: Spelling = spell(&[b"jolt::Fp128<0x", &Self::OFFSET, b"u>"]);
    const HOST_SUFFIX: Spelling = spell(&[b"fp128_", &Self::OFFSET]);
}

/// `Fp64<P>` for a 64-bit `P` is `jolt::Fp64<C>` in MSL, with `C = 2^64 − P`
/// spelled from `P`: `jolt::Fp64<0x0000003bu>` with host suffix
/// `fp64_0000003b`. A modulus below `2^63` is a build error, since
/// `jolt::Fp64` folds at bit 64.
impl<const P: u64> MslType for Fp64<P> {
    const MSL_NAME: &'static str = Fp64Spelling::<P>::MSL_NAME.text();
    const HOST_SUFFIX: &'static str = Fp64Spelling::<P>::HOST_SUFFIX.text();
}

struct Fp64Spelling<const P: u64>;

impl<const P: u64> Fp64Spelling<P> {
    /// `jolt_field` const-asserts `C (C + 1) < P < 2^64`, so `C < 2^32` and
    /// the cast is exact.
    const OFFSET: [u8; 8] = {
        assert!(P >> 63 == 1, "jolt::Fp64 needs a 64-bit modulus");
        hex8(Fp64::<P>::C as u32)
    };
    const MSL_NAME: Spelling = spell(&[b"jolt::Fp64<0x", &Self::OFFSET, b"u>"]);
    const HOST_SUFFIX: Spelling = spell(&[b"fp64_", &Self::OFFSET]);
}

/// `Ext2<F>` (non-residue 2) is `jolt::Ext2<F>` in MSL, spelled from `F`:
/// `jolt::Ext2<jolt::Fp64<0x0000003bu>>` with host suffix
/// `ext2_fp64_0000003b`.
impl<F: MetalField> MslType for Ext2<F> {
    const MSL_NAME: &'static str = Ext2Spelling::<F>::MSL_NAME.text();
    const HOST_SUFFIX: &'static str = Ext2Spelling::<F>::HOST_SUFFIX.text();
}

struct Ext2Spelling<F>(PhantomData<F>);

impl<F: MslType> Ext2Spelling<F> {
    const MSL_NAME: Spelling = spell(&[b"jolt::Ext2<", F::MSL_NAME.as_bytes(), b">"]);
    const HOST_SUFFIX: Spelling = spell(&[b"ext2_", F::HOST_SUFFIX.as_bytes()]);
}

/// Longest spelling any field type needs, with room for one more level of
/// extension.
const SPELLING_CAPACITY: usize = 96;

/// An ASCII name assembled at compile time: the first `len` bytes.
struct Spelling {
    bytes: [u8; SPELLING_CAPACITY],
    len: usize,
}

impl Spelling {
    #[expect(
        clippy::panic,
        reason = "evaluated only in constants, where any failure is a build error"
    )]
    const fn text(&'static self) -> &'static str {
        let (text, _) = self.bytes.split_at(self.len);
        match std::str::from_utf8(text) {
            Ok(text) => text,
            Err(_) => panic!("spellings are ASCII"),
        }
    }
}

/// The concatenation of `parts`.
#[expect(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    reason = "evaluated only in constants, where any failure is a build error"
)]
const fn spell(parts: &[&[u8]]) -> Spelling {
    let mut bytes = [0u8; SPELLING_CAPACITY];
    let mut len = 0;
    let mut part = 0;
    while part < parts.len() {
        let mut i = 0;
        while i < parts[part].len() {
            bytes[len] = parts[part][i];
            len += 1;
            i += 1;
        }
        part += 1;
    }
    Spelling { bytes, len }
}

/// `value` as eight lowercase hex digits.
#[expect(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    reason = "evaluated only in constants, where any failure is a build error"
)]
const fn hex8(value: u32) -> [u8; 8] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 8];
    let mut digit = 0;
    while digit < 8 {
        out[digit] = HEX[((value >> (28 - 4 * digit)) & 0xf) as usize];
        digit += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use jolt_field::solinas::{Prime128Offset275, Prime128OffsetA7F7, Prime64Offset59};

    use super::*;

    #[test]
    fn spellings_come_from_the_modulus() {
        assert_eq!(Prime128OffsetA7F7::MSL_NAME, "jolt::Fp128<0xffffa7f7u>");
        assert_eq!(Prime128OffsetA7F7::HOST_SUFFIX, "fp128_ffffa7f7");
        assert_eq!(Prime128Offset275::MSL_NAME, "jolt::Fp128<0x00000113u>");
        assert_eq!(Prime128Offset275::HOST_SUFFIX, "fp128_00000113");
        assert_eq!(Prime64Offset59::MSL_NAME, "jolt::Fp64<0x0000003bu>");
        assert_eq!(Prime64Offset59::HOST_SUFFIX, "fp64_0000003b");
        assert_eq!(
            Ext2::<Prime64Offset59>::MSL_NAME,
            "jolt::Ext2<jolt::Fp64<0x0000003bu>>"
        );
        assert_eq!(Ext2::<Prime64Offset59>::HOST_SUFFIX, "ext2_fp64_0000003b");
        assert_eq!(
            Ext2::<Prime128Offset275>::MSL_NAME,
            "jolt::Ext2<jolt::Fp128<0x00000113u>>"
        );
    }
}

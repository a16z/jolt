//! Compact bitvector type for lookup table index substrings.
//!
//! During the sumcheck protocol over lookup tables, indices are decomposed
//! into prefix/suffix substrings. [`LookupBits`] represents these substrings
//! as a compact 17-byte bitvector (vs 32 bytes for a `u128` + `u8` struct), which matters
//! because millions of these are created during proving.

use crate::uninterleave_bits;
use std::fmt::Display;
use std::ops::BitAnd;

/// A bitvector representing a substring of a lookup index.
///
/// Stores up to 128 bits in a packed byte array with a length tag.
/// The `[u8; 16]` layout avoids the 16-byte alignment that `u128` requires,
/// reducing struct size from 32 bytes (`u128` + `u8` + 15 padding) to 17.
#[derive(Clone, Copy, Debug)]
pub struct LookupBits {
    bytes: [u8; 16],
    len: u8,
}

impl LookupBits {
    /// Creates a new bitvector from the low `len` bits of `bits`.
    ///
    /// Bits beyond position `len` are masked off.
    pub fn new(mut bits: u128, len: usize) -> Self {
        debug_assert!(len <= 128);
        if len < 128 {
            bits %= 1 << len;
        }
        Self {
            bytes: bits.to_le_bytes(),
            len: len as u8,
        }
    }

    /// Splits interleaved x/y bits into separate bitvectors.
    ///
    /// Even-position bits (from MSB perspective) become `x`,
    /// odd-position bits become `y`.
    pub fn uninterleave(&self) -> (Self, Self) {
        let (x_bits, y_bits) = uninterleave_bits(u128::from_le_bytes(self.bytes));
        let x = Self::new(x_bits as u128, (self.len / 2) as usize);
        let y = Self::new(y_bits as u128, (self.len - x.len) as usize);
        (x, y)
    }

    /// Splits into `(prefix, suffix)` where `suffix.len() == suffix_len`.
    pub fn split(&self, suffix_len: usize) -> (Self, Self) {
        let bits = u128::from_le_bytes(self.bytes);
        let suffix_bits = bits % (1 << suffix_len);
        let suffix = Self::new(suffix_bits, suffix_len);
        let prefix_bits = bits >> suffix_len;
        let prefix = Self::new(prefix_bits, self.len as usize - suffix_len);
        (prefix, suffix)
    }

    /// Pops and returns the most significant bit, decrementing `len`.
    pub fn pop_msb(&mut self) -> u8 {
        debug_assert!(!self.is_empty(), "pop_msb on empty LookupBits");
        let mut bits = u128::from_le_bytes(self.bytes);
        let msb = (bits >> (self.len - 1)) & 1;
        bits %= 1 << (self.len - 1);
        self.bytes = bits.to_le_bytes();
        self.len -= 1;
        msb as u8
    }

    /// Number of bits in this bitvector.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns `true` if this bitvector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of trailing zero bits.
    pub fn trailing_zeros(&self) -> u32 {
        std::cmp::min(
            u128::from_le_bytes(self.bytes).trailing_zeros(),
            self.len as u32,
        )
    }

    /// Number of leading one bits.
    pub fn leading_ones(&self) -> u32 {
        u128::from_le_bytes(self.bytes)
            .wrapping_shl(128 - self.len as u32)
            .leading_ones()
    }

    /// Returns the raw bits as `u128`.
    #[inline]
    fn as_u128(&self) -> u128 {
        u128::from_le_bytes(self.bytes)
    }
}

impl Display for LookupBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:0width$b}", self.as_u128(), width = self.len as usize)
    }
}

impl From<LookupBits> for u128 {
    #[inline]
    fn from(value: LookupBits) -> u128 {
        value.as_u128()
    }
}

impl From<LookupBits> for usize {
    #[inline]
    fn from(value: LookupBits) -> usize {
        value.as_u128() as usize
    }
}

impl From<LookupBits> for u32 {
    #[inline]
    fn from(value: LookupBits) -> u32 {
        value.as_u128() as u32
    }
}

impl From<LookupBits> for u64 {
    #[inline]
    fn from(value: LookupBits) -> u64 {
        value.as_u128() as u64
    }
}

impl From<&LookupBits> for u128 {
    #[inline]
    fn from(value: &LookupBits) -> u128 {
        value.as_u128()
    }
}

impl From<&LookupBits> for usize {
    #[inline]
    fn from(value: &LookupBits) -> usize {
        value.as_u128() as usize
    }
}

impl From<&LookupBits> for u32 {
    #[inline]
    fn from(value: &LookupBits) -> u32 {
        value.as_u128() as u32
    }
}

impl BitAnd<usize> for LookupBits {
    type Output = usize;

    #[inline]
    fn bitand(self, rhs: usize) -> Self::Output {
        self.as_u128() as usize & rhs
    }
}

impl PartialEq for LookupBits {
    fn eq(&self, other: &Self) -> bool {
        self.as_u128() == other.as_u128()
    }
}

impl Eq for LookupBits {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interleave_bits;

    #[test]
    fn new_masks_excess_bits() {
        let bits = LookupBits::new(0xFF, 4);
        assert_eq!(u128::from(bits), 0x0F);
        assert_eq!(bits.len(), 4);
    }

    #[test]
    fn split_roundtrip() {
        let bits = LookupBits::new(0b1101_0110, 8);
        let (prefix, suffix) = bits.split(4);
        assert_eq!(u128::from(prefix), 0b1101);
        assert_eq!(u128::from(suffix), 0b0110);
        assert_eq!(prefix.len(), 4);
        assert_eq!(suffix.len(), 4);
    }

    #[test]
    fn pop_msb_sequence() {
        let mut bits = LookupBits::new(0b101, 3);
        assert_eq!(bits.pop_msb(), 1);
        assert_eq!(bits.len(), 2);
        assert_eq!(bits.pop_msb(), 0);
        assert_eq!(bits.len(), 1);
        assert_eq!(bits.pop_msb(), 1);
        assert_eq!(bits.len(), 0);
    }

    #[test]
    fn uninterleave_matches_global() {
        let x: u64 = 0xDEAD;
        let y: u64 = 0xBEEF;
        let interleaved = interleave_bits(x, y);
        let bits = LookupBits::new(interleaved, 32);
        let (bx, by) = bits.uninterleave();
        assert_eq!(u64::from(bx), x);
        assert_eq!(u64::from(by), y);
    }

    #[test]
    fn trailing_zeros_and_leading_ones() {
        let bits = LookupBits::new(0b1110_1000, 8);
        assert_eq!(bits.trailing_zeros(), 3);
        assert_eq!(bits.leading_ones(), 3);
    }

    #[test]
    fn bitand_usize() {
        let bits = LookupBits::new(0xFF, 8);
        assert_eq!(bits & 0x0F, 0x0F);
    }

    #[test]
    fn display_format() {
        let bits = LookupBits::new(0b101, 4);
        assert_eq!(format!("{bits}"), "0101");
    }

    #[test]
    fn equality() {
        let a = LookupBits::new(42, 8);
        let b = LookupBits::new(42, 8);
        let c = LookupBits::new(43, 8);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}

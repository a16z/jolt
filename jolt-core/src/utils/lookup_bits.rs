use std::{fmt::Display, ops::BitAnd};

use allocative::Allocative;

use crate::utils::uninterleave_bits;

/// A bitvector type used to represent a (substring of a) lookup index.
#[derive(Clone, Copy, Debug, Allocative)]
pub struct LookupBits {
    // Use byte array instead of u128 to avoid 16-byte alignment for this struct.
    // This way, LookupBits occupies 17 bytes instead of 32.
    bytes: [u8; 16],
    len: u8,
}

impl LookupBits {
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

    pub fn uninterleave(&self) -> (Self, Self) {
        let (x_bits, y_bits) = uninterleave_bits(u128::from_le_bytes(self.bytes));
        let x = Self::new(x_bits as u128, (self.len / 2) as usize);
        let y = Self::new(y_bits as u128, (self.len - x.len) as usize);
        (x, y)
    }

    /// Splits `self` into a tuple (prefix, suffix) of `LookupBits`, where
    /// `suffix.len() == suffix_len`.
    pub fn split(&self, suffix_len: usize) -> (Self, Self) {
        let bits = u128::from_le_bytes(self.bytes);
        let suffix_bits = bits % (1 << suffix_len);
        let suffix = Self::new(suffix_bits, suffix_len);
        let prefix_bits = bits >> suffix_len;
        let prefix = Self::new(prefix_bits, self.len as usize - suffix_len);
        (prefix, suffix)
    }

    /// Pops the most significant bit from `self`, decrementing `len`.
    pub fn pop_msb(&mut self) -> u8 {
        let mut bits = u128::from_le_bytes(self.bytes);
        let msb = (bits >> (self.len - 1)) & 1;
        bits %= 1 << (self.len - 1);
        self.bytes = bits.to_le_bytes();
        self.len -= 1;
        msb as u8
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn trailing_zeros(&self) -> u32 {
        std::cmp::min(
            u128::from_le_bytes(self.bytes).trailing_zeros(),
            self.len as u32,
        )
    }

    pub fn leading_ones(&self) -> u32 {
        u128::from_le_bytes(self.bytes)
            .wrapping_shl(128 - self.len as u32)
            .leading_ones()
    }
}

impl Display for LookupBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:0width$b}",
            u128::from_le_bytes(self.bytes),
            width = self.len as usize
        )
    }
}

impl From<LookupBits> for u128 {
    fn from(value: LookupBits) -> u128 {
        u128::from_le_bytes(value.bytes)
    }
}

impl From<LookupBits> for usize {
    fn from(value: LookupBits) -> usize {
        u128::from_le_bytes(value.bytes).try_into().unwrap()
    }
}

impl From<LookupBits> for u32 {
    fn from(value: LookupBits) -> u32 {
        u128::from_le_bytes(value.bytes).try_into().unwrap()
    }
}

impl From<LookupBits> for u64 {
    fn from(value: LookupBits) -> u64 {
        u128::from_le_bytes(value.bytes).try_into().unwrap()
    }
}

impl From<&LookupBits> for u128 {
    fn from(value: &LookupBits) -> u128 {
        u128::from_le_bytes(value.bytes)
    }
}

impl From<&LookupBits> for usize {
    fn from(value: &LookupBits) -> usize {
        u128::from_le_bytes(value.bytes).try_into().unwrap()
    }
}

impl From<&LookupBits> for u32 {
    fn from(value: &LookupBits) -> u32 {
        u128::from_le_bytes(value.bytes).try_into().unwrap()
    }
}

impl BitAnd<usize> for LookupBits {
    type Output = usize;

    fn bitand(self, rhs: usize) -> Self::Output {
        let lhs = usize::from_le_bytes(self.bytes[0..size_of::<usize>()].try_into().unwrap());
        lhs & rhs
    }
}

impl PartialEq for LookupBits {
    fn eq(&self, other: &Self) -> bool {
        u128::from(self) == u128::from(other)
    }
}

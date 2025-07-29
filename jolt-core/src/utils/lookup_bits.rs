use std::fmt::Display;

use crate::utils::uninterleave_bits;

/// A bitvector type used to represent a (substring of a) lookup index.
#[derive(Clone, Copy, Debug)]
pub struct LookupBits {
    bits: u128,
    len: usize,
}

impl LookupBits {
    pub fn new(mut bits: u128, len: usize) -> Self {
        debug_assert!(len <= 128);
        if len < 128 {
            bits %= 1 << len;
        }
        Self { bits, len }
    }

    pub fn uninterleave(&self) -> (Self, Self) {
        let (x_bits, y_bits) = uninterleave_bits(self.bits);
        let x = Self::new(x_bits as u128, self.len / 2);
        let y = Self::new(y_bits as u128, self.len - x.len);
        (x, y)
    }

    /// Splits `self` into a tuple (prefix, suffix) of `LookupBits`, where
    /// `suffix.len() == suffix_len`.
    pub fn split(&self, suffix_len: usize) -> (Self, Self) {
        let suffix_bits = self.bits % (1 << suffix_len);
        let suffix = Self::new(suffix_bits, suffix_len);
        let prefix_bits = self.bits >> suffix_len;
        let prefix = Self::new(prefix_bits, self.len - suffix_len);
        (prefix, suffix)
    }

    /// Pops the most significant bit from `self`, decrementing `len`.
    pub fn pop_msb(&mut self) -> u8 {
        let msb = (self.bits >> (self.len - 1)) & 1;
        self.bits %= 1 << (self.len - 1);
        self.len -= 1;
        msb as u8
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn trailing_zeros(&self) -> u32 {
        std::cmp::min(self.bits.trailing_zeros(), self.len as u32)
    }

    pub fn leading_ones(&self) -> u32 {
        self.bits.wrapping_shl(64 - self.len as u32).leading_ones()
    }
}

impl Display for LookupBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:0width$b}", self.bits, width = self.len)
    }
}

impl From<LookupBits> for u128 {
    fn from(value: LookupBits) -> u128 {
        value.bits
    }
}

impl From<LookupBits> for usize {
    fn from(value: LookupBits) -> usize {
        value.bits.try_into().unwrap()
    }
}

impl From<LookupBits> for u32 {
    fn from(value: LookupBits) -> u32 {
        value.bits.try_into().unwrap()
    }
}

impl From<LookupBits> for u64 {
    fn from(value: LookupBits) -> u64 {
        value.bits.try_into().unwrap()
    }
}

impl From<&LookupBits> for u128 {
    fn from(value: &LookupBits) -> u128 {
        value.bits
    }
}

impl From<&LookupBits> for usize {
    fn from(value: &LookupBits) -> usize {
        value.bits.try_into().unwrap()
    }
}

impl From<&LookupBits> for u32 {
    fn from(value: &LookupBits) -> u32 {
        value.bits.try_into().unwrap()
    }
}

impl std::ops::Rem<usize> for &LookupBits {
    type Output = usize;

    fn rem(self, rhs: usize) -> Self::Output {
        (u128::from(self) % rhs as u128) as usize
    }
}

impl std::ops::Rem<usize> for LookupBits {
    type Output = usize;

    fn rem(self, rhs: usize) -> Self::Output {
        (u128::from(self) % rhs as u128) as usize
    }
}

impl PartialEq for LookupBits {
    fn eq(&self, other: &Self) -> bool {
        u128::from(self) == u128::from(other)
    }
}

use std::fmt::Display;

use allocative::Allocative;

use crate::utils::uninterleave_bits;

/// A bitvector type used to represent a (substring of a) lookup index.
#[derive(Clone, Copy, Debug, Allocative)]
pub struct LookupBits {
    bits: u64,
    len: usize,
}

impl LookupBits {
    pub fn new(mut bits: u64, len: usize) -> Self {
        debug_assert!(len <= 64);
        if len < 64 {
            bits %= 1 << len;
        }
        Self { bits, len }
    }

    pub fn uninterleave(&self) -> (Self, Self) {
        let (x_bits, y_bits) = uninterleave_bits(self.bits);
        // For odd lengths, we need to properly distribute the bits:
        // - Even positions (0,2,4,...) get ceil(len/2) bits
        // - Odd positions (1,3,5,...) get floor(len/2) bits
        let x_len = (self.len + 1) / 2;
        let y_len = self.len / 2;
        let x = Self::new(x_bits as u64, x_len);
        let y = Self::new(y_bits as u64, y_len);
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

impl From<LookupBits> for u64 {
    fn from(value: LookupBits) -> u64 {
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

impl From<&LookupBits> for u64 {
    fn from(value: &LookupBits) -> u64 {
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
        usize::from(self) % rhs
    }
}

impl std::ops::Rem<usize> for LookupBits {
    type Output = usize;

    fn rem(self, rhs: usize) -> Self::Output {
        usize::from(self) % rhs
    }
}

impl PartialEq for LookupBits {
    fn eq(&self, other: &Self) -> bool {
        u64::from(self) == u64::from(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uninterleave_even_length() {
        // Test with even length (should work as before)
        let bits = LookupBits::new(0b1010, 4);
        let (x, y) = bits.uninterleave();
        
        // For len=4: x should have 2 bits, y should have 2 bits
        assert_eq!(x.len(), 2);
        assert_eq!(y.len(), 2);
        assert_eq!(x.len() + y.len(), bits.len());
    }

    #[test]
    fn test_uninterleave_odd_length() {
        // Test with odd length (this was the bug)
        let bits = LookupBits::new(0b10101, 5);
        let (x, y) = bits.uninterleave();
        
        // For len=5: x should have 3 bits (ceil(5/2)), y should have 2 bits (floor(5/2))
        assert_eq!(x.len(), 3);
        assert_eq!(y.len(), 2);
        assert_eq!(x.len() + y.len(), bits.len());
        
        // Verify the bits are correctly distributed
        // x should contain bits from even positions (0,2,4)
        // y should contain bits from odd positions (1,3)
        assert_eq!(x.bits, 0b101); // bits from positions 0,2,4
        assert_eq!(y.bits, 0b10);  // bits from positions 1,3
    }

    #[test]
    fn test_uninterleave_edge_cases() {
        // Test edge cases
        let bits1 = LookupBits::new(0b1, 1);
        let (x1, y1) = bits1.uninterleave();
        assert_eq!(x1.len(), 1); // ceil(1/2) = 1
        assert_eq!(y1.len(), 0); // floor(1/2) = 0
        assert_eq!(x1.len() + y1.len(), bits1.len());

        let bits3 = LookupBits::new(0b111, 3);
        let (x3, y3) = bits3.uninterleave();
        assert_eq!(x3.len(), 2); // ceil(3/2) = 2
        assert_eq!(y3.len(), 1); // floor(3/2) = 1
        assert_eq!(x3.len() + y3.len(), bits3.len());
    }

    #[test]
    fn test_uninterleave_roundtrip() {
        // Test that the sum of lengths always equals the original length
        for len in 1..=64 {
            let bits = LookupBits::new(0x1234567890ABCDEF, len);
            let (x, y) = bits.uninterleave();
            assert_eq!(x.len() + y.len(), len, "Failed for len={}", len);
        }
    }
}

//! Bit interleaving utilities for two-operand lookup table indexing.
//!
//! In the Twist/Shout lookup argument, two XLEN-bit operands are combined
//! into a single `2*XLEN`-bit index by interleaving their bits. The first
//! operand occupies even positions and the second occupies odd positions.

/// Interleaves bits from two 64-bit operands into a 128-bit lookup index.
///
/// Bit `i` of `x` is placed at position `2*i + 1` (even indices from MSB perspective),
/// and bit `i` of `y` is placed at position `2*i` (odd indices).
///
/// This matches the convention in the Jolt paper where the combined index
/// has `x` bits at even positions and `y` bits at odd positions (MSB-first).
#[inline]
pub fn interleave_bits(x: u64, y: u64) -> u128 {
    let mut x_bits = x as u128;
    x_bits = (x_bits | (x_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    let mut y_bits = y as u128;
    y_bits = (y_bits | (y_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    (x_bits << 1) | y_bits
}

/// Recovers two 64-bit operands from an interleaved 128-bit lookup index.
///
/// Inverse of [`interleave_bits`]: extracts even-position bits into `x`
/// and odd-position bits into `y`.
#[inline]
pub fn uninterleave_bits(val: u128) -> (u64, u64) {
    let mut x_bits = (val >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    let mut y_bits = val & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    x_bits = (x_bits | (x_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;

    y_bits = (y_bits | (y_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;

    (x_bits as u64, y_bits as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_small() {
        let x: u64 = 0b01;
        let y: u64 = 0b10;
        let interleaved = interleave_bits(x, y);
        assert_eq!(interleaved, 0b0110);
        let (rx, ry) = uninterleave_bits(interleaved);
        assert_eq!((rx, ry), (x, y));
    }

    #[test]
    fn roundtrip_random_patterns() {
        let pairs: &[(u64, u64)] = &[
            (0, 0),
            (u64::MAX, u64::MAX),
            (u64::MAX, 0),
            (0, u64::MAX),
            (0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0),
            (1, 1),
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
        ];
        for &(x, y) in pairs {
            let interleaved = interleave_bits(x, y);
            let (rx, ry) = uninterleave_bits(interleaved);
            assert_eq!((rx, ry), (x, y), "roundtrip failed for ({x:#x}, {y:#x})");
        }
    }
}

//! Bit interleaving utilities for two-operand lookup table indexing.
//!
//! In the Twist/Shout lookup argument, two XLEN-bit operands are combined
//! into a single `2*XLEN`-bit index by interleaving their bits. Counting from
//! the least significant bit, `x` occupies odd positions and `y` even positions.

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
/// Inverse of [`interleave_bits`]: counting from the least significant bit,
/// extracts odd-position bits into `x` and even-position bits into `y`.
#[inline]
pub fn uninterleave_bits(val: u128) -> (u64, u64) {
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "neon"
    ))]
    {
        use core::arch::aarch64::{
            vandq_u8, vcombine_u64, vcreate_u64, vdupq_n_u8, vget_lane_u64, vld1q_u8, vmovn_u16,
            vorrq_u16, vorrq_u8, vqtbl1q_u8, vreinterpret_u64_u8, vreinterpretq_u16_u8,
            vreinterpretq_u8_u64, vshlq_n_u8, vshrq_n_u16, vshrq_n_u8,
        };
        // SAFETY: this branch requires NEON. The table load reads all sixteen
        // bytes of the array; the remaining intrinsics operate only on registers.
        unsafe {
            let bytes = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(val as u64),
                vcreate_u64((val >> 64) as u64),
            ));
            // Map each nibble's two y bits to bits 0..1 and two x bits to 4..5.
            let table =
                vld1q_u8([0, 1, 16, 17, 2, 3, 18, 19, 32, 33, 48, 49, 34, 35, 50, 51].as_ptr());
            let lo = vqtbl1q_u8(table, vandq_u8(bytes, vdupq_n_u8(15)));
            let hi = vqtbl1q_u8(table, vshrq_n_u8::<4>(bytes));
            let bytes = vorrq_u8(lo, vshlq_n_u8::<2>(hi));
            let x = vreinterpretq_u16_u8(vshrq_n_u8::<4>(bytes));
            let y = vreinterpretq_u16_u8(vandq_u8(bytes, vdupq_n_u8(15)));
            let x = vmovn_u16(vorrq_u16(x, vshrq_n_u16::<4>(x)));
            let y = vmovn_u16(vorrq_u16(y, vshrq_n_u16::<4>(y)));
            (
                vget_lane_u64::<0>(vreinterpret_u64_u8(x)),
                vget_lane_u64::<0>(vreinterpret_u64_u8(y)),
            )
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "neon"
    )))]
    {
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

    #[test]
    fn uninterleave_matches_bit_positions() {
        fn check(value: u128) {
            let mut expected = (0_u64, 0_u64);
            for bit in 0..64 {
                expected.0 |= (((value >> (2 * bit + 1)) & 1) as u64) << bit;
                expected.1 |= (((value >> (2 * bit)) & 1) as u64) << bit;
            }
            assert_eq!(uninterleave_bits(value), expected);
        }
        check(0);
        check(u128::MAX);
        check(0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa);
        check(0x5555_5555_5555_5555_5555_5555_5555_5555);
        for bit in 0..128 {
            check(1_u128 << bit);
            check(!(1_u128 << bit));
        }
        let mut state = 0x572b_e4b3_e836_9f27_5204_9077_e55c_ba7b_u128;
        for _ in 0..2048 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            check(state);
        }
    }
}

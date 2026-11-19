/// Writes the requested suffix of the integer's big-endian radix digits.
#[inline(always)]
pub(super) fn fill_digits<const BITS: usize>(mut value: u128, columns: &mut [u8]) {
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "neon"
    ))]
    if BITS == 4 && (16..=32).contains(&columns.len()) {
        // SAFETY: NEON is enabled for this target. The load reads a complete
        // 16-byte array that remains live for the call. Each store writes 16 bytes
        // within a slice of length 16..=32. The overlapping stores agree on
        // their common digits, so no padding or extra output storage is needed.
        unsafe {
            use std::arch::aarch64::*;
            let words = vsetq_lane_u64::<1>((value >> 64) as u64, vdupq_n_u64(value as u64));
            let swapped = vrev64q_u8(vreinterpretq_u8_u64(words));
            let bytes = vextq_u8::<8>(swapped, swapped);
            let hi = vshrq_n_u8::<4>(bytes);
            let lo = vandq_u8(bytes, vdupq_n_u8(15));
            let first = vzip1q_u8(hi, lo);
            let last = vzip2q_u8(hi, lo);
            let indices = vaddq_u8(
                vld1q_u8([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].as_ptr()),
                vdupq_n_u8((32 - columns.len()) as u8),
            );
            let prefix = vqtbl2q_u8(uint8x16x2_t(first, last), indices);
            vst1q_u8(columns.as_mut_ptr(), prefix);
            vst1q_u8(columns.as_mut_ptr().add(columns.len() - 16), last);
        }
        return;
    }
    for column in columns.iter_mut().rev() {
        *column = (value & ((1 << BITS) - 1)) as u8;
        value >>= BITS;
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test encoding consists only of hexadecimal digits"
)]
mod tests {
    use super::fill_digits;

    #[test]
    fn digit_suffixes_match_integer_encoding() {
        let values = [0, u128::MAX, 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210]
            .into_iter()
            .chain((0..128).map(|bit| 1u128 << bit));
        for value in values {
            let hex: Vec<_> = format!("{value:032x}")
                .chars()
                .map(|digit| digit.to_digit(16).unwrap() as u8)
                .collect();
            for width in 0..=32 {
                let mut guarded = [0xff; 34];
                fill_digits::<4>(value, &mut guarded[1..=width]);
                assert_eq!(&guarded[1..=width], &hex[32 - width..]);
                assert_eq!(guarded[0], 0xff);
                assert!(guarded[1 + width..].iter().all(|&byte| byte == 0xff));
            }
            for width in 0..=16 {
                let mut guarded = [0xff; 18];
                fill_digits::<8>(value, &mut guarded[1..=width]);
                assert_eq!(&guarded[1..=width], &value.to_be_bytes()[16 - width..]);
                assert_eq!(guarded[0], 0xff);
                assert!(guarded[1 + width..].iter().all(|&byte| byte == 0xff));
            }
        }
    }
}

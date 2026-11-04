//! Architecture-neutral validation for balanced signed NTT digits.

/// Return whether every signed value lies in `[-bound, bound)`.
///
/// AArch64 uses its mandatory NEON unit internally, and x86-64 uses AVX2 when
/// the host has it. Callers do not select or observe the hardware backend, so
/// verifier-facing validation remains independent of NTT benchmarking
/// overrides.
#[must_use]
pub fn i16_values_in_balanced_range(values: &[i16], bound: i16) -> bool {
    if bound <= 0 {
        return false;
    }

    #[cfg(target_arch = "aarch64")]
    let values_valid = super::neon::i16_values_in_balanced_range(values, bound);
    #[cfg(target_arch = "x86_64")]
    let values_valid = if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was detected at runtime.
        unsafe { i16_values_in_balanced_range_avx2(values, bound) }
    } else {
        scalar_i16_values_in_balanced_range(values, bound)
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let values_valid = scalar_i16_values_in_balanced_range(values, bound);
    values_valid
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_i16_values_in_balanced_range(values: &[i16], bound: i16) -> bool {
    values.iter().all(|&value| value >= -bound && value < bound)
}

/// Check 64 values per early-exit branch; the scalar `all` loop does not
/// vectorize because of its per-element exit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i16_values_in_balanced_range_avx2(values: &[i16], bound: i16) -> bool {
    use core::arch::x86_64::{
        __m256i, _mm256_cmpgt_epi16, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi16,
        _mm256_setzero_si256, _mm256_testz_si256,
    };
    const LANES: usize = 16;
    const VECTORS_PER_BLOCK: usize = 4;

    let max_valid = _mm256_set1_epi16(bound - 1);
    let min_valid = _mm256_set1_epi16(-bound);
    let violations = |chunk: &[i16]| -> __m256i {
        // SAFETY: every chunk passed below holds exactly `LANES` values.
        let value = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
        _mm256_or_si256(
            _mm256_cmpgt_epi16(value, max_valid),
            _mm256_cmpgt_epi16(min_valid, value),
        )
    };

    let mut blocks = values.chunks_exact(LANES * VECTORS_PER_BLOCK);
    for block in &mut blocks {
        let any = block
            .chunks_exact(LANES)
            .fold(_mm256_setzero_si256(), |any, chunk| {
                _mm256_or_si256(any, violations(chunk))
            });
        if _mm256_testz_si256(any, any) == 0 {
            return false;
        }
    }
    let mut vectors = blocks.remainder().chunks_exact(LANES);
    for chunk in &mut vectors {
        let any = violations(chunk);
        if _mm256_testz_si256(any, any) == 0 {
            return false;
        }
    }
    scalar_i16_values_in_balanced_range(vectors.remainder(), bound)
}

#[cfg(test)]
mod tests {
    use super::i16_values_in_balanced_range;

    #[test]
    fn balanced_i16_range_checks_vector_and_tail() {
        let mut values = [-128i16; 19];
        values[7] = 127;
        values[18] = 0;
        assert!(i16_values_in_balanced_range(&values, 128));

        values[8] = 128;
        assert!(!i16_values_in_balanced_range(&values, 128));
        values[8] = 0;
        values[18] = -129;
        assert!(!i16_values_in_balanced_range(&values, 128));

        for bound in [1i16, 2, 128, 1024, 16384, i16::MAX] {
            for len in 0..160 {
                let values = (0..len)
                    .map(|index| {
                        let span = i32::from(bound) * 2;
                        ((index * 137 + 19) % span - i32::from(bound)) as i16
                    })
                    .collect::<Vec<_>>();
                let scalar = values.iter().all(|&value| value >= -bound && value < bound);
                assert_eq!(i16_values_in_balanced_range(&values, bound), scalar);
            }
        }
        assert!(!i16_values_in_balanced_range(&[], 0));
    }

    #[test]
    fn balanced_i16_range_rejects_each_position() {
        for len in [1usize, 15, 16, 17, 63, 64, 65, 97, 137] {
            for index in 0..len {
                for bad in [-1025i16, 1024, i16::MIN, i16::MAX] {
                    let mut values = vec![-1024i16; len];
                    values[len - 1] = 1023;
                    values[index] = bad;
                    assert!(!i16_values_in_balanced_range(&values, 1024));
                }
            }
        }
    }
}

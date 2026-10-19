//! Transposed frequency-row accumulation for the operator-norm predicate.
//!
//! Precomputes `cos_at[pos][k]`, `sin_at[pos][k]` so each nonzero updates all
//! `D/2` frequency accumulators from a contiguous row. On aarch64 the inner
//! row update uses NEON (`i64x2`).

use crate::sampler::op_norm::OpNormTable;

pub(super) const FUSED_CHUNK_FREQS: usize = 4;

/// Build `freq_cos_at[pos * half_d + k]` and `freq_sin_at[...]` from the
/// certified base tables.
pub(super) fn build_freq_at_tables(
    d: usize,
    base_cos: &[i64],
    base_sin: &[i64],
) -> (Vec<i64>, Vec<i64>) {
    let half_d = d / 2;
    let two_d = 2 * d;
    let mut freq_cos = vec![0i64; d * half_d];
    let mut freq_sin = vec![0i64; d * half_d];
    for pos in 0..d {
        let row = pos * half_d;
        for k in 0..half_d {
            let mult = (2 * k + 1) % two_d;
            let idx = (mult * pos) % two_d;
            freq_cos[row + k] = base_cos[idx];
            freq_sin[row + k] = base_sin[idx];
        }
    }
    (freq_cos, freq_sin)
}

/// Accumulate a contiguous frequency chunk into register-sized arrays.
#[inline]
pub(super) fn accumulate_transposed_chunk(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    start_k: usize,
    len: usize,
    half_d: usize,
) -> ([i64; FUSED_CHUNK_FREQS], [i64; FUSED_CHUNK_FREQS]) {
    debug_assert!(len <= FUSED_CHUNK_FREQS);
    debug_assert!(start_k + len <= half_d);

    #[cfg(target_arch = "x86_64")]
    {
        if len == FUSED_CHUNK_FREQS && std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                accumulate_transposed_chunk_avx2(table, positions, coeffs, start_k, half_d)
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if len == FUSED_CHUNK_FREQS {
            return unsafe {
                accumulate_transposed_chunk_neon(table, positions, coeffs, start_k, half_d)
            };
        }
    }

    accumulate_transposed_chunk_scalar(table, positions, coeffs, start_k, len, half_d)
}

#[inline]
pub(super) fn accumulate_transposed_chunk_scalar(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    start_k: usize,
    len: usize,
    half_d: usize,
) -> ([i64; FUSED_CHUNK_FREQS], [i64; FUSED_CHUNK_FREQS]) {
    let mut acc_re = [0i64; FUSED_CHUNK_FREQS];
    let mut acc_im = [0i64; FUSED_CHUNK_FREQS];
    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let coeff = i64::from(coeff);
        let row = table.freq_row(pos as usize, half_d);
        let (cos_row, sin_row) = row;
        for lane in 0..len {
            let k = start_k + lane;
            acc_re[lane] += coeff * cos_row[k];
            acc_im[lane] += coeff * sin_row[k];
        }
    }
    (acc_re, acc_im)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_transposed_chunk_avx2(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    start_k: usize,
    half_d: usize,
) -> ([i64; FUSED_CHUNK_FREQS], [i64; FUSED_CHUNK_FREQS]) {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_sub_epi64,
    };

    let mut acc_re = _mm256_setzero_si256();
    let mut acc_im = _mm256_setzero_si256();
    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let row = table.freq_row(pos as usize, half_d);
        let (cos_row, sin_row) = row;
        let cos_v = unsafe { _mm256_loadu_si256(cos_row.as_ptr().add(start_k).cast::<__m256i>()) };
        let sin_v = unsafe { _mm256_loadu_si256(sin_row.as_ptr().add(start_k).cast::<__m256i>()) };
        match coeff {
            1 => {
                acc_re = _mm256_add_epi64(acc_re, cos_v);
                acc_im = _mm256_add_epi64(acc_im, sin_v);
            }
            -1 => {
                acc_re = _mm256_sub_epi64(acc_re, cos_v);
                acc_im = _mm256_sub_epi64(acc_im, sin_v);
            }
            2 => {
                acc_re = _mm256_add_epi64(acc_re, _mm256_add_epi64(cos_v, cos_v));
                acc_im = _mm256_add_epi64(acc_im, _mm256_add_epi64(sin_v, sin_v));
            }
            -2 => {
                acc_re = _mm256_sub_epi64(acc_re, _mm256_add_epi64(cos_v, cos_v));
                acc_im = _mm256_sub_epi64(acc_im, _mm256_add_epi64(sin_v, sin_v));
            }
            _ => {
                return accumulate_transposed_chunk_scalar(
                    table,
                    positions,
                    coeffs,
                    start_k,
                    FUSED_CHUNK_FREQS,
                    half_d,
                );
            }
        }
    }

    let mut re = [0i64; FUSED_CHUNK_FREQS];
    let mut im = [0i64; FUSED_CHUNK_FREQS];
    unsafe {
        _mm256_storeu_si256(re.as_mut_ptr().cast::<__m256i>(), acc_re);
        _mm256_storeu_si256(im.as_mut_ptr().cast::<__m256i>(), acc_im);
    }
    (re, im)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_transposed_chunk_neon(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    start_k: usize,
    half_d: usize,
) -> ([i64; FUSED_CHUNK_FREQS], [i64; FUSED_CHUNK_FREQS]) {
    use core::arch::aarch64::{vaddq_s64, vdupq_n_s64, vld1q_s64, vst1q_s64, vsubq_s64};

    let mut re0 = vdupq_n_s64(0);
    let mut im0 = vdupq_n_s64(0);
    let mut re1 = vdupq_n_s64(0);
    let mut im1 = vdupq_n_s64(0);
    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let row = table.freq_row(pos as usize, half_d);
        let (cos_row, sin_row) = row;
        let cos0 = unsafe { vld1q_s64(cos_row.as_ptr().add(start_k)) };
        let sin0 = unsafe { vld1q_s64(sin_row.as_ptr().add(start_k)) };
        let cos1 = unsafe { vld1q_s64(cos_row.as_ptr().add(start_k + 2)) };
        let sin1 = unsafe { vld1q_s64(sin_row.as_ptr().add(start_k + 2)) };
        match coeff {
            1 => {
                re0 = vaddq_s64(re0, cos0);
                im0 = vaddq_s64(im0, sin0);
                re1 = vaddq_s64(re1, cos1);
                im1 = vaddq_s64(im1, sin1);
            }
            -1 => {
                re0 = vsubq_s64(re0, cos0);
                im0 = vsubq_s64(im0, sin0);
                re1 = vsubq_s64(re1, cos1);
                im1 = vsubq_s64(im1, sin1);
            }
            2 => {
                re0 = vaddq_s64(re0, vaddq_s64(cos0, cos0));
                im0 = vaddq_s64(im0, vaddq_s64(sin0, sin0));
                re1 = vaddq_s64(re1, vaddq_s64(cos1, cos1));
                im1 = vaddq_s64(im1, vaddq_s64(sin1, sin1));
            }
            -2 => {
                re0 = vsubq_s64(re0, vaddq_s64(cos0, cos0));
                im0 = vsubq_s64(im0, vaddq_s64(sin0, sin0));
                re1 = vsubq_s64(re1, vaddq_s64(cos1, cos1));
                im1 = vsubq_s64(im1, vaddq_s64(sin1, sin1));
            }
            _ => {
                return accumulate_transposed_chunk_scalar(
                    table,
                    positions,
                    coeffs,
                    start_k,
                    FUSED_CHUNK_FREQS,
                    half_d,
                );
            }
        }
    }

    let mut re = [0i64; FUSED_CHUNK_FREQS];
    let mut im = [0i64; FUSED_CHUNK_FREQS];
    unsafe {
        vst1q_s64(re.as_mut_ptr(), re0);
        vst1q_s64(im.as_mut_ptr(), im0);
        vst1q_s64(re.as_mut_ptr().add(2), re1);
        vst1q_s64(im.as_mut_ptr().add(2), im1);
    }
    (re, im)
}

#[inline]
#[cfg(test)]
pub(super) fn accumulate_transposed_scalar(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    acc_re: &mut [i64],
    acc_im: &mut [i64],
) {
    let half_d = acc_re.len();
    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let coeff = i64::from(coeff);
        let row = table.freq_row(pos as usize, half_d);
        let (cos_row, sin_row) = row;
        for k in 0..half_d {
            acc_re[k] += coeff * cos_row[k];
            acc_im[k] += coeff * sin_row[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::op_norm::OpNormTable;

    fn table(d: usize) -> OpNormTable {
        OpNormTable::new(d, 48, (2 * d) as u64, 64).unwrap()
    }

    #[test]
    fn transposed_scalar_matches_legacy_accumulation() {
        let d = 64;
        let t = table(d);
        let half_d = d / 2;
        let mut state = 0xC0FFEE_u64;
        for _ in 0..500 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut positions = Vec::new();
            let mut coeffs = Vec::new();
            let weight = 8 + (state as usize % 20);
            for i in 0..weight {
                let shift = (i as u32).min(63);
                positions.push((state.wrapping_shr(shift * 3) as u32) % d as u32);
                coeffs.push(if state.wrapping_shr(shift) & 1 == 0 {
                    1
                } else {
                    -2
                });
            }
            let mut legacy_re = vec![0i64; half_d];
            let mut legacy_im = vec![0i64; half_d];
            let two_d = 2 * d;
            for k in 0..half_d {
                let mult = (2 * k + 1) % two_d;
                for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
                    let idx = (mult * pos as usize) % two_d;
                    let coeff = i64::from(coeff);
                    legacy_re[k] += coeff * t.base_cos_i64(idx);
                    legacy_im[k] += coeff * t.base_sin_i64(idx);
                }
            }
            let mut trans_re = vec![0i64; half_d];
            let mut trans_im = vec![0i64; half_d];
            accumulate_transposed_scalar(&t, &positions, &coeffs, &mut trans_re, &mut trans_im);
            assert_eq!(legacy_re, trans_re);
            assert_eq!(legacy_im, trans_im);
            for start_k in (0..half_d).step_by(FUSED_CHUNK_FREQS) {
                let len = (half_d - start_k).min(FUSED_CHUNK_FREQS);
                let (chunk_re, chunk_im) =
                    accumulate_transposed_chunk(&t, &positions, &coeffs, start_k, len, half_d);
                assert_eq!(&legacy_re[start_k..start_k + len], &chunk_re[..len]);
                assert_eq!(&legacy_im[start_k..start_k + len], &chunk_im[..len]);
            }
        }
    }

    #[test]
    fn transposed_chunk_handles_generic_coefficients() {
        let d = 64;
        let t = table(d);
        let half_d = d / 2;
        let positions = [0, 7, 19, 42, 63];
        let coeffs = [3, -4, 1, -2, 5];
        let mut legacy_re = vec![0i64; half_d];
        let mut legacy_im = vec![0i64; half_d];
        let two_d = 2 * d;
        for k in 0..half_d {
            let mult = (2 * k + 1) % two_d;
            for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
                let idx = (mult * pos as usize) % two_d;
                let coeff = i64::from(coeff);
                legacy_re[k] += coeff * t.base_cos_i64(idx);
                legacy_im[k] += coeff * t.base_sin_i64(idx);
            }
        }
        for start_k in (0..half_d).step_by(FUSED_CHUNK_FREQS) {
            let len = (half_d - start_k).min(FUSED_CHUNK_FREQS);
            let (chunk_re, chunk_im) =
                accumulate_transposed_chunk(&t, &positions, &coeffs, start_k, len, half_d);
            assert_eq!(&legacy_re[start_k..start_k + len], &chunk_re[..len]);
            assert_eq!(&legacy_im[start_k..start_k + len], &chunk_im[..len]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn transposed_chunk_avx2_matches_legacy_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let d = 64;
        let t = table(d);
        let half_d = d / 2;
        let positions = [1, 5, 9, 17, 23, 31, 47, 58];
        let coeffs = [1, -1, 2, -2, 1, 2, -1, -2];
        for start_k in (0..half_d).step_by(FUSED_CHUNK_FREQS) {
            let scalar = accumulate_transposed_chunk_scalar(
                &t,
                &positions,
                &coeffs,
                start_k,
                FUSED_CHUNK_FREQS,
                half_d,
            );
            let avx2 = unsafe {
                accumulate_transposed_chunk_avx2(&t, &positions, &coeffs, start_k, half_d)
            };
            assert_eq!(scalar, avx2);
        }
    }
}

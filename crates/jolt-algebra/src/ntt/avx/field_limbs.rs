//! AVX2 reduction of canonical field coefficients into i32 CRT residues.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::ntt::field_limbs::{FieldLimbScales, FieldModulusLimbs};
use crate::ntt::prime::MontCoeff;

/// AVX2 form of [`crate::ntt::field_limbs::field_residues`] for i32 primes.
///
/// Each block of eight coefficients is transposed into 32-bit words, compared
/// against `⌊q/2⌋`, and cut into signed 26-bit limbs that stay in registers
/// while every prime accumulates its even and odd lanes with `vpmuldq` and
/// applies one signed Montgomery reduction.
///
/// # Safety
///
/// The caller must ensure AVX2 is available, `canonical.len()` is a multiple
/// of eight, `start + canonical.len() <= D`, `L` covers `modulus`, and every
/// coefficient is below that modulus.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn field_residues_i32<const L: usize, const K: usize, const D: usize>(
    out: &mut [[MontCoeff<i32>; D]; K],
    start: usize,
    canonical: &[u128],
    modulus: &FieldModulusLimbs,
    scales: &[FieldLimbScales; K],
) {
    let sign = _mm256_set1_epi32(i32::MIN);
    let mask = _mm256_set1_epi32((1 << 26) - 1);
    let half = [
        modulus.half as u32,
        (modulus.half >> 32) as u32,
        (modulus.half >> 64) as u32,
        (modulus.half >> 96) as u32,
    ]
    .map(|word| _mm256_set1_epi32(word as i32));
    let half_flipped = half.map(|word| _mm256_xor_si256(word, sign));
    let q = modulus.limbs.map(|limb| _mm256_set1_epi32(limb));
    let src = canonical.as_ptr().cast::<__m128i>();
    for block in (0..canonical.len()).step_by(8) {
        // SAFETY: the length is a multiple of eight, so coefficients
        // `block..block + 8` are in bounds; the loads are unaligned.
        let [w0, w1, w2, w3] = unsafe {
            let pair = |low: usize| {
                _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(src.add(block + low))),
                    _mm_loadu_si128(src.add(block + low + 4)),
                )
            };
            let (r0, r1, r2, r3) = (pair(0), pair(1), pair(2), pair(3));
            let a = _mm256_unpacklo_epi32(r0, r1);
            let b = _mm256_unpackhi_epi32(r0, r1);
            let c = _mm256_unpacklo_epi32(r2, r3);
            let d = _mm256_unpackhi_epi32(r2, r3);
            [
                _mm256_unpacklo_epi64(a, c),
                _mm256_unpackhi_epi64(a, c),
                _mm256_unpacklo_epi64(b, d),
                _mm256_unpackhi_epi64(b, d),
            ]
        };
        // Unsigned lexicographic `c > ⌊q/2⌋`: an equal word defers to the
        // words below it. Coefficients lie below `2^(26 L)`, so word `j >= L`
        // is zero on both sides.
        let gt = |word: __m256i, index: usize| {
            _mm256_cmpgt_epi32(_mm256_xor_si256(word, sign), half_flipped[index])
        };
        let defer = |center: __m256i, word: __m256i, index: usize| {
            _mm256_or_si256(
                gt(word, index),
                _mm256_and_si256(_mm256_cmpeq_epi32(word, half[index]), center),
            )
        };
        let mut center = gt(w0, 0);
        if L > 1 {
            center = defer(center, w1, 1);
        }
        if L > 2 {
            center = defer(center, w2, 2);
        }
        if L > 3 {
            center = defer(center, w3, 3);
        }

        let slices = [
            _mm256_and_si256(w0, mask),
            _mm256_and_si256(
                _mm256_or_si256(_mm256_srli_epi32::<26>(w0), _mm256_slli_epi32::<6>(w1)),
                mask,
            ),
            _mm256_and_si256(
                _mm256_or_si256(_mm256_srli_epi32::<20>(w1), _mm256_slli_epi32::<12>(w2)),
                mask,
            ),
            _mm256_and_si256(
                _mm256_or_si256(_mm256_srli_epi32::<14>(w2), _mm256_slli_epi32::<18>(w3)),
                mask,
            ),
            _mm256_srli_epi32::<8>(w3),
        ];
        let even: [__m256i; L] =
            std::array::from_fn(|j| _mm256_sub_epi32(slices[j], _mm256_and_si256(q[j], center)));
        // Move each odd lane into the low half that `vpmuldq` reads; a 64-bit
        // shift would be folded into an arithmetic shift AVX2 lacks.
        let odd = even.map(|limb| _mm256_shuffle_epi32::<0b1111_0101>(limb));

        for (residues, scales) in out.iter_mut().zip(scales) {
            let mut acc_even = _mm256_setzero_si256();
            let mut acc_odd = _mm256_setzero_si256();
            for j in 0..L {
                let scale = _mm256_set1_epi32(scales.scales[j]);
                acc_even = _mm256_add_epi64(acc_even, _mm256_mul_epi32(even[j], scale));
                acc_odd = _mm256_add_epi64(acc_odd, _mm256_mul_epi32(odd[j], scale));
            }
            let p = _mm256_set1_epi32(scales.p);
            let pinv = _mm256_set1_epi32(scales.pinv);
            let redc = |acc: __m256i| {
                // SAFETY: AVX2 is required by this function, and each signed
                // accumulator satisfies the shared reducer's input bound.
                unsafe { super::montgomery::mont_reduce_i32_products_avx2(acc, p, pinv) }
            };
            let reduced = _mm256_blend_epi32::<0b1010_1010>(
                redc(acc_even),
                _mm256_slli_epi64::<32>(redc(acc_odd)),
            );
            // SAFETY: `start + block + 8 <= D`, and MontCoeff<i32> is
            // transparent.
            unsafe {
                _mm256_storeu_si256(
                    residues.as_mut_ptr().add(start + block).cast::<__m256i>(),
                    reduced,
                );
            }
        }
    }
}

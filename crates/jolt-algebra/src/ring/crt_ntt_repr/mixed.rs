use std::array::from_fn;

use crate::ntt::crt::modular_inverse;
use crate::{CanonicalEncoding, CyclotomicRing, Field};
use akita_error::AkitaError;

use super::{CrtNttParamSet, CyclotomicCrtNtt};

/// CRT parameters with an i32 prefix and one i16 exactness tail prime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct I16TailParams<const K: usize, const D: usize> {
    /// Existing 30-bit CRT profile.
    pub wide: CrtNttParamSet<i32, K, D>,
    /// One 14-bit tail prime, materialized only for schedules that require it.
    pub tail: CrtNttParamSet<i16, 1, D>,
    tail_residue_weight: i64,
    tail_digit_weights: [i64; K],
}

impl<const K: usize, const D: usize> I16TailParams<K, D> {
    /// Extend an existing i32 profile by one coprime i16 prime.
    #[must_use]
    pub fn new(wide: CrtNttParamSet<i32, K, D>, tail: CrtNttParamSet<i16, 1, D>) -> Self {
        let tail_modulus = i64::from(tail.primes[0].p);
        let tail_gamma: [i64; K] = from_fn(|i| {
            modular_inverse(
                i64::from(wide.primes[i].p).rem_euclid(tail_modulus) as u64,
                tail_modulus as u64,
            )
            .expect("CRT tail prime must be coprime to the base profile") as i64
        });
        // The final mixed-radix digit is affine in the tail residue and the
        // already reconstructed wide digits. Precompute that linear form so
        // reconstruction needs one reduction, not K dependent reductions.
        let mut tail_residue_weight = 1;
        let mut tail_digit_weights = [0; K];
        for i in (0..K).rev() {
            tail_residue_weight = (tail_residue_weight * tail_gamma[i]) % tail_modulus;
            tail_digit_weights[i] = (-tail_residue_weight).rem_euclid(tail_modulus);
        }
        Self {
            wide,
            tail,
            tail_residue_weight,
            tail_digit_weights,
        }
    }
}

/// Multiply a homogeneous i32 prepared matrix plus a homogeneous i16 tail by
/// one signed-i16 ring vector.
///
/// Keeping the two residue widths in separate slices avoids duplicating the
/// base matrix when an exactness tail is added lazily. Shape relationships are
/// checked before indexing so verifier callers reject malformed prepared state.
pub fn mat_vec_i16_with_tail<F: Field + CanonicalEncoding, const K: usize, const D: usize>(
    wide_matrix: &[CyclotomicCrtNtt<i32, K, D>],
    tail_matrix: &[CyclotomicCrtNtt<i16, 1, D>],
    num_rows: usize,
    num_cols: usize,
    rhs: &[[i16; D]],
    params: &I16TailParams<K, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    let wide_accumulators =
        CyclotomicCrtNtt::mat_vec_i16_ntt(wide_matrix, num_rows, num_cols, rhs, &params.wide)?;
    let tail_accumulators =
        CyclotomicCrtNtt::mat_vec_i16_ntt(tail_matrix, num_rows, num_cols, rhs, &params.tail)?;
    Ok(wide_accumulators
        .iter()
        .zip(&tail_accumulators)
        .map(|(wide, tail)| ntt_with_i16_tail_to_ring(wide, tail, params))
        .collect())
}

/// Reconstruct one negacyclic NTT accumulator from a wide CRT prefix and its
/// exactness-only i16 tail.
pub fn ntt_with_i16_tail_to_ring<F: Field + CanonicalEncoding, const K: usize, const D: usize>(
    wide_ntt: &CyclotomicCrtNtt<i32, K, D>,
    tail_ntt: &CyclotomicCrtNtt<i16, 1, D>,
    params: &I16TailParams<K, D>,
) -> CyclotomicRing<F, D> {
    let wide = wide_ntt.centered_coefficients_with_params(&params.wide);
    let tail = tail_ntt.centered_coefficients_with_params(&params.tail);
    mixed_coefficients_to_ring(&wide, &tail, params)
}

/// Reconstruct one cyclic NTT accumulator from a wide CRT prefix and its
/// exactness-only i16 tail.
pub fn cyclic_ntt_with_i16_tail_to_ring<
    F: Field + CanonicalEncoding,
    const K: usize,
    const D: usize,
>(
    wide_ntt: &CyclotomicCrtNtt<i32, K, D>,
    tail_ntt: &CyclotomicCrtNtt<i16, 1, D>,
    params: &I16TailParams<K, D>,
) -> CyclotomicRing<F, D> {
    let wide = wide_ntt.centered_cyclic_coefficients_with_params(&params.wide);
    let tail = tail_ntt.centered_cyclic_coefficients_with_params(&params.tail);
    mixed_coefficients_to_ring(&wide, &tail, params)
}

fn mixed_coefficients_to_ring<F: Field + CanonicalEncoding, const K: usize, const D: usize>(
    wide: &[[i32; D]; K],
    tail: &[[i16; D]; 1],
    params: &I16TailParams<K, D>,
) -> CyclotomicRing<F, D> {
    let tail_modulus = i64::from(params.tail.primes[0].p);
    let garner = &params.wide.garner;
    let (field_weights, tail_field_weight) = garner.field_weights::<F>();

    let mut wide_digits = wide.map(|limb| limb.map(i64::from));
    garner.centered_mixed_radix(&mut wide_digits);

    let coefficients = from_fn(|d| {
        let digits = from_fn(|limb| wide_digits[limb][d]);

        // Centered i32 digits are below 2^30 and tail weights below 2^14, so
        // each product is below 2^44 and the short linear form fits in i64.
        let tail_digit = i64::from(tail[0][d]) * params.tail_residue_weight
            + digits
                .iter()
                .zip(params.tail_digit_weights)
                .map(|(digit, weight)| digit * weight)
                .sum::<i64>();
        let mut tail_digit = tail_digit.rem_euclid(tail_modulus);
        if tail_digit > tail_modulus / 2 {
            tail_digit -= tail_modulus;
        }

        garner.digits_to_field(&digits, &field_weights)
            + F::from_i64(tail_digit) * tail_field_weight
    });
    CyclotomicRing::from_coefficients(coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::prime::I32_LAZY_DOT_BATCH;
    use crate::ntt::tables::{
        q128_primes, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q32_PRIMES, Q64_NUM_PRIMES,
        Q64_PRIMES,
    };
    use jolt_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

    fn assert_split_i16_matvec<F, const K: usize, const D: usize>(
        wide_params: CrtNttParamSet<i32, K, D>,
    ) where
        F: Field + CanonicalEncoding,
    {
        let tail_params = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
        let params = I16TailParams::new(wide_params.clone(), tail_params.clone());
        const NUM_ROWS: usize = 2;
        const NUM_COLS: usize = I32_LAZY_DOT_BATCH + 1;
        let matrix = (0..NUM_ROWS * NUM_COLS)
            .map(|entry| {
                CyclotomicRing::<F, D>::from_coefficients(from_fn(|coefficient| {
                    F::from_i64(((entry * 17 + coefficient * 5) % 31) as i64 - 15)
                }))
            })
            .collect::<Vec<_>>();
        let wide = matrix
            .iter()
            .map(|ring| CyclotomicCrtNtt::from_ring(ring, &wide_params))
            .collect::<Vec<_>>();
        let tail = matrix
            .iter()
            .map(|ring| CyclotomicCrtNtt::from_ring(ring, &tail_params))
            .collect::<Vec<_>>();
        let rhs = (0..NUM_COLS)
            .map(|column| {
                from_fn(|coefficient| match (column + coefficient) % 6 {
                    0 => i16::MIN,
                    1 => -1024,
                    2 => -1,
                    3 => 0,
                    4 => 1023,
                    _ => i16::MAX,
                })
            })
            .collect::<Vec<_>>();
        let expected = matrix
            .chunks_exact(NUM_COLS)
            .map(|row| {
                row.iter()
                    .zip(&rhs)
                    .fold(CyclotomicRing::<F, D>::zero(), |sum, (lhs, digits)| {
                        let rhs = CyclotomicRing::from_coefficients(
                            digits.map(|digit| F::from_i64(i64::from(digit))),
                        );
                        sum + *lhs * rhs
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            mat_vec_i16_with_tail::<F, K, D>(&wide, &tail, NUM_ROWS, NUM_COLS, &rhs, &params)
                .expect("split i16 matvec"),
            expected
        );
    }

    #[test]
    fn split_i16_matvec_matches_all_fields_and_multiple_ring_dimensions() {
        assert_split_i16_matvec::<Prime32Offset99, Q32_NUM_PRIMES, 64>(CrtNttParamSet::new(
            Q32_PRIMES,
        ));
        assert_split_i16_matvec::<Prime32Offset99, Q32_NUM_PRIMES, 128>(CrtNttParamSet::new(
            Q32_PRIMES,
        ));
        assert_split_i16_matvec::<Prime64Offset59, Q64_NUM_PRIMES, 64>(CrtNttParamSet::new(
            Q64_PRIMES,
        ));
        assert_split_i16_matvec::<Prime64Offset59, Q64_NUM_PRIMES, 128>(CrtNttParamSet::new(
            Q64_PRIMES,
        ));
        assert_split_i16_matvec::<Prime128OffsetA7F7, Q128_NUM_PRIMES, 64>(CrtNttParamSet::new(
            q128_primes(),
        ));
        assert_split_i16_matvec::<Prime128OffsetA7F7, Q128_NUM_PRIMES, 128>(CrtNttParamSet::new(
            q128_primes(),
        ));
    }

    #[test]
    fn split_i16_matvec_rejects_malformed_shapes() {
        const D: usize = 64;
        let wide_params = CrtNttParamSet::<i32, Q64_NUM_PRIMES, D>::new(Q64_PRIMES);
        let tail_params = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
        let params = I16TailParams::new(wide_params, tail_params);
        let wide = vec![CyclotomicCrtNtt::zero()];
        let tail = vec![CyclotomicCrtNtt::zero()];
        let rhs = vec![[1i16; D]; 2];
        assert!(matches!(
            mat_vec_i16_with_tail::<Prime64Offset59, Q64_NUM_PRIMES, D>(
                &wide, &tail, 2, 2, &rhs, &params
            ),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert!(matches!(
            mat_vec_i16_with_tail::<Prime64Offset59, Q64_NUM_PRIMES, D>(
                &wide, &tail, 1, 1, &rhs, &params
            ),
            Err(AkitaError::InvalidSize {
                expected: 1,
                actual: 2
            })
        ));
    }
}

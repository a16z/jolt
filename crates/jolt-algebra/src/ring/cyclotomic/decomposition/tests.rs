use super::*;
use jolt_field::{
    Fp64, Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime40Offset195, Prime64Offset59,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn modulus<F: Field + CanonicalEncoding>() -> u128 {
    (-F::one())
        .to_u128_checked()
        .expect("test fields fit in u128")
        + 1
}

/// Independent oracle for balanced base-`2^log_basis` digits.
///
/// Runs the digit recurrence with the carry written as
/// `floor(c / b) + [raw >= b/2]`, which cannot overflow. The centered value
/// is held as its low 128 bits plus a sign, so the first quotient is exact
/// even when `q - threshold > i128::MAX`. Recomposition is checked modulo
/// `2^128`, and digits covering the field width must leave no final carry:
/// they recompose the centered value exactly, so recomposition is `c mod q`.
fn exact_balanced_digits(
    canonical: u128,
    q: u128,
    threshold: u128,
    levels: usize,
    log_basis: u32,
) -> Vec<i32> {
    let b = 1i128 << log_basis;
    let half_b = b >> 1;
    let mask = (1u128 << log_basis) - 1;
    let negative = canonical > threshold;
    let centered_low = if negative {
        canonical.wrapping_sub(q)
    } else {
        canonical
    };
    let sign_fill = if negative {
        !(u128::MAX >> log_basis)
    } else {
        0
    };
    let mut raw = (centered_low & mask) as i128;
    let mut floor = ((centered_low >> log_basis) | sign_fill) as i128;
    let mut carry = 0i128;
    let mut recomposed = 0u128;
    let mut digits = Vec::with_capacity(levels);
    for level in 0..levels {
        let high = raw >= half_b;
        let digit = if high { raw - b } else { raw };
        digits.push(digit as i32);
        recomposed = recomposed.wrapping_add(
            (digit as u128)
                .checked_shl(level as u32 * log_basis)
                .unwrap_or(0),
        );
        carry = floor + i128::from(high);
        raw = carry & (b - 1);
        floor = carry >> log_basis;
    }
    let tail = (carry as u128)
        .checked_shl(levels as u32 * log_basis)
        .unwrap_or(0);
    assert_eq!(recomposed.wrapping_add(tail), centered_low);
    let field_bits = 128 - (q - 1).leading_zeros();
    if levels as u32 * log_basis >= field_bits {
        assert_eq!(
            carry, 0,
            "digits covering the field width recompose c mod q"
        );
    }
    digits
}

fn edge_values(
    q: u128,
    threshold: u128,
    log_basis: u32,
    levels: usize,
    rng: &mut StdRng,
) -> Vec<u128> {
    let mut values = vec![0, 1, 2, q - 1, q - 2, q / 2 - 1, q / 2, q / 2 + 1];
    for delta in 0..3 {
        values.push(threshold.wrapping_sub(delta));
        values.push(threshold.wrapping_add(delta));
    }
    let field_bits = 128 - (q - 1).leading_zeros();
    for bit in 0..field_bits {
        let power = 1u128 << bit;
        values.extend([power - 1, power, power + 1]);
        values.extend([q - power - 1, q - power, q - power + 1]);
    }
    // Carry-propagation boundaries: partial sums of `H` and their negations.
    let half_b = 1u128 << (log_basis - 1);
    let mut partial = 0u128;
    for level in 0..levels as u32 {
        let shift = level * log_basis;
        if shift >= field_bits {
            break;
        }
        partial = partial.wrapping_add(half_b << shift);
        for delta in 0..3 {
            values.push(partial.wrapping_add(delta));
            values.push(partial.wrapping_sub(delta));
            values.push(q.wrapping_sub(partial).wrapping_add(delta));
            values.push(q.wrapping_sub(partial).wrapping_sub(delta));
        }
    }
    for _ in 0..64 {
        values.push(rng.gen_range(0..q));
    }
    values.retain(|&value| value < q);
    values
}

/// Run every compiled kernel instantiation over `coefficients`, twice with
/// different prefills so an unwritten digit cannot pass by accident.
fn check_signed_kernels<F, T>(
    coefficients: &[F],
    params: &BalancedDecomposePow2Params<F>,
    expected: &[Vec<i32>],
    context: &str,
) where
    F: Field + CanonicalEncoding,
    T: BalancedSignedDigit + From<i8> + Into<i32>,
{
    let width = coefficients.len();
    let check = |label: &str, run: &dyn Fn(&mut [T])| {
        for fill in [0i8, 1] {
            let mut out = vec![T::from(fill); width * params.levels];
            run(&mut out);
            for (coefficient, digits) in expected.iter().enumerate() {
                for (level, &digit) in digits.iter().enumerate() {
                    assert_eq!(
                        out[level * width + coefficient].into(),
                        digit,
                        "{label} {context} width={width} coefficient={coefficient} level={level}"
                    );
                }
            }
        }
    };
    check("portable", &|out| {
        balanced_decompose_coefficients_pow2_signed_kernel(coefficients, out, params)
    });
    check("dispatch", &|out| {
        balanced_decompose_coefficients_pow2_signed_into(coefficients, out, params)
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection guarantees AVX2.
            check("avx2", &|out| unsafe {
                x86::balanced_decompose_coefficients_pow2_signed_avx2(coefficients, out, params)
            });
        }
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512dq")
        {
            // SAFETY: runtime feature detection guarantees every enabled feature.
            check("avx512", &|out| unsafe {
                x86::balanced_decompose_coefficients_pow2_signed_avx512(coefficients, out, params)
            });
        }
    }
}

fn check_field_against_exact_recurrence<F: Field + CanonicalEncoding>(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let q = modulus::<F>();
    let field_bits = (128 - (q - 1).leading_zeros()) as usize;
    for log_basis in 1..=16u32 {
        let covering = field_bits.div_ceil(log_basis as usize);
        let max_levels = ((128 + log_basis) / log_basis) as usize;
        // `word_levels` fills the first 64-bit word of `x + H`, the limit of the
        // u32 SIMD kernels and of the one-word portable instantiation.
        let word_levels = 64 / log_basis as usize;
        let mut level_counts = vec![
            1,
            2,
            covering - 1,
            covering,
            covering + 1,
            word_levels,
            word_levels + 1,
            max_levels,
        ];
        // `new` rejects base-2 digits past the field width.
        level_counts.retain(|&levels| {
            (1..=max_levels).contains(&levels) && (log_basis > 1 || levels <= field_bits)
        });
        level_counts.sort_unstable();
        level_counts.dedup();
        for levels in level_counts {
            let params = BalancedDecomposePow2Params::new(levels, log_basis);
            let values = edge_values(q, params.threshold, log_basis, levels, &mut rng);
            let coefficients = values
                .iter()
                .map(|&value| F::from_u128_reduced(value))
                .collect::<Vec<_>>();
            let expected = values
                .iter()
                .map(|&value| exact_balanced_digits(value, q, params.threshold, levels, log_basis))
                .collect::<Vec<_>>();
            if log_basis <= 8 {
                // The public u32 SIMD path requires an aligned width. Pad the
                // complete corpus so carry boundaries and random values reach it.
                let aligned_width = values.len().next_multiple_of(8);
                let mut aligned = coefficients.clone();
                aligned.resize(aligned_width, F::zero());
                let mut out = vec![0i8; aligned_width * levels];
                balanced_decompose_coefficients_pow2_i8_into(&aligned, &mut out, &params);
                for (coefficient, digits) in expected.iter().enumerate() {
                    for (level, &digit) in digits.iter().enumerate() {
                        assert_eq!(
                            i32::from(out[level * aligned_width + coefficient]),
                            digit,
                            "aligned i8 corpus q={q:#x} log_basis={log_basis} levels={levels} coefficient={coefficient} level={level}"
                        );
                    }
                }
            }
            for width in [values.len(), 1, 8, 40, 63, 64, 65, 72, 129] {
                let width = width.min(values.len());
                let context = format!("q={q:#x} log_basis={log_basis} levels={levels}");
                check_signed_kernels::<F, i16>(
                    &coefficients[..width],
                    &params,
                    &expected[..width],
                    &context,
                );
                if log_basis > 8 {
                    continue;
                }
                check_signed_kernels::<F, i8>(
                    &coefficients[..width],
                    &params,
                    &expected[..width],
                    &context,
                );
                // Public i8 entry point, including its u32 SIMD path.
                let mut out = vec![0i8; width * levels];
                balanced_decompose_coefficients_pow2_i8_into(
                    &coefficients[..width],
                    &mut out,
                    &params,
                );
                for (coefficient, digits) in expected[..width].iter().enumerate() {
                    for (level, &digit) in digits.iter().enumerate() {
                        assert_eq!(
                            i32::from(out[level * width + coefficient]),
                            digit,
                            "i8 entry {context} width={width} coefficient={coefficient} level={level}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn parameters_derive_the_field_modulus() {
    let params = BalancedDecomposePow2Params::<Prime32Offset99>::new(4, 8);
    assert_eq!(params.q, modulus::<Prime32Offset99>());
}

fn check_small_signed_decompose<F: Field + CanonicalEncoding>() {
    let q = modulus::<F>();
    let field_bits = 128 - (q - 1).leading_zeros();
    for log_basis in 1..=8 {
        let half_basis = 1i16 << (log_basis - 1);
        let ring = [
            i16::MIN,
            i16::MIN + 1,
            -257,
            -256,
            -129,
            -128,
            -half_basis - 1,
            -half_basis,
            -half_basis + 1,
            -2,
            -1,
            0,
            1,
            2,
            half_basis - 1,
            half_basis,
            half_basis + 1,
            127,
            128,
            255,
            256,
            i16::MAX - 1,
            i16::MAX,
        ];
        let i8_ring = [
            i8::MIN,
            -127,
            -65,
            -64,
            -2,
            -1,
            0,
            1,
            2,
            63,
            64,
            126,
            i8::MAX,
        ];
        let field_ring = CyclotomicRing::from_coefficients(ring.map(|v| F::from_i64(i64::from(v))));
        let i8_field_ring =
            CyclotomicRing::from_coefficients(i8_ring.map(|v| F::from_i64(i64::from(v))));
        let max_levels = if log_basis == 1 {
            field_bits
        } else {
            (128 + log_basis) / log_basis
        } as usize;
        for levels in 0..=max_levels {
            let params = BalancedDecomposePow2Params::<F>::new(levels, log_basis);
            if log_basis == 1 && levels == field_bits as usize {
                assert_eq!(params.threshold, 0);
            }
            let mut expected_i8 = vec![[0i8; 23]; levels];
            field_ring.balanced_decompose_pow2_i8_into_with_params(&mut expected_i8, &params);
            let mut expected_i16 = vec![[0i16; 23]; levels];
            field_ring.balanced_decompose_pow2_i16_into(&mut expected_i16, &params);
            let mut expected_i8_carrier = vec![[0i8; 13]; levels];
            i8_field_ring
                .balanced_decompose_pow2_i8_into_with_params(&mut expected_i8_carrier, &params);
            for fill in [0, 1] {
                let mut actual_i8 = vec![[fill as i8; 23]; levels];
                params.decompose_small_signed_into(&ring, &mut actual_i8);
                assert_eq!(
                    actual_i8, expected_i8,
                    "i8: basis={log_basis} levels={levels}"
                );
                let mut actual_i16 = vec![[fill as i16; 23]; levels];
                params.decompose_small_signed_into(&ring, &mut actual_i16);
                assert_eq!(
                    actual_i16, expected_i16,
                    "i16: basis={log_basis} levels={levels}"
                );
                let mut actual_i8_carrier = vec![[fill as i8; 13]; levels];
                params.decompose_small_signed_into(&i8_ring, &mut actual_i8_carrier);
                assert_eq!(actual_i8_carrier, expected_i8_carrier);
            }
        }
    }
}

#[test]
fn decompose_small_signed_matches_field_all_accepted_shapes() {
    check_small_signed_decompose::<Prime32Offset99>();
    check_small_signed_decompose::<Prime40Offset195>();
    check_small_signed_decompose::<Prime64Offset59>();
    check_small_signed_decompose::<Prime128Offset275>();
    check_small_signed_decompose::<Prime128OffsetA7F7>();
}

#[test]
fn add_bias_decomposition_matches_exact_recurrence_fp128() {
    check_field_against_exact_recurrence::<Prime128OffsetA7F7>(0xa7f7);
    check_field_against_exact_recurrence::<Prime128Offset275>(0x275);
}

#[test]
fn add_bias_decomposition_matches_exact_recurrence_narrow_fields() {
    check_field_against_exact_recurrence::<Prime64Offset59>(0x59);
    check_field_against_exact_recurrence::<Fp64<4294967197>>(0x4294967197);
    check_field_against_exact_recurrence::<Prime40Offset195>(0x195);
    check_field_against_exact_recurrence::<Prime32Offset99>(0x99);
}

#[test]
fn fp128_near_half_modulus_does_not_wrap_the_carry() {
    // With q = 2^128 - 275 and log_basis = 9, centered values within b/2 of
    // 2^127 overflowed the former i128 carry `(c - digit) >> log_basis`, so
    // the digits recomposed to `x - 2^128` once `levels * log_basis > 128`.
    type F = Prime128Offset275;
    let q = modulus::<F>();
    let (levels, log_basis) = (15, 9);
    let params = BalancedDecomposePow2Params::new(levels, log_basis);
    let value = F::from_u128_reduced(q / 2 - 1);
    let mut digits = [0i16; 15];
    balanced_decompose_coefficients_pow2_signed_into(&[value], &mut digits, &params);

    let basis = F::from_u64(1 << log_basis);
    let mut recomposed = F::zero();
    let mut power = F::one();
    for digit in digits {
        recomposed += F::from_i64(i64::from(digit)) * power;
        power *= basis;
    }
    assert_eq!(recomposed, value);
}

/// With `q = 2^32 - 99`, 33 base-2 digits of 1 were all -1, which recomposes
/// to `-(2^33 - 1)` rather than 1.
#[test]
fn log_basis_one_accepts_levels_up_to_the_field_width() {
    type F = Prime32Offset99;
    let params = BalancedDecomposePow2Params::new(32, 1);
    let one = [F::one(); 8];
    let mut digits = [0i8; 8 * 32];
    balanced_decompose_coefficients_pow2_i8_into(&one, &mut digits, &params);
    let mut recomposed = F::zero();
    let mut power = F::one();
    for level in 0..32 {
        recomposed += F::from_i64(i64::from(digits[level * 8])) * power;
        power += power;
    }
    assert_eq!(recomposed, F::one());
}

#[test]
#[should_panic(expected = "log_basis 1 needs levels <= the field width")]
fn log_basis_one_rejects_levels_past_the_field_width() {
    BalancedDecomposePow2Params::<Prime32Offset99>::new(33, 1);
}

/// `2^32 + 1` levels used to truncate to 1 in the digit-budget guard.
#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic(expected = "levels must fit in u32")]
fn levels_beyond_u32_are_rejected() {
    BalancedDecomposePow2Params::<Prime32Offset99>::new((1 << 32) + 1, 16);
}

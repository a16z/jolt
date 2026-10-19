use std::arch::aarch64::*;

use super::i32_kernels::centered_reduce_4x_i32;
use super::{
    forward_ntt_centered_i16_i32, forward_ntt_cyclic_i32, forward_ntt_cyclic_i8_i32,
    forward_ntt_i32, forward_ntt_i8_i32, inverse_ntt_cyclic_i32, inverse_ntt_i32,
};
use crate::ntt::butterfly::{forward_ntt, forward_ntt_cyclic, inverse_ntt, inverse_ntt_cyclic};
use crate::ntt::tables::{Q128_RAW_PRIMES, Q64_PRIMES};
use crate::ntt::{MontCoeff, NttKernelPlan, NttPrime, NttTwiddles};

#[test]
fn centered_reduction_range_residue_and_rounding_boundaries() {
    let moduli = [3, 5, 17, 257, 12289, (1 << 29) - 1, (1 << 30) - 1];
    for p in moduli
        .into_iter()
        .chain(Q128_RAW_PRIMES)
        .chain(Q64_PRIMES.map(|p| p.p))
    {
        let p64 = i64::from(p);
        let reciprocal = ((1i64 << 31) + p64 / 2) / p64;
        let mut inputs = vec![
            -2 * p64 + 1,
            -p64 - 1,
            -p64,
            -p64 + 1,
            -1,
            0,
            1,
            p64 - 1,
            p64,
            p64 + 1,
            2 * p64 - 1,
        ];
        // Include neighbors of every possible sqrdmulh rounding boundary.
        for quotient in -2..=2 {
            let threshold = ((2 * quotient + 1) * (1i64 << 30)) / reciprocal;
            inputs.extend(
                (-2..=2)
                    .map(|delta| threshold + delta)
                    .filter(|&x| x.abs() < 2 * p64),
            );
        }
        if p <= 257 {
            inputs.extend(-2 * p64 + 1..2 * p64);
        }
        let mut state = 0x483aef827u64;
        for _ in 0..8192 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            inputs.push((state % (4 * p64 - 1) as u64) as i64 - 2 * p64 + 1);
        }
        for chunk in inputs.chunks(4) {
            let values = std::array::from_fn::<_, 4, _>(|i| chunk[i % chunk.len()] as i32);
            let mut output = [0i32; 4];
            unsafe {
                vst1q_s32(
                    output.as_mut_ptr(),
                    centered_reduce_4x_i32(
                        vld1q_s32(values.as_ptr()),
                        vdupq_n_s32(p),
                        vdupq_n_s32(reciprocal as i32),
                    ),
                );
            }
            for (&x, &y) in values.iter().zip(&output) {
                assert!(i64::from(y).abs() < p64, "x={x}, y={y}, p={p}");
                assert_eq!(i64::from(y).rem_euclid(p64), i64::from(x).rem_euclid(p64));
                let q = (i128::from(x) * i128::from(reciprocal) + (1i128 << 30)) >> 31;
                assert_eq!(i128::from(y), i128::from(x) - q * i128::from(p));
            }
        }
    }
}

fn check_transform<const D: usize>(prime: NttPrime<i32>) {
    let tw = NttTwiddles::<i32, D>::compute(prime);
    let p = i64::from(prime.p);
    let mut state = 0xfa09c8e41u64;
    for case in 0..5 {
        let input = std::array::from_fn(|i| {
            let x = match case {
                0 => p - 1,
                1 => -p + 1,
                2 => {
                    if i & 1 == 0 {
                        p - 1
                    } else {
                        -p + 1
                    }
                }
                3 => [0, 1, -1, p / 2, -p / 2][i % 5],
                _ => {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % (2 * p - 1) as u64) as i64 - p + 1
                }
            };
            MontCoeff::from_raw(x as i32)
        });
        let mut expected = input;
        forward_ntt(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
        let mut actual = input;
        unsafe {
            forward_ntt_i32(&mut actual, prime, &tw);
        }
        assert_eq!(actual, expected, "D={D}, p={p}, case={case}");

        let mut expected = input;
        forward_ntt_cyclic(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
        let mut actual = input;
        unsafe {
            forward_ntt_cyclic_i32(&mut actual, prime, &tw);
        }
        assert_eq!(actual, expected, "cyclic D={D}, p={p}, case={case}");

        let canonical = |values: [MontCoeff<i32>; D]| values.map(|value| prime.to_canonical(value));
        let mut expected = input;
        inverse_ntt(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
        let mut actual = input;
        unsafe {
            inverse_ntt_i32(&mut actual, prime, &tw);
        }
        assert!(actual.iter().all(|value| i64::from(value.raw()).abs() < p));
        assert_eq!(
            canonical(actual),
            canonical(expected),
            "inverse D={D}, p={p}, case={case}"
        );

        let mut expected = input;
        inverse_ntt_cyclic(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
        let mut actual = input;
        unsafe {
            inverse_ntt_cyclic_i32(&mut actual, prime, &tw);
        }
        assert!(actual.iter().all(|value| i64::from(value.raw()).abs() < p));
        assert_eq!(
            canonical(actual),
            canonical(expected),
            "inverse cyclic D={D}, p={p}, case={case}"
        );
    }

    // The negacyclic forward transform accepts any representative below 2^31.
    let input = std::array::from_fn(|i| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = [i32::MAX, -i32::MAX, (state >> 32) as i32][i % 3];
        MontCoeff::from_raw(x.max(-i32::MAX))
    });
    let mut expected = input;
    forward_ntt(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
    let mut actual = input;
    unsafe {
        forward_ntt_i32(&mut actual, prime, &tw);
    }
    assert_eq!(actual, expected, "wide input D={D}, p={p}");
    let digits = std::array::from_fn(|i| (i as i8).wrapping_mul(37).wrapping_add(11));
    let mut expected = std::array::from_fn(|i| prime.from_canonical(i32::from(digits[i])));
    forward_ntt(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
    let mut actual = [MontCoeff::from_raw(0); D];
    unsafe {
        forward_ntt_i8_i32(&mut actual, &digits, prime, &tw);
    }
    assert_eq!(actual, expected, "signed digits D={D}, p={p}");
    let digit_cases: [[i8; D]; 4] = [
        digits,
        [i8::MIN; D],
        [i8::MAX; D],
        std::array::from_fn(|i| [i8::MIN, i8::MAX, -1, 0, 1][i % 5]),
    ];
    for (case, digits) in digit_cases.into_iter().enumerate() {
        let mut expected = digits.map(|x| prime.from_canonical(i32::from(x)));
        forward_ntt_cyclic(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
        let mut actual = [MontCoeff::from_raw(0); D];
        unsafe {
            forward_ntt_cyclic_i8_i32(&mut actual, &digits, prime, &tw);
        }
        assert_eq!(
            actual, expected,
            "cyclic signed digits D={D}, p={p}, case={case}"
        );
    }
    let coefficients: [i16; D] = std::array::from_fn(|i| match i % 5 {
        0 => i16::MIN,
        1 => i16::MAX,
        _ => (i as i16).wrapping_mul(7919).wrapping_add(-1234),
    });
    let mut expected = coefficients.map(|x| prime.from_canonical(i32::from(x)));
    forward_ntt(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
    let mut actual = [MontCoeff::from_raw(0); D];
    unsafe {
        forward_ntt_centered_i16_i32(&mut actual, &coefficients, prime, &tw);
    }
    assert_eq!(actual, expected, "centered i16 D={D}, p={p}");
}

#[test]
fn transforms_match_scalar_for_negacyclic_cyclic_and_signed_inputs() {
    for p in Q128_RAW_PRIMES.into_iter().chain(Q64_PRIMES.map(|p| p.p)) {
        let prime = NttPrime::compute(p);
        check_transform::<8>(prime);
        check_transform::<16>(prime);
        check_transform::<32>(prime);
        check_transform::<64>(prime);
        check_transform::<128>(prime);
        check_transform::<256>(prime);
        check_transform::<512>(prime);
        check_transform::<1024>(prime);
    }
    for prime in Q64_PRIMES {
        check_transform::<2048>(prime);
    }
}

//! Each kernel test returns early when the CPU lacks the kernel's features,
//! so a pass on such a host covers only the scalar code.
//! [`avx512_hardware_tests_do_not_skip`] fails instead; the portability
//! workflow runs it under Intel SDE.

use super::*;
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth, I32_LAZY_DOT_BATCH};
use crate::ntt::tables::{I16_TAIL_PRIME, Q128_RAW_PRIMES, Q64_PRIMES};

const AVX2_ONLY: AvxCpuFeatures = AvxCpuFeatures {
    avx2: true,
    avx512: false,
};

const AVX512: AvxCpuFeatures = AvxCpuFeatures {
    avx2: true,
    avx512: true,
};

const NO_AVX2: AvxCpuFeatures = AvxCpuFeatures {
    avx2: false,
    avx512: false,
};

#[test]
fn avx_mode_defaults_to_avx2_when_supported() {
    assert_eq!(
        select_avx_ntt_mode(None, None, AVX2_ONLY),
        Some(AvxNttMode::Avx2)
    );
    assert_eq!(
        select_avx_ntt_mode(None, None, AVX512),
        Some(AvxNttMode::Avx2)
    );
}

#[test]
fn avx512_is_opt_in_and_requires_the_features() {
    assert_eq!(
        select_avx_ntt_mode(None, Some("1"), AVX512),
        Some(AvxNttMode::Avx512)
    );
    assert_eq!(
        select_avx_ntt_mode(None, Some("1"), AVX2_ONLY),
        Some(AvxNttMode::Avx2)
    );
    assert_eq!(
        select_avx_ntt_mode(None, Some("0"), AVX512),
        Some(AvxNttMode::Avx2)
    );
}

#[test]
fn x86_ntt_requires_avx2() {
    assert_eq!(select_avx_ntt_mode(None, None, NO_AVX2), None);
    assert_eq!(select_avx_ntt_mode(None, Some("1"), NO_AVX2), None);
}

#[test]
fn scalar_kill_switch_disables_x86_ntt_simd() {
    assert_eq!(select_avx_ntt_mode(Some("1"), None, AVX2_ONLY), None);
    assert_eq!(select_avx_ntt_mode(Some("1"), Some("1"), AVX512), None);
}

fn random_mont_array<W: PrimeWidth, const D: usize>(
    prime: NttPrime<W>,
    seed: u64,
) -> [MontCoeff<W>; D] {
    let mut state = seed;
    std::array::from_fn(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        prime.from_canonical(W::from_i64((state >> 33) as i64 % prime.p.to_i64()))
    })
}

fn edge_mont_array_i32<const D: usize>(prime: NttPrime<i32>) -> [MontCoeff<i32>; D] {
    let values = [
        0,
        1,
        -1,
        prime.p - 1,
        1 - prime.p,
        prime.p / 2,
        -(prime.p / 2),
        0x4000_1234_i32,
        -0x3fff_4321_i32,
    ];
    std::array::from_fn(|i| MontCoeff::from_raw(values[i % values.len()]))
}

fn edge_mont_array_i16<const D: usize>(prime: NttPrime<i16>) -> [MontCoeff<i16>; D] {
    let values = [
        0,
        1,
        -1,
        prime.p - 1,
        1 - prime.p,
        prime.p / 2,
        -(prime.p / 2),
        0x3a5a_i16,
        -0x3211_i16,
    ];
    std::array::from_fn(|i| MontCoeff::from_raw(values[i % values.len()]))
}

fn scalar_pointwise_i32<const D: usize>(
    acc: &mut [MontCoeff<i32>; D],
    lhs: &[MontCoeff<i32>; D],
    rhs: &[MontCoeff<i32>; D],
    prime: NttPrime<i32>,
) {
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(prod.raw()));
        acc[i] = prime.reduce_range(sum);
    }
}

fn scalar_pointwise_i16<const D: usize>(
    acc: &mut [MontCoeff<i16>; D],
    lhs: &[MontCoeff<i16>; D],
    rhs: &[MontCoeff<i16>; D],
    prime: NttPrime<i16>,
) {
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(prod.raw()));
        acc[i] = prime.reduce_range(sum);
    }
}

fn scalar_add_reduce_i32<const D: usize>(
    acc: &mut [MontCoeff<i32>; D],
    other: &[MontCoeff<i32>; D],
    prime: NttPrime<i32>,
) {
    for i in 0..D {
        let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(other[i].raw()));
        acc[i] = prime.reduce_range(sum);
    }
}

fn assert_i32_crt_ops<const D: usize>() {
    for (prime_index, raw_prime) in Q128_RAW_PRIMES.into_iter().enumerate() {
        let prime = NttPrime::compute(raw_prime);
        let lhs = edge_mont_array_i32::<D>(prime);
        let rhs = random_mont_array::<_, D>(prime, 0x9173 + prime_index as u64);

        let mut expected_add = lhs;
        scalar_add_reduce_i32(&mut expected_add, &rhs, prime);

        if std::is_x86_feature_detected!("avx2") {
            let mut add = lhs;
            unsafe {
                add_reduce_i32(
                    add.as_mut_ptr().cast(),
                    add.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                );
            }
            assert_eq!(add, expected_add, "AVX2 add p={raw_prime} D={D}");
        }

        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw")
        {
            let mut add = lhs;
            unsafe {
                add_reduce_i32_avx512(
                    add.as_mut_ptr().cast(),
                    add.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                );
            }
            assert_eq!(add, expected_add, "AVX-512 add p={raw_prime} D={D}");
        }
    }
}

#[test]
fn q128_i32_crt_ops_match_scalar_at_vector_boundaries() {
    assert_i32_crt_ops::<7>();
    assert_i32_crt_ops::<8>();
    assert_i32_crt_ops::<9>();
    assert_i32_crt_ops::<15>();
    assert_i32_crt_ops::<16>();
    assert_i32_crt_ops::<17>();
    assert_i32_crt_ops::<31>();
    assert_i32_crt_ops::<32>();
    assert_i32_crt_ops::<33>();
}

fn scalar_add_reduce_i16<const D: usize>(
    acc: &mut [MontCoeff<i16>; D],
    other: &[MontCoeff<i16>; D],
    prime: NttPrime<i16>,
) {
    for i in 0..D {
        let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(other[i].raw()));
        acc[i] = prime.reduce_range(sum);
    }
}

/// A transform entry with the scalar reference's signature.
type Transform<'a, W, const D: usize> =
    &'a dyn Fn(&mut [MontCoeff<W>; D], NttPrime<W>, &NttTwiddles<W, D>);

/// A fused forward entry that converts `[S; D]` inputs on load.
type FusedTransform<'a, W, S, const D: usize> =
    &'a dyn Fn(&mut [MontCoeff<W>; D], &[S; D], NttPrime<W>, &NttTwiddles<W, D>);

fn scalar_forward_ntt<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    for (ai, psi) in a.iter_mut().zip(tw.psi_pows.iter()) {
        *ai = prime.mul(*ai, *psi);
    }
    scalar_forward_ntt_cyclic(a, prime, tw);
}

fn scalar_inverse_ntt<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    scalar_inverse_dit(a, prime, tw);
    for (ai, fused) in a.iter_mut().zip(tw.d_inv_psi_inv.iter()) {
        *ai = prime.mul(*ai, *fused);
    }
}

fn scalar_forward_ntt_cyclic<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        for start in (0..D).step_by(2 * len) {
            for j in 0..len {
                let w = tw.fwd_twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = a[start + j + len];
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
            }
        }
        len /= 2;
    }
    prime.reduce_range_in_place(a);
}

fn scalar_inverse_ntt_cyclic<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    scalar_inverse_dit(a, prime, tw);
    for c in a.iter_mut() {
        *c = prime.mul(*c, tw.d_inv);
    }
}

/// The unscaled inverse DIT stages shared by both scalar inverses.
fn scalar_inverse_dit<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        for start in (0..D).step_by(2 * len) {
            for j in 0..len {
                let w = tw.inv_twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = prime.mul(a[start + j + len], w);
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
            }
        }
        len *= 2;
    }
}

fn canonical<W: PrimeWidth, const D: usize>(a: &[MontCoeff<W>; D], p: W) -> [i64; D] {
    a.map(|x| x.raw().to_i64().rem_euclid(p.to_i64()))
}

/// Whether the host can run the AVX-512 `i32` transform instantiations.
fn avx512_transform_available() -> bool {
    runtime::detect_cpu_features().avx512
}

/// The x86 transforms `[forward, inverse, forward_cyclic, inverse_cyclic]`
/// agree with the scalar reference modulo `p`; forward outputs are canonical
/// and inverse outputs lie in `(-p, p)`.
fn assert_transforms_match_scalar<W: PrimeWidth, const D: usize>(
    prime: NttPrime<W>,
    [forward, inverse, forward_cyclic, inverse_cyclic]: [Transform<'_, W, D>; 4],
    context: &str,
) {
    let p = prime.p.to_i64();
    let tw = NttTwiddles::<W, D>::compute(prime);
    let random = random_mont_array::<W, D>(prime, 0x5150 ^ D as u64 ^ p as u64);
    let half = 1_i64 << (W::R_LOG - 1);
    let extremes: [MontCoeff<W>; D] = std::array::from_fn(|index| {
        MontCoeff::from_raw(W::from_i64([-half, half - 1, 1 - p, p - 1, 0][index % 5]))
    });
    let in_range = extremes.map(|x| MontCoeff::from_raw(W::from_i64(x.raw().to_i64() % p)));
    let canonical_output =
        |a: &[MontCoeff<W>; D]| a.iter().all(|x| (0..p).contains(&x.raw().to_i64()));
    let signed_output = |a: &[MontCoeff<W>; D]| a.iter().all(|x| x.raw().to_i64().abs() < p);

    // The negacyclic forward transform accepts any input.
    for input in [random, in_range, extremes] {
        let mut avx = input;
        let mut scalar =
            input.map(|x| MontCoeff::from_raw(W::from_i64(x.raw().to_i64().rem_euclid(p))));
        forward(&mut avx, prime, &tw);
        scalar_forward_ntt(&mut scalar, prime, &tw);
        assert!(canonical_output(&avx), "{context}");
        assert_eq!(
            canonical(&avx, prime.p),
            canonical(&scalar, prime.p),
            "{context}"
        );
    }

    type Scalar<W, const D: usize> = fn(&mut [MontCoeff<W>; D], NttPrime<W>, &NttTwiddles<W, D>);
    let cases: [(Transform<'_, W, D>, Scalar<W, D>, bool); 3] = [
        (inverse, scalar_inverse_ntt, false),
        (forward_cyclic, scalar_forward_ntt_cyclic, true),
        (inverse_cyclic, scalar_inverse_ntt_cyclic, false),
    ];
    for input in [random, in_range] {
        for (x86, scalar_reference, forward_output) in cases {
            let mut avx = input;
            let mut scalar = input;
            x86(&mut avx, prime, &tw);
            scalar_reference(&mut scalar, prime, &tw);
            if forward_output {
                assert!(canonical_output(&avx), "{context}");
            } else {
                assert!(signed_output(&avx), "{context}");
            }
            assert_eq!(
                canonical(&avx, prime.p),
                canonical(&scalar, prime.p),
                "{context}"
            );
        }
    }

    // Round trip through the negacyclic pair returns the input.
    let mut round_trip = random;
    forward(&mut round_trip, prime, &tw);
    inverse(&mut round_trip, prime, &tw);
    assert_eq!(
        canonical(&round_trip, prime.p),
        canonical(&random, prime.p),
        "{context}"
    );
}

/// The shipped `i32` primes that support negacyclic degree `D`: all of the
/// q128 set through 1024 and the q64 set at 2048.
fn i32_primes<const D: usize>() -> impl Iterator<Item = i32> {
    Q128_RAW_PRIMES
        .into_iter()
        .chain(Q64_PRIMES.map(|prime| prime.p))
        .filter(|p| ((p - 1) as usize).is_multiple_of(2 * D))
}

fn assert_ntt_i32_transforms_match_scalar<const D: usize>(use_avx512: bool) {
    // SAFETY (each entry): the caller checks the target features.
    for raw_prime in i32_primes::<D>() {
        assert_transforms_match_scalar::<i32, D>(
            NttPrime::compute(raw_prime),
            [
                &|a, prime, tw| unsafe { forward_ntt_i32(a, prime, tw, use_avx512) },
                &|a, prime, tw| unsafe { inverse_ntt_i32(a, prime, tw, use_avx512) },
                &|a, prime, tw| unsafe { forward_ntt_cyclic_i32(a, prime, tw, use_avx512) },
                &|a, prime, tw| unsafe { inverse_ntt_cyclic_i32(a, prime, tw, use_avx512) },
            ],
            &format!("prime={raw_prime}, D={D}, avx512={use_avx512}"),
        );
    }
}

fn assert_ntt_i32_transforms_match_scalar_all_sizes(use_avx512: bool) {
    assert_ntt_i32_transforms_match_scalar::<64>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<128>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<256>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<512>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<1024>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<2048>(use_avx512);
}

#[test]
fn avx2_ntt_i32_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    assert_ntt_i32_transforms_match_scalar_all_sizes(false);
}

#[test]
fn avx512_ntt_i32_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") || !avx512_transform_available() {
        return;
    }
    assert_ntt_i32_transforms_match_scalar_all_sizes(true);
}

/// The fused signed-digit (negacyclic and cyclic) and centered-i16 forward
/// transforms agree with converting through `from_canonical` and running the
/// scalar reference.
fn assert_fused_transforms_match_scalar<W: PrimeWidth, const D: usize>(
    prime: NttPrime<W>,
    digits_entry: FusedTransform<'_, W, i8, D>,
    cyclic_digits_entry: FusedTransform<'_, W, i8, D>,
    centered_entry: FusedTransform<'_, W, i16, D>,
    context: &str,
) {
    let tw = NttTwiddles::<W, D>::compute(prime);
    let digits: [i8; D] =
        std::array::from_fn(|index| [i8::MIN, -17, -1, 0, 1, 13, 63, i8::MAX][index % 8]);
    let coefficients: [i16; D] = std::array::from_fn(|index| {
        [i16::MIN, -1024, -17, -1, 0, 1, 1023, i16::MAX][index % 8] ^ (index as i16 & 0x70)
    });
    let expected = |values: [i64; D]| {
        let mut expected = values.map(|x| prime.from_canonical(W::from_i64(x)));
        scalar_forward_ntt(&mut expected, prime, &tw);
        canonical(&expected, prime.p)
    };
    let mut actual = [MontCoeff::from_raw(W::default()); D];

    digits_entry(&mut actual, &digits, prime, &tw);
    assert_eq!(
        canonical(&actual, prime.p),
        expected(digits.map(i64::from)),
        "digits, {context}"
    );
    assert!(actual
        .iter()
        .all(|x| (0..prime.p.to_i64()).contains(&x.raw().to_i64())));

    cyclic_digits_entry(&mut actual, &digits, prime, &tw);
    let mut expected_cyclic = digits.map(|x| prime.from_canonical(W::from_i64(i64::from(x))));
    scalar_forward_ntt_cyclic(&mut expected_cyclic, prime, &tw);
    assert_eq!(
        canonical(&actual, prime.p),
        canonical(&expected_cyclic, prime.p),
        "cyclic digits, {context}"
    );
    assert!(actual
        .iter()
        .all(|x| (0..prime.p.to_i64()).contains(&x.raw().to_i64())));

    centered_entry(&mut actual, &coefficients, prime, &tw);
    assert_eq!(
        canonical(&actual, prime.p),
        expected(coefficients.map(i64::from)),
        "centered, {context}"
    );
    assert!(actual
        .iter()
        .all(|x| (0..prime.p.to_i64()).contains(&x.raw().to_i64())));
}

fn assert_fused_ntt_i32_matches_scalar<const D: usize>(use_avx512: bool) {
    // SAFETY (each entry): the caller checks the target features.
    for raw_prime in i32_primes::<D>() {
        assert_fused_transforms_match_scalar::<i32, D>(
            NttPrime::compute(raw_prime),
            &|a, digits, prime, tw| unsafe { forward_ntt_i8_i32(a, digits, prime, tw, use_avx512) },
            &|a, digits, prime, tw| unsafe {
                forward_ntt_cyclic_i8_i32(a, digits, prime, tw, use_avx512)
            },
            &|a, coefficients, prime, tw| unsafe {
                forward_ntt_centered_i16_i32(a, coefficients, prime, tw, use_avx512)
            },
            &format!("prime={raw_prime}, D={D}, avx512={use_avx512}"),
        );
    }
}

#[test]
fn fused_ntt_i32_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut widths = vec![false];
    if avx512_transform_available() {
        widths.push(true);
    }
    for use_avx512 in widths {
        assert_fused_ntt_i32_matches_scalar::<64>(use_avx512);
        assert_fused_ntt_i32_matches_scalar::<128>(use_avx512);
        assert_fused_ntt_i32_matches_scalar::<256>(use_avx512);
        assert_fused_ntt_i32_matches_scalar::<512>(use_avx512);
        assert_fused_ntt_i32_matches_scalar::<1024>(use_avx512);
        assert_fused_ntt_i32_matches_scalar::<2048>(use_avx512);
    }
}

/// The `i16` NTT primes the crate ships (the CRT tail and the i16 limb set).
const I16_PRIMES: [i16; 3] = [I16_TAIL_PRIME.p, 13313, 15361];

fn assert_ntt_i16_transforms_match_scalar<const D: usize>() {
    // SAFETY (each entry): the caller checks AVX2 support.
    // 13313 and 15361 support negacyclic degrees through 512, and 12289
    // through 2048.
    for raw_prime in I16_PRIMES
        .into_iter()
        .filter(|p| ((p - 1) as usize).is_multiple_of(2 * D))
    {
        let prime = NttPrime::compute(raw_prime);
        let context = format!("prime={raw_prime}, D={D}");
        assert_transforms_match_scalar::<i16, D>(
            prime,
            [
                &|a, prime, tw| unsafe { forward_ntt_i16(a, prime, tw) },
                &|a, prime, tw| unsafe { inverse_ntt_i16(a, prime, tw) },
                &|a, prime, tw| unsafe { forward_ntt_cyclic_i16(a, prime, tw) },
                &|a, prime, tw| unsafe { inverse_ntt_cyclic_i16(a, prime, tw) },
            ],
            &context,
        );
        assert_fused_transforms_match_scalar::<i16, D>(
            prime,
            &|a, digits, prime, tw| unsafe { forward_ntt_i8_i16(a, digits, prime, tw) },
            &|a, digits, prime, tw| unsafe { forward_ntt_cyclic_i8_i16(a, digits, prime, tw) },
            &|a, coefficients, prime, tw| unsafe {
                forward_ntt_centered_i16_i16(a, coefficients, prime, tw)
            },
            &context,
        );
    }
}

#[test]
fn avx2_ntt_i16_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    assert_ntt_i16_transforms_match_scalar::<64>();
    assert_ntt_i16_transforms_match_scalar::<128>();
    assert_ntt_i16_transforms_match_scalar::<256>();
    assert_ntt_i16_transforms_match_scalar::<512>();
    assert_ntt_i16_transforms_match_scalar::<1024>();
    assert_ntt_i16_transforms_match_scalar::<2048>();
}

#[test]
fn avx2_pointwise_mul_acc_i32_matches_scalar_with_tail() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    const D: usize = 19;
    let acc_init = random_mont_array::<_, D>(prime, 0x1111);
    let lhs = edge_mont_array_i32::<D>(prime);
    let rhs = random_mont_array::<_, D>(prime, 0x3333);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by the runtime AVX2 detection above.
    unsafe {
        pointwise_mul_acc_i32(
            avx_acc.as_mut_ptr() as *mut i32,
            lhs.as_ptr() as *const i32,
            rhs.as_ptr() as *const i32,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_pointwise_i32(&mut scalar_acc, &lhs, &rhs, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx2_lazy_i32_dot_matches_repeated_reduction() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    const D: usize = 19;
    for raw_prime in Q128_RAW_PRIMES {
        let prime = NttPrime::compute(raw_prime);
        let lhs: [[MontCoeff<i32>; D]; I32_LAZY_DOT_BATCH] = std::array::from_fn(|index| {
            if index == 0 {
                edge_mont_array_i32(prime)
            } else {
                random_mont_array(prime, 0x1000 + index as u64)
            }
        });
        let rhs: [[MontCoeff<i32>; D]; I32_LAZY_DOT_BATCH] = std::array::from_fn(|index| {
            if index + 1 == I32_LAZY_DOT_BATCH {
                edge_mont_array_i32(prime)
            } else {
                random_mont_array(prime, 0x2000 + index as u64)
            }
        });
        let lhs_pointers: [*const i32; I32_LAZY_DOT_BATCH] =
            std::array::from_fn(|index| lhs[index].as_ptr().cast::<i32>());
        let rhs_pointers: [*const i32; I32_LAZY_DOT_BATCH] =
            std::array::from_fn(|index| rhs[index].as_ptr().cast::<i32>());

        for count in 1..=I32_LAZY_DOT_BATCH {
            let initial = random_mont_array::<_, D>(prime, 0x3000 + count as u64);
            let mut actual = initial;
            // SAFETY: guarded by runtime AVX2 detection; every pointer covers D values.
            unsafe {
                pointwise_dot_acc_i32(
                    actual.as_mut_ptr().cast::<i32>(),
                    lhs_pointers.as_ptr(),
                    rhs_pointers.as_ptr(),
                    count,
                    D,
                    prime.p,
                    prime.pinv,
                );
            }
            let mut expected = initial;
            for product in 0..count {
                scalar_pointwise_i32(&mut expected, &lhs[product], &rhs[product], prime);
            }
            assert_eq!(actual, expected, "prime={raw_prime}, count={count}");
        }
    }
}

#[test]
fn avx2_add_reduce_i32_matches_scalar_with_tail() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    const D: usize = 19;
    let acc_init = random_mont_array::<_, D>(prime, 0x4444);
    let other = edge_mont_array_i32::<D>(prime);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by the runtime AVX2 detection above.
    unsafe {
        add_reduce_i32(
            avx_acc.as_mut_ptr() as *mut i32,
            avx_acc.as_ptr() as *const i32,
            other.as_ptr() as *const i32,
            D,
            prime.p,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_add_reduce_i32(&mut scalar_acc, &other, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx512_pointwise_mul_acc_i32_matches_scalar_with_tail() {
    if !(std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw"))
    {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    const D: usize = 29;
    let acc_init = random_mont_array::<_, D>(prime, 0x5151);
    let lhs = edge_mont_array_i32::<D>(prime);
    let rhs = random_mont_array::<_, D>(prime, 0x7171);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by runtime AVX-512 feature detection above.
    unsafe {
        pointwise_mul_acc_i32_avx512(
            avx_acc.as_mut_ptr() as *mut i32,
            lhs.as_ptr() as *const i32,
            rhs.as_ptr() as *const i32,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_pointwise_i32(&mut scalar_acc, &lhs, &rhs, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx512_add_reduce_i32_matches_scalar_with_tail() {
    if !(std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw"))
    {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    const D: usize = 29;
    let acc_init = random_mont_array::<_, D>(prime, 0x8181);
    let other = edge_mont_array_i32::<D>(prime);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by runtime AVX-512 feature detection above.
    unsafe {
        add_reduce_i32_avx512(
            avx_acc.as_mut_ptr() as *mut i32,
            avx_acc.as_ptr() as *const i32,
            other.as_ptr() as *const i32,
            D,
            prime.p,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_add_reduce_i32(&mut scalar_acc, &other, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx2_pointwise_mul_acc_i16_matches_scalar_with_tail() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(15361_i16);
    const D: usize = 23;
    let acc_init = random_mont_array::<_, D>(prime, 0xaaaa);
    let lhs = edge_mont_array_i16::<D>(prime);
    let rhs = random_mont_array::<_, D>(prime, 0xcccc);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by the runtime AVX2 detection above.
    unsafe {
        pointwise_mul_acc_i16(
            avx_acc.as_mut_ptr() as *mut i16,
            lhs.as_ptr() as *const i16,
            rhs.as_ptr() as *const i16,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_pointwise_i16(&mut scalar_acc, &lhs, &rhs, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx2_add_reduce_i16_matches_scalar_with_tail() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(15361_i16);
    const D: usize = 23;
    let acc_init = random_mont_array::<_, D>(prime, 0xdddd);
    let other = edge_mont_array_i16::<D>(prime);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by the runtime AVX2 detection above.
    unsafe {
        add_reduce_i16(
            avx_acc.as_mut_ptr() as *mut i16,
            other.as_ptr() as *const i16,
            D,
            prime.p,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_add_reduce_i16(&mut scalar_acc, &other, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
#[ignore = "requires AVX-512F/DQ/BW hardware or emulation"]
fn avx512_hardware_tests_do_not_skip() {
    assert!(
        avx512_transform_available(),
        "AVX-512F/DQ/BW is unavailable"
    );
    q128_i32_crt_ops_match_scalar_at_vector_boundaries();
    avx512_ntt_i32_transforms_match_scalar();
    fused_ntt_i32_matches_scalar();
    avx512_pointwise_mul_acc_i32_matches_scalar_with_tail();
    avx512_add_reduce_i32_matches_scalar_with_tail();
}

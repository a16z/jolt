use jolt_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

use super::lut::{balanced_limbs, CenteredMontReducer};
use super::*;
use crate::ntt::butterfly::forward_ntt;
use crate::ntt::prime::NttPrime;
use crate::ntt::tables::{
    q128_primes, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q128_RAW_PRIMES, Q32_NUM_PRIMES, Q32_PRIMES,
    Q64_NUM_PRIMES, Q64_PRIMES,
};
use crate::CyclotomicRing;

const SYNTHETIC_I16_NUM_PRIMES: usize = 3;

fn synthetic_i16_primes() -> [NttPrime<i16>; SYNTHETIC_I16_NUM_PRIMES] {
    [
        NttPrime::compute(15361_i16),
        NttPrime::compute(13313_i16),
        NttPrime::compute(12289_i16),
    ]
}

fn reducer_cases() -> Vec<i128> {
    let mut cases = vec![0, 1, -1, i128::MAX, i128::MIN + 1, i128::MIN];
    for shift in [15, 16, 31, 32, 33, 63, 64, 95, 96, 97, 126] {
        for offset in [-1i128, 0, 1] {
            cases.push((1i128 << shift) + offset);
            cases.push(-(1i128 << shift) + offset);
        }
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u128;
    for _ in 0..512 {
        state = state
            .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645)
            .wrapping_add(0x5851_f42d_4c95_7f2d_1405_7b7e_f767_814f);
        cases.push((state as i128) >> (state % 97));
    }
    cases
}

fn check_reducer<W: PrimeWidth>(prime: NttPrime<W>) {
    let p = prime.p.to_i64();
    let reducer = CenteredMontReducer::new(prime);
    for value in reducer_cases() {
        let limbs = balanced_limbs(value);
        let rebuilt = limbs.iter().rev().fold(0i128, |acc, &limb| {
            (acc << 32).wrapping_add(i128::from(limb))
        });
        assert_eq!(rebuilt, value);
        assert!(limbs[..3].iter().all(|&limb| i32::try_from(limb).is_ok()));
        assert!(limbs[3].abs() <= 1 << 31);

        let mont = reducer.reduce_limbs(limbs);
        assert!(mont.raw().to_i64().abs() < p);
        let expected = value.rem_euclid(i128::from(p)) as i64;
        assert_eq!(prime.to_canonical(mont).to_i64(), expected, "{value}");

        let narrow = value as i32;
        let mont = reducer.reduce_i32(narrow);
        assert!(mont.raw().to_i64().abs() < p);
        let expected = i64::from(narrow).rem_euclid(p);
        assert_eq!(prime.to_canonical(mont).to_i64(), expected, "{narrow}");
    }
}

#[test]
fn centered_mont_reducer_matches_euclidean_residues() {
    for prime in synthetic_i16_primes() {
        check_reducer(prime);
    }
    check_reducer(I16_TAIL_PRIME);
    for prime in Q32_PRIMES.into_iter().chain(Q64_PRIMES) {
        check_reducer(prime);
    }
    for p in Q128_RAW_PRIMES.into_iter().chain([1_073_707_009]) {
        check_reducer(NttPrime::compute(p));
    }
}

/// Canonical coefficients at every centering and 26-bit slice boundary below
/// `modulus`, followed by pseudo-random fill.
fn field_residue_probes<const D: usize>(modulus: u128) -> [u128; D] {
    let half = modulus / 2;
    let mut edges = vec![0, 1, 2, half - 1, half, half + 1, modulus - 2, modulus - 1];
    for bit in (26..128).step_by(26) {
        for value in [1u128 << bit, (1u128 << bit) - 1, half ^ (1u128 << bit)] {
            if value < modulus {
                edges.push(value);
                edges.push(modulus - 1 - value);
            }
        }
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834u128;
    std::array::from_fn(|index| {
        edges.get(index).copied().unwrap_or_else(|| {
            state = state
                .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645)
                .wrapping_add(0x5851_f42d_4c95_7f2d_1405_7b7e_f767_814f);
            state % modulus
        })
    })
}

fn assert_field_residues_match<F, W, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
) where
    F: CrtNttConvertibleField,
    W: PrimeWidth,
{
    let modulus = (-F::one()).to_u128_checked().unwrap() + 1;
    let canonical = field_residue_probes::<D>(modulus);
    let ring = CyclotomicRing::<F, D>::from_coefficients(
        canonical.map(|value| F::from_u128_checked(value).unwrap()),
    );
    let mut residues = [[MontCoeff::from_raw(W::default()); D]; K];
    params.field_residues(&ring, &mut residues);
    for (prime, residues) in params.primes.iter().zip(&residues) {
        let p = i128::from(prime.p.to_i64());
        for (&value, residue) in canonical.iter().zip(residues) {
            let centered = if value > modulus / 2 {
                -((modulus - value) as i128)
            } else {
                value as i128
            };
            let expected =
                prime.from_canonical(prime.center(W::from_i64(centered.rem_euclid(p) as i64)));
            assert!(
                residue.raw().to_i64().unsigned_abs() < prime.p.to_i64() as u64,
                "residue of {value} outside (-p, p) for p = {p}"
            );
            assert_eq!(
                prime.normalize(*residue),
                prime.normalize(expected),
                "residue of {value} modulo {p}"
            );
        }
    }
}

#[test]
fn field_residues_match_centered_reduction() {
    fn check<F: CrtNttConvertibleField>() {
        // Several conversion chunks, so the SIMD kernels see nonzero offsets.
        assert_field_residues_match::<F, i32, Q128_NUM_PRIMES, 256>(&CrtNttParamSet::new(
            q128_primes(),
        ));
        assert_field_residues_match::<F, i32, Q32_NUM_PRIMES, 4>(&CrtNttParamSet::new(Q32_PRIMES));
        assert_field_residues_match::<F, i16, 1, 128>(&CrtNttParamSet::new([I16_TAIL_PRIME]));
        assert_field_residues_match::<F, i16, SYNTHETIC_I16_NUM_PRIMES, 64>(&CrtNttParamSet::new(
            synthetic_i16_primes(),
        ));
    }
    check::<Prime32Offset99>();
    check::<Prime64Offset59>();
    check::<Prime128OffsetA7F7>();
}

#[test]
fn wide_centered_conversion_matches_per_prime_residues() {
    const D: usize = 64;
    fn check<W: PrimeWidth, const K: usize>(
        params: &CrtNttParamSet<W, K, D>,
        centered: &[i128; D],
    ) {
        let actual = CyclotomicCrtNtt::from_centered_coefficients(centered, params);
        let expected = std::array::from_fn(|k| {
            let prime = params.primes[k];
            let modulus = i128::from(prime.p.to_i64());
            let mut limb = centered
                .map(|value| prime.from_canonical(W::from_i64(value.rem_euclid(modulus) as i64)));
            forward_ntt(&mut limb, prime, &params.twiddles[k], params.kernel_plan);
            limb
        });
        assert_eq!(actual.limbs, expected);
    }

    // The 7,000-valued coefficients fit the first i16 prime's centered
    // range but require wide reduction for the other two primes.
    let mixed = std::array::from_fn(|index| match index % 4 {
        0 => 7_000,
        1 => -7_000,
        2 => 1,
        _ => 0,
    });
    check(&CrtNttParamSet::new(synthetic_i16_primes()), &mixed);

    let wide = std::array::from_fn(|index| match index % 4 {
        0 => 1i128 << 100,
        1 => -(1i128 << 100),
        2 => (1i128 << 63) + 1,
        _ => 0,
    });
    check(&CrtNttParamSet::new(q128_primes()), &wide);
}

#[test]
fn centered_mont_lut_matches_centered_residue_boundary() {
    const D: usize = 64;
    let primes = synthetic_i16_primes();
    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(primes);
    let prime = params.primes[0];
    let half = i32::from(prime.p) / 2;
    let lut = CenteredMontLut::<i16, SYNTHETIC_I16_NUM_PRIMES>::new(&params, half + 1);

    let canonical = |value| lut.get(0, value).map(|mont| prime.to_canonical(mont));
    assert_eq!(canonical(half), Some(half as i16));
    assert_eq!(canonical(half + 1), Some((half + 1) as i16));
    assert_eq!(
        canonical(-half - 1),
        Some((i32::from(prime.p) - half - 1) as i16)
    );
    assert_eq!(lut.get(0, half + 2), None);
}

#[test]
#[should_panic(expected = "lazy pointwise dot requires an i32 SIMD parameter set")]
fn lazy_pointwise_dot_rejects_non_i32_parameter_sets() {
    const D: usize = 64;
    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(synthetic_i16_primes());
    let lut = DigitMontLut::new_with_digit_bound(&params, 2);
    let mut accs = [CyclotomicCrtNtt::zero()];
    let matrix_row = [CyclotomicCrtNtt::zero()];
    let ntt_mat = [matrix_row.as_slice()];
    let digits = [[0i8; D]];
    let mut scratch = [[MontCoeff::from_raw(0i16); D]; I32_LAZY_DOT_BATCH];

    CyclotomicCrtNtt::add_assign_col_pointwise_dot_i8_multi_with_lut_scratch(
        &mut accs,
        &ntt_mat,
        0,
        &digits,
        &params,
        &lut,
        &mut scratch,
    );
}

/// Only i32 parameters carry the NEON Barrett tables.
#[cfg(target_arch = "aarch64")]
#[test]
fn i16_twiddles_hold_only_montgomery_tables() {
    const D: usize = 1024;
    assert_eq!(size_of::<<i16 as PrimeWidth>::NeonTables<D>>(), 0);
    // Eight D-entry tables, then D^-1 and the stage count (16 bytes with
    // padding), rounded up to the 64-byte struct alignment.
    let tables = 8 * D * size_of::<i16>() + 2 * size_of::<usize>();
    assert_eq!(
        size_of::<crate::ntt::butterfly::NttTwiddles<i16, D>>(),
        tables.next_multiple_of(64)
    );
}

/// Embedding applications and rayon workers may run on 2 MiB stacks, which
/// the workspace `RUST_MIN_STACK` does not raise for an explicit size.
#[test]
fn q128_parameters_build_on_a_2_mib_stack() {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| CrtNttParamSet::<i32, Q128_NUM_PRIMES, 1024>::new(q128_primes()))
        .expect("spawn")
        .join()
        .expect("build Q128 parameters");
}
/// Digit LUT conversions take the fused SIMD paths where the platform has
/// them; both must match the scalar negacyclic and cyclic references.
fn check_digit_lut_conversions<W: PrimeWidth, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
) {
    use crate::ntt::butterfly::forward_ntt_cyclic;
    use crate::ntt::NttKernelPlan;

    for bound in [8_u64, 128] {
        let lut = DigitMontLut::new_with_digit_bound(params, bound);
        let low = -(bound as i64);
        let cases: [[i8; D]; 3] = [
            [low as i8; D],
            [(bound - 1) as i8; D],
            std::array::from_fn(|i| (low + (i as i64 * 37 + 11) % (2 * bound as i64)) as i8),
        ];
        for (case, digits) in cases.iter().enumerate() {
            let mut actual = CyclotomicCrtNtt::<W, K, D>::zero();
            actual.assign_i8_with_lut(digits, params, &lut);
            assert_eq!(
                actual,
                CyclotomicCrtNtt::from_i8_with_params(digits, params),
                "negacyclic D={D}, bound={bound}, case={case}"
            );
            let mut expected = [[MontCoeff::from_raw(W::default()); D]; K];
            for ((limb, prime), tw) in expected
                .iter_mut()
                .zip(params.primes.iter())
                .zip(params.twiddles.iter())
            {
                *limb = digits.map(|digit| prime.from_canonical(W::from_i64(i64::from(digit))));
                forward_ntt_cyclic(limb, *prime, tw, NttKernelPlan::SCALAR);
            }
            actual.assign_i8_cyclic_with_lut(digits, params, &lut);
            for (k, (actual, expected)) in actual.limbs.iter().zip(expected.iter()).enumerate() {
                let p = params.primes[k].p.to_i64();
                let canonical = |x: &MontCoeff<W>| x.raw().to_i64().rem_euclid(p);
                assert!(
                    actual
                        .iter()
                        .map(canonical)
                        .eq(expected.iter().map(canonical)),
                    "cyclic D={D}, bound={bound}, case={case}, limb={k}"
                );
            }
        }
    }
}

#[test]
fn digit_lut_conversions_match_scalar_references() {
    fn all_sizes<W: PrimeWidth, const K: usize>(primes: [NttPrime<W>; K]) {
        check_digit_lut_conversions(&CrtNttParamSet::<W, K, 16>::new(primes));
        check_digit_lut_conversions(&CrtNttParamSet::<W, K, 64>::new(primes));
        check_digit_lut_conversions(&CrtNttParamSet::<W, K, 256>::new(primes));
    }
    all_sizes(Q32_PRIMES);
    all_sizes(Q64_PRIMES);
    all_sizes(q128_primes());
    all_sizes(synthetic_i16_primes());
    all_sizes([I16_TAIL_PRIME]);
}
/// Deterministic centered values in `[-bound, bound]`, including both ends.
fn centered_probe(seed: usize, bound: i128) -> i128 {
    match seed % 5 {
        0 => bound,
        1 => -bound,
        _ => {
            let mixed = (seed as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(17);
            i128::from(mixed).rem_euclid(2 * bound + 1) - bound
        }
    }
}

/// Exact `matrix * rhs` over `Z[X] / (X^D + 1)` by schoolbook multiplication.
fn schoolbook_mat_vec<const D: usize>(
    matrix: &[[i128; D]],
    num_rows: usize,
    rhs: &[[i16; D]],
) -> Vec<[i128; D]> {
    matrix
        .chunks_exact(rhs.len())
        .take(num_rows)
        .map(|row| {
            let mut out = [0i128; D];
            for (entry, digits) in row.iter().zip(rhs) {
                for (i, &lhs) in entry.iter().enumerate() {
                    for (j, &digit) in digits.iter().enumerate() {
                        let term = lhs * i128::from(digit);
                        if i + j < D {
                            out[i + j] += term;
                        } else {
                            out[i + j - D] -= term;
                        }
                    }
                }
            }
            out
        })
        .collect()
}

fn assert_matches_schoolbook<F: Field + CanonicalEncoding, const D: usize>(
    actual: &[CyclotomicRing<F, D>],
    expected: &[[i128; D]],
    modulus: i128,
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let expected = expected.map(|value| {
            let residue = value.rem_euclid(modulus);
            if residue > modulus / 2 {
                residue - modulus
            } else {
                residue
            }
        });
        assert_eq!(actual.centered_coefficients_i128(), expected, "D={D}");
    }
}

fn rings<F: Field + CanonicalEncoding, const D: usize>(
    entries: &[[i128; D]],
) -> Vec<CyclotomicRing<F, D>> {
    entries
        .iter()
        .map(|entry| {
            CyclotomicRing::from_coefficients(
                entry.map(|value| F::from_i64(i64::try_from(value).expect("probe fits i64"))),
            )
        })
        .collect()
}

/// Full-range fp64 entries through the three-prime i32 profile, whose
/// worst case `2 * 3 * D * floor(q / 2) * 128` stays below 2^84 at D = 2048.
fn assert_i32_mat_vec_matches_schoolbook<const D: usize>() {
    type F = Prime64Offset59;
    const Q: i128 = (1 << 64) - 59;
    let (num_rows, num_cols) = (2, 3);
    let params = CrtNttParamSet::<i32, Q64_NUM_PRIMES, D>::new(Q64_PRIMES);
    let entries = (0..num_rows * num_cols)
        .map(|entry| from_fn(|index| centered_probe(entry * D + index, Q / 2)))
        .collect::<Vec<[i128; D]>>();
    let rhs = (0..num_cols)
        .map(|column| from_fn(|index| centered_probe(7 + column * D + index, 128) as i16))
        .collect::<Vec<[i16; D]>>();
    let matrix = rings::<F, D>(&entries)
        .iter()
        .map(|ring| CyclotomicCrtNtt::from_ring(ring, &params))
        .collect::<Vec<_>>();
    let actual = CyclotomicCrtNtt::mat_vec_i16::<F>(&matrix, num_rows, num_cols, &rhs, &params)
        .expect("i32 matvec");
    assert_matches_schoolbook(&actual, &schoolbook_mat_vec(&entries, num_rows, &rhs), Q);
}

#[test]
fn i32_crt_mat_vec_matches_schoolbook() {
    assert_i32_mat_vec_matches_schoolbook::<256>();
    assert_i32_mat_vec_matches_schoolbook::<1024>();
    assert_i32_mat_vec_matches_schoolbook::<2048>();
}

/// A pure i16 profile of about 2^41.2 with `|A| <= 2^20` and `|x| <= 4`.
fn assert_i16_mat_vec_matches_schoolbook<const D: usize>() {
    type F = Prime32Offset99;
    const Q: i128 = (1 << 32) - 99;
    let (num_rows, num_cols) = (2, 2);
    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(synthetic_i16_primes());
    let entries = (0..num_rows * num_cols)
        .map(|entry| from_fn(|index| centered_probe(entry * D + index, 1 << 20)))
        .collect::<Vec<[i128; D]>>();
    let rhs = (0..num_cols)
        .map(|column| from_fn(|index| centered_probe(3 + column * D + index, 4) as i16))
        .collect::<Vec<[i16; D]>>();
    let matrix = rings::<F, D>(&entries)
        .iter()
        .map(|ring| CyclotomicCrtNtt::from_ring(ring, &params))
        .collect::<Vec<_>>();
    let actual = CyclotomicCrtNtt::mat_vec_i16::<F>(&matrix, num_rows, num_cols, &rhs, &params)
        .expect("i16 matvec");
    assert_matches_schoolbook(&actual, &schoolbook_mat_vec(&entries, num_rows, &rhs), Q);
}

#[test]
fn i16_crt_mat_vec_matches_schoolbook() {
    assert_i16_mat_vec_matches_schoolbook::<256>();
    assert_i16_mat_vec_matches_schoolbook::<512>();
}

/// Row 0 is the aligned worst case: constant `floor(q / 2)` entries against
/// constant `-2^15` digits reach `4 * (D - 2) * floor(q / 2) * 2^15`, about
/// 2^91 at D = 2048, past the three i32 primes and within the i16 tail.
#[test]
fn i32_crt_mat_vec_with_i16_tail_matches_schoolbook() {
    const D: usize = 2048;
    type F = Prime64Offset59;
    const Q: i128 = (1 << 64) - 59;
    let (num_rows, num_cols) = (2, 4);
    let wide = CrtNttParamSet::<i32, Q64_NUM_PRIMES, D>::new(Q64_PRIMES);
    let tail = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
    let entries = (0..num_rows * num_cols)
        .map(|entry| {
            from_fn(|index| {
                if entry < num_cols {
                    Q / 2
                } else {
                    centered_probe(entry * D + index, Q / 2)
                }
            })
        })
        .collect::<Vec<[i128; D]>>();
    let rhs = vec![[i16::MIN; D]; num_cols];
    let rings = rings::<F, D>(&entries);
    let wide_matrix = rings
        .iter()
        .map(|ring| CyclotomicCrtNtt::from_ring(ring, &wide))
        .collect::<Vec<_>>();
    let tail_matrix = rings
        .iter()
        .map(|ring| CyclotomicCrtNtt::from_ring(ring, &tail))
        .collect::<Vec<_>>();
    let expected = schoolbook_mat_vec(&entries, num_rows, &rhs);

    let actual = mat_vec_i16_with_tail::<F, Q64_NUM_PRIMES, D>(
        &wide_matrix,
        &tail_matrix,
        num_rows,
        num_cols,
        &rhs,
        &I16TailParams::new(wide.clone(), tail),
    )
    .expect("tail matvec");
    assert_matches_schoolbook(&actual, &expected, Q);

    // The wide primes alone wrap on the aligned row, so the tail is load-bearing.
    let wide_only =
        CyclotomicCrtNtt::mat_vec_i16::<F>(&wide_matrix, num_rows, num_cols, &rhs, &wide)
            .expect("wide matvec");
    assert_ne!(wide_only[0], actual[0]);
}

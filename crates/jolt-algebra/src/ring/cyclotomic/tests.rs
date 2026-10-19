use super::*;
use jolt_field::{Fp128x8i32, Fp32x2i32, Fp64x4i32, WithCommitAccumulator};
use jolt_field::{Fp64, Prime128Offset275, Prime32Offset99, Prime64Offset59};
use rand::rngs::StdRng;
use rand::SeedableRng;

type F64 = Fp64<4294967197>;
type F128 = Prime128Offset275;
type F32 = Prime32Offset99;
/// Production 64-bit prime: unlike `F64`, its canonical values fill all four
/// 16-bit commit lanes.
type F64Full = Prime64Offset59;
const D: usize = 64;

#[test]
fn cyclotomic_ring_satisfies_jolt_ring_core() {
    fn assert_ring_core<R: Ring>() {}
    assert_ring_core::<CyclotomicRing<F64, D>>();

    let x = CyclotomicRing::<F64, D>::x();
    assert_eq!(x.square(), x * x);
    assert_eq!(
        [x, CyclotomicRing::one()]
            .into_iter()
            .product::<CyclotomicRing<F64, D>>(),
        x
    );
}

#[test]
fn shift_accumulate_into_matches_negacyclic_shift() {
    let mut rng = StdRng::seed_from_u64(0x1234);
    let a = CyclotomicRing::<F64, D>::random(&mut rng);
    let dst = CyclotomicRing::<F64, D>::random(&mut rng);

    for k in 0..D {
        let expected = dst + a.negacyclic_shift(k);
        let mut actual = dst;
        a.shift_accumulate_into(&mut actual, k);
        assert_eq!(actual, expected, "shift_accumulate_into k={k}");
    }
}

#[test]
fn shift_sub_into_matches_negacyclic_shift() {
    let mut rng = StdRng::seed_from_u64(0x1234);
    let a = CyclotomicRing::<F64, D>::random(&mut rng);
    let dst = CyclotomicRing::<F64, D>::random(&mut rng);

    for k in 0..D {
        let expected = dst - a.negacyclic_shift(k);
        let mut actual = dst;
        a.shift_sub_into(&mut actual, k);
        assert_eq!(actual, expected, "shift_sub_into k={k}");
    }
}

#[test]
fn shift_scale_accumulate_into_matches_scaled_negacyclic_shift() {
    let mut rng = StdRng::seed_from_u64(0x2468);
    let a = CyclotomicRing::<F64, D>::random(&mut rng);
    let dst = CyclotomicRing::<F64, D>::random(&mut rng);
    let scales = [
        F64::zero(),
        F64::one(),
        -F64::one(),
        F64::from_u64(7),
        F64::from_u64(4294967196),
    ];

    for k in 0..D {
        for &scale in &scales {
            let mut actual = dst;
            a.shift_scale_accumulate_into(&mut actual, k, scale);

            let expected = dst + a.scale(&scale).negacyclic_shift(k);
            assert_eq!(
                actual, expected,
                "shift_scale_accumulate_into k={k} scale={scale:?}"
            );
        }
    }
}

#[test]
fn scale_accumulate_into_matches_separate_scale_and_add() {
    let mut rng = StdRng::seed_from_u64(0x1357_2468);
    let source = CyclotomicRing::<F128, D>::random(&mut rng);
    let initial = CyclotomicRing::<F128, D>::random(&mut rng);
    for scale in [F128::zero(), F128::one(), -F128::one(), F128::from_u64(19)] {
        let mut actual = initial;
        source.scale_accumulate_into(&mut actual, scale);
        assert_eq!(actual, initial + source.scale(&scale));
    }
}

#[test]
fn wide_shift_accumulate_matches_narrow_fp64() {
    let mut rng = StdRng::seed_from_u64(0x1234);
    let src = CyclotomicRing::<F64, D>::random(&mut rng);
    let initial = CyclotomicRing::<F64, D>::random(&mut rng);

    for k in 0..D {
        let mut narrow = initial;
        src.shift_accumulate_into(&mut narrow, k);

        let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
        let mut wide_dst = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&initial);
        wide_src.shift_accumulate_into(&mut wide_dst, k);
        let wide_reduced: CyclotomicRing<F64, D> = wide_dst.reduce();

        assert_eq!(narrow, wide_reduced, "shift_accumulate k={k}");
    }
}

#[test]
fn wide_mul_by_monomial_sum_matches_narrow_fp64() {
    let mut rng = StdRng::seed_from_u64(0xabcd);
    let src = CyclotomicRing::<F64, D>::random(&mut rng);
    let positions = vec![0, 5, 17, 42, 63];

    let mut narrow = CyclotomicRing::<F64, D>::zero();
    for &k in &positions {
        narrow += src.negacyclic_shift(k);
    }

    let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
    let mut wide_dst = WideCyclotomicRing::<Fp64x4i32, D>::zero();
    for &k in &positions {
        wide_src.shift_accumulate_into(&mut wide_dst, k);
    }
    let wide_reduced: CyclotomicRing<F64, D> = wide_dst.reduce();

    assert_eq!(narrow, wide_reduced);
}

#[test]
fn wide_shift_accumulation_matches_narrow_at_field_cap() {
    let src = CyclotomicRing::<F64, D>::from_coefficients([-F64::one(); D]);
    let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
    let mut narrow = CyclotomicRing::<F64, D>::zero();
    let mut wide = WideCyclotomicRing::<Fp64x4i32, D>::zero();

    for _ in 0..<F64 as WithCommitAccumulator>::MAX_COMMIT_ACCUMULATIONS {
        src.shift_accumulate_into(&mut narrow, 0);
        wide_src.shift_accumulate_into(&mut wide, 0);
    }

    assert_eq!(wide.reduce::<F64>(), narrow);
}

#[test]
fn wide_many_accumulations_fp128() {
    let mut rng = StdRng::seed_from_u64(0xbeef);
    let src = CyclotomicRing::<F128, D>::random(&mut rng);

    let mut narrow = CyclotomicRing::<F128, D>::zero();
    let wide_src = WideCyclotomicRing::<Fp128x8i32, D>::from_ring(&src);
    let mut wide_dst = WideCyclotomicRing::<Fp128x8i32, D>::zero();

    for k in 0..50 {
        src.shift_accumulate_into(&mut narrow, k % D);
        wide_src.shift_accumulate_into(&mut wide_dst, k % D);
    }

    let wide_reduced: CyclotomicRing<F128, D> = wide_dst.reduce();
    assert_eq!(narrow, wide_reduced);
}

fn decompose_i8(ring: &CyclotomicRing<F128, D>, levels: usize, log_basis: u32) -> Vec<[i8; D]> {
    let mut digits = vec![[0i8; D]; levels];
    ring.balanced_decompose_pow2_i8_into_with_params(
        &mut digits,
        &BalancedDecomposePow2Params::new(levels, log_basis),
    );
    digits
}

#[test]
fn asymmetric_centering_boundary_roundtrip_fp128() {
    let q = (-F128::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let i128_max = i128::MAX as u128;

    for &(log_basis, levels) in &[(2u32, 64usize), (4u32, 32usize)] {
        let threshold = decompose_centering_threshold(levels, log_basis, q);
        let boundary_values = [
            0,
            1,
            threshold.saturating_sub(1),
            threshold,
            threshold + 1,
            q - i128_max - 1,
            q - i128_max,
            q - 2,
            q - 1,
        ];
        let ring = CyclotomicRing::<F128, D>::from_coefficients(from_fn(|i| {
            F128::from_u128_reduced(boundary_values[i % boundary_values.len()])
        }));

        let i8_digits = decompose_i8(&ring, levels, log_basis);
        let recomposed_i8 = CyclotomicRing::gadget_recompose_pow2_i8(&i8_digits, log_basis);
        assert_eq!(
            ring, recomposed_i8,
            "i8 roundtrip failed for log_basis={log_basis}, levels={levels}"
        );
    }
}

#[test]
fn fp32_i8_decomposition_with_zero_levels_is_a_noop() {
    let params = BalancedDecomposePow2Params::new(0, 8);
    let coefficients = [F32::one(); D];
    let mut output = [];

    balanced_decompose_coefficients_pow2_i8_into(&coefficients, &mut output, &params);
    assert!(output.is_empty());
}

#[test]
fn balanced_i16_decomposition_supports_bases_ten_and_eleven() {
    let ring = CyclotomicRing::<F128, D>::from_coefficients(from_fn(|i| match i % 6 {
        0 => F128::from_i64(-1024),
        1 => F128::from_i64(-512),
        2 => F128::from_i64(-1),
        3 => F128::zero(),
        4 => F128::from_i64(511),
        _ => F128::from_i64(1023),
    }));

    for log_basis in [10, 11] {
        let mut digits = vec![[0i16; D]; 12];
        ring.balanced_decompose_pow2_i16_into(
            &mut digits,
            &BalancedDecomposePow2Params::new(12, log_basis),
        );
        let bound = 1i16 << (log_basis - 1);
        assert!(digits
            .iter()
            .flatten()
            .all(|digit| (-bound..bound).contains(digit)));
        for coefficient in 0..D {
            let mut recomposed = F128::zero();
            let mut power = F128::one();
            let basis = F128::from_u64(1u64 << log_basis);
            for plane in &digits {
                recomposed += F128::from_i64(i64::from(plane[coefficient])) * power;
                power *= basis;
            }
            assert_eq!(recomposed, ring.coeffs[coefficient]);
        }
    }
}

#[test]
fn balanced_i8_decomposition_includes_bases_seven_and_eight() {
    let ring = CyclotomicRing::<F128, D>::from_coefficients(from_fn(|i| match i % 4 {
        0 => F128::from_i64(-128),
        1 => F128::from_i64(-64),
        2 => F128::from_i64(63),
        _ => F128::from_i64(127),
    }));
    for log_basis in [7, 8] {
        let digits = decompose_i8(&ring, 16, log_basis);
        let recomposed = CyclotomicRing::gadget_recompose_pow2_i8(&digits, log_basis);
        assert_eq!(recomposed, ring);
    }
}

#[test]
fn coefficient_inner_product_is_constant_term_of_conjugate_product() {
    let mut rng = StdRng::seed_from_u64(0xC7);
    for _ in 0..16 {
        let a = CyclotomicRing::<F128, D>::random(&mut rng);
        let b = CyclotomicRing::<F128, D>::random(&mut rng);
        assert_eq!(a.constant_term(), a.coefficients()[0]);
        assert_eq!(
            a.coefficient_inner_product(&b),
            (a * b.sigma_m1()).constant_term()
        );
        assert_eq!(
            a.coefficient_inner_product(&b),
            b.coefficient_inner_product(&a)
        );
    }
    let x = CyclotomicRing::<F64, D>::x();
    assert_eq!(x.sigma_m1().coefficients()[D - 1], -F64::one());
    assert_eq!(x.coefficient_inner_product(&x), F64::one());
    assert_eq!(x.constant_term(), F64::zero());
}

fn check_shift_windows_match_wide_accumulation<F, const D: usize>(seed: u64)
where
    F: Field + WithCommitAccumulator,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let edge = CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| match i % 4 {
        0 => -F::one(),
        1 => F::one(),
        2 => F::zero(),
        _ => -F::from_u64(2),
    }));
    let mut windows = NegacyclicShiftWindows::<F, D>::default();
    for src in [CyclotomicRing::<F, D>::random(&mut rng), edge] {
        windows.load(&src);
        let wide_src = WideCyclotomicRing::<F::Wide, D>::from_ring(&src);
        for batch in [0, 1, 3, 7, 11, 2 * D] {
            let initial = CyclotomicRing::<F, D>::random(&mut rng);
            let shifts: Vec<usize> = (0..batch)
                .map(|_| (rng.next_u64() % D as u64) as usize)
                .collect();
            let mut expected = WideCyclotomicRing::<F::Wide, D>::from_ring(&initial);
            for &shift in &shifts {
                wide_src.shift_accumulate_into(&mut expected, shift);
            }
            let mut actual = WideCyclotomicRing::<F::Wide, D>::from_ring(&initial);
            windows.accumulate_shifts_into(&mut actual, &shifts);
            assert_eq!(
                actual.reduce::<F>(),
                expected.reduce::<F>(),
                "D={D} batch={batch}"
            );
        }
    }
}

#[test]
fn shift_windows_match_wide_accumulation() {
    check_shift_windows_match_wide_accumulation::<F32, 64>(0x51);
    check_shift_windows_match_wide_accumulation::<F32, 512>(0x52);
    check_shift_windows_match_wide_accumulation::<F64, 64>(0x53);
    check_shift_windows_match_wide_accumulation::<F64, 512>(0x54);
    check_shift_windows_match_wide_accumulation::<F64Full, 64>(0x59);
    check_shift_windows_match_wide_accumulation::<F64Full, 512>(0x5a);
    check_shift_windows_match_wide_accumulation::<F128, 64>(0x55);
    check_shift_windows_match_wide_accumulation::<F128, 128>(0x56);
    check_shift_windows_match_wide_accumulation::<F128, 256>(0x57);
    check_shift_windows_match_wide_accumulation::<F128, 512>(0x58);
}

fn check_shift_windows_stay_exact_at_commit_budget<F>()
where
    F: Field + WithCommitAccumulator,
{
    // `-1` has the largest canonical lanes; shift 0 reads it through the
    // positive half and shift `D - 1` of `1` reads it through the negative half.
    let budget = <F as WithCommitAccumulator>::MAX_COMMIT_ACCUMULATIONS;
    for (value, shift) in [(-F::one(), 0), (F::one(), D - 1)] {
        let src = CyclotomicRing::<F, D>::from_coefficients([value; D]);
        let mut windows = NegacyclicShiftWindows::<F, D>::default();
        windows.load(&src);
        let mut wide = WideCyclotomicRing::<F::Wide, D>::zero();
        windows.accumulate_shifts_into(&mut wide, &vec![shift; budget]);

        let expected = src
            .negacyclic_shift(shift)
            .scale(&F::from_u64(budget as u64));
        assert_eq!(wide.reduce::<F>(), expected, "shift={shift}");
    }
}

#[test]
fn shift_windows_stay_exact_at_commit_budget() {
    check_shift_windows_stay_exact_at_commit_budget::<F32>();
    check_shift_windows_stay_exact_at_commit_budget::<F64Full>();
    check_shift_windows_stay_exact_at_commit_budget::<F128>();
}

#[test]
#[should_panic(expected = "out of range")]
fn shift_windows_reject_shift_of_d() {
    let src = CyclotomicRing::<F32, D>::one();
    let mut windows = NegacyclicShiftWindows::<F32, D>::default();
    windows.load(&src);
    let mut wide = WideCyclotomicRing::<Fp32x2i32, D>::zero();
    windows.accumulate_shifts_into(&mut wide, &[D]);
}

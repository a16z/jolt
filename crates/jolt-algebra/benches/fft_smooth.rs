#![expect(clippy::unwrap_used, reason = "bench setup on fixed valid domains")]

use jolt_algebra::fft::SmoothFftField;
use jolt_algebra::fft::{field_pow, primitive_nth_root, rs_extend_fft, SmoothDomain};
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_algebra::Prime128OffsetA7F7;
use jolt_algebra::{Field, One, Zero};
use rand::{rngs::StdRng, SeedableRng};

#[cfg(feature = "parallel")]
use jolt_algebra::Ring;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::fmt::Debug;

type F = Prime128OffsetA7F7;

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_forward");

    // Prime128OffsetA7F7: smooth subgroup `17_496 = 2³·3⁷`.
    for &n in &[243, 729, 1458, 2187, 4374, 8748, 17496] {
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega = primitive_nth_root::<F>(n).unwrap();
        let domain = SmoothDomain::new(omega, n).unwrap();
        let mut rng = StdRng::seed_from_u64(0xff00 + n as u64);
        let input: Vec<F> = (0..n).map(|_| Field::random(&mut rng)).collect();

        let label = format!("A7F7/N={n}");
        let _ = group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| black_box(domain.forward(black_box(&input)).unwrap()));
        });
    }

    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_inverse");

    for &n in &[243, 729, 1458, 2187, 4374, 8748, 17496] {
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega = primitive_nth_root::<F>(n).unwrap();
        let domain = SmoothDomain::new(omega, n).unwrap();
        let mut rng = StdRng::seed_from_u64(0xff01 + n as u64);
        let input: Vec<F> = (0..n).map(|_| Field::random(&mut rng)).collect();
        let transformed = domain.forward(&input).unwrap();

        let _ = group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(domain.inverse(black_box(&transformed)).unwrap()));
        });
    }

    group.finish();
}

fn bench_rs_extend(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_rs_extend");

    for &(k, blowup) in &[(243, 3), (243, 6), (729, 3), (1458, 3)] {
        let n = k * blowup;
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega_n = primitive_nth_root::<F>(n).unwrap();
        let omega_k = field_pow(omega_n, blowup as u64);
        let domain_k = SmoothDomain::new(omega_k, k).unwrap();
        let mut rng = StdRng::seed_from_u64(0xff02 + k as u64);
        let evals: Vec<F> = (0..k).map(|_| Field::random(&mut rng)).collect();

        let label = format!("{k}x{blowup}");
        let _ = group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| {
                black_box(rs_extend_fft(black_box(&evals), &domain_k, omega_n, blowup).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_rs_expand_256_to_1024(c: &mut Criterion) {
    let domain_size = 1458usize;
    let k = 256usize;
    let omega = primitive_nth_root::<F>(domain_size).unwrap();
    let domain = SmoothDomain::new(omega, domain_size).unwrap();

    let mut rng = StdRng::seed_from_u64(0xff03);
    let base_evals: Vec<F> = (0..k).map(|_| Field::random(&mut rng)).collect();

    // Zero-pad the 256 evaluations to the 1458-point domain and IFFT to
    // get a coefficient vector. This is a synthetic benchmark payload, not
    // a true RS interpolation (the zero-padding does not correspond to
    // evaluations at the remaining domain points).
    let mut padded_evals = vec![F::zero(); domain_size];
    padded_evals[..k].copy_from_slice(&base_evals);
    let coeffs = domain.inverse(&padded_evals).unwrap();

    let _ = c.bench_function("fft_rs_expand/256_to_1024_via_1458", |b| {
        b.iter(|| {
            let all_evals = domain.coset_forward(black_box(&coeffs), F::one()).unwrap();
            let _ = black_box(&all_evals[k..k + 1024]);
            all_evals
        });
    });
}

#[cfg(feature = "parallel")]
fn run_rs_expand_256_par<Fld>(
    c: &mut Criterion,
    label: &str,
    omega: Fld,
    domain_size: usize,
    seed_tag: u64,
) where
    Fld: Field + Ring + Debug + Send + Sync,
{
    let k = 256usize;
    let count = 32_768usize;
    let domain = SmoothDomain::new(omega, domain_size).unwrap();

    let coeffs_batch: Vec<Vec<Fld>> = (0..count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed_tag.wrapping_add(i as u64));
            let base: Vec<Fld> = (0..k).map(|_| Field::random(&mut rng)).collect();
            let mut padded = vec![Fld::zero(); domain_size];
            padded[..k].copy_from_slice(&base);
            domain.inverse(&padded).unwrap()
        })
        .collect();

    let _ = c.bench_function(label, |b| {
        b.iter(|| {
            let results: Vec<Vec<Fld>> = coeffs_batch
                .par_iter()
                .map(|coeffs| domain.coset_forward(coeffs, Fld::one()).unwrap())
                .collect();
            let _ = black_box(&results);
        });
    });
}

#[cfg(feature = "parallel")]
fn bench_rs_expand_256_to_1024_par(c: &mut Criterion) {
    // Prime128OffsetA7F7: smooth subgroup of order `2³·3⁷ = 17_496`.
    // There is no 1470-divisor; the closest analogue to `2·3·5·7² = 1470`
    // that still covers the 256+1024 RS-extend budget is `1458 = 2·3⁶`.
    {
        let domain_size = 1458usize;
        let omega = primitive_nth_root::<F>(domain_size).unwrap();
        run_rs_expand_256_par::<F>(
            c,
            "fft_rs_expand/256_to_1024_via_1458_x32768_par_A7F7",
            omega,
            domain_size,
            0xff05,
        );
    }
}

#[cfg(not(feature = "parallel"))]
fn bench_rs_expand_256_to_1024_par(_c: &mut Criterion) {}

criterion_group!(
    fft_smooth,
    bench_forward,
    bench_inverse,
    bench_rs_extend,
    bench_rs_expand_256_to_1024,
    bench_rs_expand_256_to_1024_par,
);
criterion_main!(fft_smooth);

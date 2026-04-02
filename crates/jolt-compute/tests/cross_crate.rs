//! Cross-crate integration tests for jolt-compute.
//!
//! Verifies that CpuBackend primitives agree with the canonical implementations
//! in jolt-poly (Polynomial::bind, EqPolynomial::evaluations).

use jolt_compiler::kernel_spec::Iteration;
use jolt_compute::{BindingOrder, Buf, ComputeBackend, DeviceBuffer};
use jolt_cpu::{CpuBackend, CpuKernel};
use jolt_field::{Field, Fr};
use jolt_poly::{EqPolynomial, Polynomial};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn backend() -> CpuBackend {
    CpuBackend
}

// interpolate_inplace ↔ Polynomial::bind

/// CpuBackend::interpolate_inplace(HighToLow) matches Polynomial::bind.
///
/// Both use split-half layout: `evals[i]` paired with `evals[i + half]`.
/// `interpolate_inplace` is in-place, so we clone before uploading.
#[test]
fn interpolate_matches_polynomial_bind() {
    let mut rng = ChaCha20Rng::seed_from_u64(1000);
    let b = backend();

    for nv in 2..=6 {
        let poly = Polynomial::<Fr>::random(nv, &mut rng);
        let scalar = Fr::random(&mut rng);

        // Via jolt-poly
        let mut poly_bound = poly.clone();
        poly_bound.bind(scalar);

        // Via jolt-compute: clone data, upload, interpolate in-place, download
        let mut buf = b.upload(poly.evaluations());
        b.interpolate_inplace(&mut buf, scalar, BindingOrder::HighToLow);
        let result = b.download(&buf);

        assert_eq!(
            result.as_slice(),
            poly_bound.evaluations(),
            "nv={nv}: interpolate_inplace(HighToLow) must match Polynomial::bind"
        );
    }
}

/// Repeated interpolation (binding all variables) converges to evaluate.
///
/// Polynomial::bind uses HighToLow (split-half) layout, so we use
/// `interpolate_inplace(HighToLow)` at each round to get matching results.
#[test]
fn iterated_interpolation_converges_to_evaluate() {
    let mut rng = ChaCha20Rng::seed_from_u64(1001);
    let b = backend();
    let nv = 5;
    let poly = Polynomial::<Fr>::random(nv, &mut rng);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let expected = poly.evaluate(&point);

    let mut buf = b.upload(poly.evaluations());
    for &r in &point {
        b.interpolate_inplace(&mut buf, r, BindingOrder::HighToLow);
    }
    assert_eq!(buf.len(), 1);
    let result = b.download(&buf);
    assert_eq!(result[0], expected);
}

// eq_table ↔ EqPolynomial::evaluations

/// CpuBackend::eq_table matches EqPolynomial::evaluations.
#[test]
fn eq_table_matches_eq_polynomial() {
    let mut rng = ChaCha20Rng::seed_from_u64(2000);
    let b = backend();

    for nv in 1..=6 {
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

        let expected = EqPolynomial::new(point.clone()).evaluations();
        let table_buf = b.eq_table(&point);
        let actual = b.download(&table_buf);

        assert_eq!(
            actual, expected,
            "nv={nv}: eq_table must match EqPolynomial"
        );
    }
}

// reduce with kernel

/// Identity kernel via reduce: f(lo, hi) at t evaluates
/// `sum_k interp(lo[k], hi[k], t)` for each input buffer.
///
/// Uses `CpuKernel::new` with `Iteration::Dense` and `BindingOrder::LowToHigh`.
/// The dense reduce is unit-weighted (no separate weight buffer).
#[test]
fn reduce_identity_kernel() {
    let b = backend();
    let num_evals = 2; // degree-1: evals at t=0, t=1

    let kernel = CpuKernel::new(
        move |lo: &[Fr], hi: &[Fr], _challenges: &[Fr], out: &mut [Fr]| {
            for (t, slot) in out.iter_mut().enumerate() {
                let t_f = Fr::from_u64(t as u64);
                *slot = (0..lo.len()).map(|k| lo[k] + t_f * (hi[k] - lo[k])).sum();
            }
        },
        num_evals,
        Iteration::Dense,
        BindingOrder::LowToHigh,
    );

    // Two buffers with 4 pairs each (LowToHigh: pairs at [2i, 2i+1])
    let buf_a: Buf<CpuBackend, Fr> = DeviceBuffer::Field(vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
        Fr::from_u64(5),
        Fr::from_u64(6),
        Fr::from_u64(7),
        Fr::from_u64(8),
    ]);
    let buf_b: Buf<CpuBackend, Fr> = DeviceBuffer::Field(vec![
        Fr::from_u64(10),
        Fr::from_u64(20),
        Fr::from_u64(30),
        Fr::from_u64(40),
        Fr::from_u64(50),
        Fr::from_u64(60),
        Fr::from_u64(70),
        Fr::from_u64(80),
    ]);

    let result = b.reduce(&kernel, &[&buf_a, &buf_b], &[]);

    // LowToHigh pairs: (buf[2i], buf[2i+1])
    // t=0: sum of lo values across both buffers
    //   pair 0: lo_a=1, lo_b=10; pair 1: lo_a=3, lo_b=30;
    //   pair 2: lo_a=5, lo_b=50; pair 3: lo_a=7, lo_b=70
    //   sum = (1+10) + (3+30) + (5+50) + (7+70) = 176
    let t0: Fr = [1 + 10, 3 + 30, 5 + 50, 7 + 70]
        .iter()
        .map(|&x| Fr::from_u64(x))
        .sum();
    // t=1: sum of hi values across both buffers
    //   pair 0: hi_a=2, hi_b=20; pair 1: hi_a=4, hi_b=40;
    //   pair 2: hi_a=6, hi_b=60; pair 3: hi_a=8, hi_b=80
    //   sum = (2+20) + (4+40) + (6+60) + (8+80) = 220
    let t1: Fr = [2 + 20, 4 + 40, 6 + 60, 8 + 80]
        .iter()
        .map(|&x| Fr::from_u64(x))
        .sum();

    assert_eq!(result.len(), 2);
    assert_eq!(result[0], t0, "t=0 mismatch");
    assert_eq!(result[1], t1, "t=1 mismatch");
}

// Full sumcheck-style round

/// One sumcheck round: compute eq table, verify s(0)+s(1)=sum, bind, repeat.
#[test]
fn sumcheck_round_invariant() {
    let mut rng = ChaCha20Rng::seed_from_u64(6000);
    let nv = 4;

    let poly = Polynomial::<Fr>::random(nv, &mut rng);
    let tau: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    // Initial sum: sum_{x in {0,1}^nv} eq(x, tau) * f(x) = f(tau)
    let eq_evals = EqPolynomial::new(tau.clone()).evaluations();
    let mut running_sum: Fr = poly
        .evaluations()
        .iter()
        .zip(eq_evals.iter())
        .map(|(a, b)| *a * *b)
        .sum();

    // Verify via direct evaluation
    assert_eq!(
        running_sum,
        poly.evaluate(&tau),
        "initial sum must equal f(tau)"
    );

    let mut eq_poly = Polynomial::new(eq_evals);
    let mut f_poly = poly;

    for round in 0..nv {
        let half = eq_poly.len() / 2;

        // s(0) = sum over x_0=0 half, s(1) = sum over x_0=1 half
        let mut s0 = Fr::from_u64(0);
        let mut s1 = Fr::from_u64(0);
        for i in 0..half {
            s0 += eq_poly.evaluations()[i] * f_poly.evaluations()[i];
            s1 += eq_poly.evaluations()[i + half] * f_poly.evaluations()[i + half];
        }

        assert_eq!(
            s0 + s1,
            running_sum,
            "round {round}: s(0) + s(1) must equal running sum"
        );

        // Bind at a random challenge
        let r = Fr::random(&mut rng);
        eq_poly.bind(r);
        f_poly.bind(r);

        // New running sum: inner product after binding
        running_sum = eq_poly
            .evaluations()
            .iter()
            .zip(f_poly.evaluations().iter())
            .map(|(a, b)| *a * *b)
            .sum();
    }

    // After all rounds, both polynomials are single values
    assert_eq!(eq_poly.len(), 1);
    assert_eq!(f_poly.len(), 1);
}

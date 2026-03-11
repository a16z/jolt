//! Cross-crate integration tests for jolt-compute.
//!
//! Verifies that CpuBackend primitives agree with the canonical implementations
//! in jolt-poly (Polynomial::bind, EqPolynomial::evaluations).

use jolt_compute::{BindingOrder, ComputeBackend};
use jolt_cpu::{CpuBackend, CpuKernel};
use jolt_field::{Field, Fr};
use jolt_poly::{EqPolynomial, Polynomial};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn backend() -> CpuBackend {
    CpuBackend
}

// interpolate_pairs ↔ Polynomial::bind

/// CpuBackend::interpolate_pairs matches Polynomial::bind for field elements.
///
/// Polynomial::bind uses split-half layout `[lo_half | hi_half]` while
/// CpuBackend::interpolate_pairs uses interleaved `[lo0, hi0, lo1, hi1, ...]`.
/// We interleave before uploading.
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

        // Interleave: split-half → paired layout
        let data = poly.evaluations();
        let half = data.len() / 2;
        let mut interleaved = Vec::with_capacity(data.len());
        for i in 0..half {
            interleaved.push(data[i]);
            interleaved.push(data[i + half]);
        }

        // Via jolt-compute
        let buf = b.upload(&interleaved);
        let result_buf: Vec<Fr> = b.interpolate_pairs(buf, scalar);
        let result = b.download(&result_buf);

        assert_eq!(
            result.as_slice(),
            poly_bound.evaluations(),
            "nv={nv}: interpolate_pairs must match Polynomial::bind"
        );
    }
}

/// Repeated interpolation (binding all variables) converges to evaluate.
///
/// Polynomial::bind uses split-half layout, so we must interleave before
/// each `interpolate_pairs` call to get the same pairing.
#[test]
fn iterated_interpolation_converges_to_evaluate() {
    let mut rng = ChaCha20Rng::seed_from_u64(1001);
    let b = backend();
    let nv = 5;
    let poly = Polynomial::<Fr>::random(nv, &mut rng);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let expected = poly.evaluate(&point);

    let mut data: Vec<Fr> = poly.evaluations().to_vec();
    for &r in &point {
        let half = data.len() / 2;
        let mut interleaved = Vec::with_capacity(data.len());
        for i in 0..half {
            interleaved.push(data[i]);
            interleaved.push(data[i + half]);
        }
        let buf = b.upload(&interleaved);
        data = b.interpolate_pairs(buf, r);
    }
    assert_eq!(data.len(), 1);
    assert_eq!(data[0], expected);
}

// product_table ↔ EqPolynomial::evaluations

/// CpuBackend::product_table matches EqPolynomial::evaluations.
#[test]
fn product_table_matches_eq_polynomial() {
    let mut rng = ChaCha20Rng::seed_from_u64(2000);
    let b = backend();

    for nv in 1..=6 {
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

        let expected = EqPolynomial::new(point.clone()).evaluations();
        let table_buf = b.product_table(&point);
        let actual = b.download(&table_buf);

        assert_eq!(
            actual, expected,
            "nv={nv}: product_table must match EqPolynomial"
        );
    }
}

// pairwise_reduce with kernel

/// Identity kernel: f(t) = sum_k (lo[k] + t*(hi[k] - lo[k])) * weight[k].
/// At t=0 and t=1 this gives the weighted sum of even/odd halves.
#[test]
fn pairwise_reduce_identity_kernel() {
    let b = backend();
    let num_evals = 2; // degree-1 polynomial: evals at t=0, t=1

    let kernel = CpuKernel::new(move |lo: &[Fr], hi: &[Fr], out: &mut [Fr]| {
        for (t, slot) in out.iter_mut().enumerate() {
            let t_f = Fr::from_u64(t as u64);
            *slot = (0..lo.len()).map(|k| lo[k] + t_f * (hi[k] - lo[k])).sum();
        }
    });

    // Two buffers with 4 pairs each
    let buf_a = b.upload(&[
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
        Fr::from_u64(5),
        Fr::from_u64(6),
        Fr::from_u64(7),
        Fr::from_u64(8),
    ]);
    let buf_b = b.upload(&[
        Fr::from_u64(10),
        Fr::from_u64(20),
        Fr::from_u64(30),
        Fr::from_u64(40),
        Fr::from_u64(50),
        Fr::from_u64(60),
        Fr::from_u64(70),
        Fr::from_u64(80),
    ]);
    let weights = b.upload(&[
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
    ]);

    let result = b.pairwise_reduce(
        &[&buf_a, &buf_b],
        &weights,
        &kernel,
        num_evals,
        BindingOrder::LowToHigh,
    );

    // t=0: sum over pairs of (lo_a + lo_b) * 1
    // pair 0: lo_a=1, lo_b=10 → 11; pair 1: lo_a=3, lo_b=30 → 33; ...
    let t0: Fr = [1 + 10, 3 + 30, 5 + 50, 7 + 70]
        .iter()
        .map(|&x| Fr::from_u64(x))
        .sum();
    // t=1: sum over pairs of (hi_a + hi_b) * 1
    let t1: Fr = [2 + 20, 4 + 40, 6 + 60, 8 + 80]
        .iter()
        .map(|&x| Fr::from_u64(x))
        .sum();

    assert_eq!(result.len(), 2);
    assert_eq!(result[0], t0, "t=0 mismatch");
    assert_eq!(result[1], t1, "t=1 mismatch");
}

// Batch interpolation consistency

/// interpolate_pairs_batch matches individual calls.
#[test]
fn batch_interpolate_matches_individual() {
    let mut rng = ChaCha20Rng::seed_from_u64(5000);
    let b = backend();
    let nv = 4;
    let scalar = Fr::random(&mut rng);
    let num_bufs = 5;

    let polys: Vec<Polynomial<Fr>> = (0..num_bufs)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();

    // Individual
    let individual: Vec<Vec<Fr>> = polys
        .iter()
        .map(|p| {
            let buf = b.upload(p.evaluations());
            let r: Vec<Fr> = b.interpolate_pairs(buf, scalar);
            b.download(&r)
        })
        .collect();

    // Batch
    let bufs: Vec<Vec<Fr>> = polys.iter().map(|p| b.upload(p.evaluations())).collect();
    let batch_results = b.interpolate_pairs_batch(bufs, scalar);
    let batch: Vec<Vec<Fr>> = batch_results.iter().map(|buf| b.download(buf)).collect();

    assert_eq!(individual, batch);
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

    // Run multiple sumcheck rounds, checking the invariant each time
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

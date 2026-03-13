//! Cross-type integration tests for jolt-poly.
//!
//! These tests verify composition patterns between polynomial types
//! (Polynomial, EqPolynomial, UnivariatePoly, IdentityPolynomial, RlcSource)
//! that are used throughout the proving system.

use jolt_field::{Field, Fr};
use jolt_poly::{
    EqPolynomial, IdentityPolynomial, MultilinearPoly, Polynomial, RlcSource, UnivariatePoly,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

// Polynomial ↔ EqPolynomial: the fundamental MLE identity

/// ⟨f, eq(·, r)⟩ = f̃(r) for any multilinear f and point r.
#[test]
fn inner_product_with_eq_is_evaluation() {
    let mut rng = ChaCha20Rng::seed_from_u64(1000);
    for nv in 1..=6 {
        let poly = Polynomial::<Fr>::random(nv, &mut rng);
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

        let expected = poly.evaluate(&point);

        let eq_evals = EqPolynomial::new(point).evaluations();
        let inner: Fr = poly
            .evaluations()
            .iter()
            .zip(eq_evals.iter())
            .map(|(a, b)| *a * *b)
            .sum();

        assert_eq!(inner, expected, "nv={nv}: inner product ≠ evaluate");
    }
}

// Sequential binding converges to evaluate

/// Binding all variables one-by-one yields the same result as evaluate.
#[test]
fn sequential_bind_equals_evaluate() {
    let mut rng = ChaCha20Rng::seed_from_u64(2000);
    for nv in 1..=5 {
        let poly = Polynomial::<Fr>::random(nv, &mut rng);
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

        let expected = poly.evaluate(&point);

        let mut working = poly.clone();
        for &r in &point {
            working.bind(r);
        }
        assert_eq!(working.len(), 1);
        assert_eq!(working.evaluations()[0], expected, "nv={nv}");
    }
}

// Compact polynomial promotion

/// Polynomial<u8>::bind_to_field agrees with Polynomial<Fr> built from the same data.
#[test]
fn compact_u8_bind_matches_field_bind() {
    let mut rng = ChaCha20Rng::seed_from_u64(3000);
    let nv = 4;
    let data: Vec<u8> = (0..1 << nv).map(|i| (i * 37 + 13) as u8).collect();
    let scalar = Fr::random(&mut rng);

    let compact = Polynomial::new(data.clone());
    let promoted = compact.bind_to_field::<Fr>(scalar);

    let field_poly: Polynomial<Fr> =
        Polynomial::new(data.iter().map(|&x| Fr::from_u64(x as u64)).collect());
    let mut expected = field_poly;
    expected.bind(scalar);

    assert_eq!(
        promoted.evaluations(),
        expected.evaluations(),
        "compact bind_to_field must match field bind"
    );
}

// UnivariatePoly interpolation

/// Lagrange interpolation recovers the original polynomial at domain points.
#[test]
fn univariate_interpolation_recovers_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(4000);
    let degree = 5;
    let points: Vec<(Fr, Fr)> = (0..=degree)
        .map(|i| (Fr::from_u64(i as u64), Fr::random(&mut rng)))
        .collect();

    let poly = UnivariatePoly::interpolate(&points);

    for (x, y) in &points {
        let eval = poly.evaluate(*x);
        assert_eq!(eval, *y, "interpolation must recover point at x={x:?}");
    }
}

/// Interpolation over integers matches evaluate at integer domain.
#[test]
fn univariate_interpolation_over_integers() {
    let evals = vec![
        Fr::from_u64(1),
        Fr::from_u64(4),
        Fr::from_u64(9),
        Fr::from_u64(16),
    ];
    let poly = UnivariatePoly::interpolate_over_integers(&evals);

    for (i, expected) in evals.iter().enumerate() {
        let eval = poly.evaluate(Fr::from_u64(i as u64));
        assert_eq!(&eval, expected, "mismatch at domain point {i}");
    }
}

// CompressedPoly round-trip

/// compress → decompress preserves the polynomial.
#[test]
fn compressed_round_trip() {
    let mut rng = ChaCha20Rng::seed_from_u64(5000);
    let coeffs: Vec<Fr> = (0..5).map(|_| Fr::random(&mut rng)).collect();
    let original = UnivariatePoly::new(coeffs);
    let hint = original.evaluate(Fr::from_u64(0)) + original.evaluate(Fr::from_u64(1));

    let compressed = original.compress();
    let recovered = compressed.decompress(hint);

    // Check evaluation at several points
    for i in 0..10 {
        let x = Fr::from_u64(i);
        assert_eq!(
            original.evaluate(x),
            recovered.evaluate(x),
            "compress/decompress mismatch at x={i}"
        );
    }
}

/// CompressedPoly::evaluate_with_hint matches the original polynomial.
#[test]
fn compressed_evaluate_with_hint() {
    let mut rng = ChaCha20Rng::seed_from_u64(5001);
    let coeffs: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let poly = UnivariatePoly::new(coeffs);
    let hint = poly.evaluate(Fr::from_u64(0)) + poly.evaluate(Fr::from_u64(1));
    let compressed = poly.compress();

    for i in 0..8 {
        let x = Fr::from_u64(i);
        assert_eq!(
            poly.evaluate(x),
            compressed.evaluate_with_hint(hint, x),
            "evaluate_with_hint mismatch at x={i}"
        );
    }
}

// IdentityPolynomial

/// IdentityPolynomial maps Boolean hypercube points to their integer index.
#[test]
fn identity_polynomial_boolean_indexing() {
    let nv = 4;
    let id = IdentityPolynomial::new(nv);

    for idx in 0..(1 << nv) {
        let bits: Vec<Fr> = (0..nv)
            .map(|j| {
                if (idx >> (nv - 1 - j)) & 1 == 1 {
                    Fr::from_u64(1)
                } else {
                    Fr::from_u64(0)
                }
            })
            .collect();
        let eval = id.evaluate::<Fr>(&bits);
        assert_eq!(eval, Fr::from_u64(idx as u64), "identity at index {idx}");
    }
}

/// IdentityPolynomial at a random point matches manual computation.
#[test]
fn identity_polynomial_random_point() {
    let mut rng = ChaCha20Rng::seed_from_u64(6000);
    let nv = 5;
    let id = IdentityPolynomial::new(nv);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let eval = id.evaluate::<Fr>(&point);

    // Manual: sum_i r_i * 2^(n-1-i)
    let expected: Fr = point
        .iter()
        .enumerate()
        .map(|(i, r)| *r * Fr::from_u64(1u64 << (nv - 1 - i)))
        .sum();

    assert_eq!(eval, expected);
}

// RlcSource: lazy random linear combination

/// RlcSource evaluation matches materializing and linearly combining.
#[test]
fn rlc_source_matches_materialized_combination() {
    let mut rng = ChaCha20Rng::seed_from_u64(7000);
    let nv = 3;
    let num_polys = 4;

    let polys: Vec<Polynomial<Fr>> = (0..num_polys)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let scalars: Vec<Fr> = (0..num_polys).map(|_| Fr::random(&mut rng)).collect();
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    // Materialized: sum_i scalar_i * poly_i.evaluate(point)
    let expected: Fr = polys
        .iter()
        .zip(scalars.iter())
        .map(|(p, s)| *s * p.evaluate(&point))
        .sum();

    // Lazy via RlcSource
    let rlc = RlcSource::new(polys, scalars);
    let actual = rlc.evaluate(&point);

    assert_eq!(actual, expected);
}

// Polynomial arithmetic

/// Addition is commutative: a + b == b + a.
#[test]
fn polynomial_addition_commutative() {
    let mut rng = ChaCha20Rng::seed_from_u64(8000);
    let nv = 4;
    let a = Polynomial::<Fr>::random(nv, &mut rng);
    let b = Polynomial::<Fr>::random(nv, &mut rng);

    let ab = a.clone() + b.clone();
    let ba = b + a;
    assert_eq!(ab.evaluations(), ba.evaluations());
}

/// Scalar multiplication distributes over addition: s*(a+b) == s*a + s*b.
#[test]
fn scalar_mul_distributes_over_addition() {
    let mut rng = ChaCha20Rng::seed_from_u64(8001);
    let nv = 3;
    let a = Polynomial::<Fr>::random(nv, &mut rng);
    let b = Polynomial::<Fr>::random(nv, &mut rng);
    let s = Fr::random(&mut rng);

    let sum_then_scale = (a.clone() + b.clone()) * s;
    let scale_then_sum = a * s + b * s;
    assert_eq!(sum_then_scale.evaluations(), scale_then_sum.evaluations());
}

// Serialization

/// bincode round-trip preserves a Polynomial<Fr>.
#[test]
fn polynomial_bincode_round_trip() {
    let mut rng = ChaCha20Rng::seed_from_u64(9000);
    let nv = 5;
    let poly = Polynomial::<Fr>::random(nv, &mut rng);

    let bytes = bincode::serialize(&poly).expect("serialize");
    let recovered: Polynomial<Fr> = bincode::deserialize(&bytes).expect("deserialize");

    assert_eq!(poly.evaluations(), recovered.evaluations());
    assert_eq!(poly.num_vars(), recovered.num_vars());
}

/// bincode round-trip preserves UnivariatePoly<Fr>.
#[test]
fn univariate_bincode_round_trip() {
    let mut rng = ChaCha20Rng::seed_from_u64(9001);
    let coeffs: Vec<Fr> = (0..6).map(|_| Fr::random(&mut rng)).collect();
    let poly = UnivariatePoly::new(coeffs);

    let bytes = bincode::serialize(&poly).expect("serialize");
    let recovered: UnivariatePoly<Fr> = bincode::deserialize(&bytes).expect("deserialize");

    for i in 0..10 {
        let x = Fr::from_u64(i);
        assert_eq!(poly.evaluate(x), recovered.evaluate(x));
    }
}

#![expect(clippy::expect_used)]
//! Cross-type integration tests for jolt-poly.
//!
//! These tests verify composition patterns between polynomial types
//! (Polynomial, EqPolynomial, UnivariatePoly, IdentityPolynomial, RlcSource)
//! that are used throughout the proving system.

use jolt_field::{Ext2, Field, Fr, One, Prime64Offset59, Ring, Zero};
use jolt_poly::{
    EqPolynomial, IdentityPolynomial, MultilinearEvaluation, MultilinearPoly, NormalizedPoly,
    Polynomial, RlcSource, UnivariatePoly, UnivariatePolynomial,
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

fn check_equispaced_interpolation<F: Field>(coefficients: Vec<F>) {
    let original = UnivariatePoly::new(coefficients);
    let evals: Vec<F> = (0..original.coefficients().len())
        .map(|x| original.evaluate(F::from_u64(x as u64)))
        .collect();
    let recovered = UnivariatePoly::from_evals(&evals);
    assert_eq!(recovered.coefficients(), original.coefficients());
    for x in 0..9 {
        assert_eq!(
            recovered.evaluate(F::from_u64(x)),
            original.evaluate(F::from_u64(x))
        );
    }
}

#[test]
fn equispaced_interpolation_preserves_coefficient_count() {
    for coefficients in [
        vec![],
        vec![Fr::from_u64(7)],
        vec![Fr::from_u64(7), Fr::zero(), Fr::zero()],
        vec![
            Fr::from_u64(7),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::zero(),
        ],
    ] {
        check_equispaced_interpolation(coefficients);
    }

    let mut low_degree = UnivariatePoly::from_evals(&[Fr::from_u64(5); 4]);
    assert_eq!(low_degree.coefficients().len(), 4);
    low_degree.trim_trailing_zeros();
    assert_eq!(low_degree.coefficients(), &[Fr::from_u64(5)]);

    let mut zero = UnivariatePoly::from_evals(&[Fr::zero(); 3]);
    zero.trim_trailing_zeros();
    assert_eq!(zero.coefficients(), &[Fr::zero()]);
    let mut empty = UnivariatePoly::<Fr>::zero();
    empty.trim_trailing_zeros();
    assert!(empty.coefficients().is_empty());
}

#[test]
fn equispaced_interpolation_over_extension_field() {
    type Extension = Ext2<Prime64Offset59>;
    let element = |a, b| Extension::new(Prime64Offset59::from_u64(a), Prime64Offset59::from_u64(b));
    check_equispaced_interpolation(vec![
        element(4, 7),
        element(1, 2),
        element(5, 3),
        Extension::zero(),
    ]);
}

#[test]
fn compression_handles_empty_constant_and_trailing_zeros() {
    for coefficients in [
        vec![],
        vec![Fr::from_u64(8)],
        vec![Fr::from_u64(8), Fr::from_u64(2)],
        vec![Fr::from_u64(8), Fr::from_u64(2), Fr::zero()],
    ] {
        let original = UnivariatePoly::new(coefficients);
        let hint = original.evaluate(Fr::zero()) + original.evaluate(Fr::one());
        let compressed = original.compress();
        assert_eq!(compressed.degree(), original.degree().max(1));
        assert!(!compressed.is_empty());
        assert_eq!(compressed.decompress(hint).compress(), compressed);
        for x in 0..7 {
            let point = Fr::from_u64(x);
            assert_eq!(
                compressed.evaluate_with_hint(hint, point),
                original.evaluate(point)
            );
        }
    }
}

#[test]
fn normalized_payload_preserves_shape_and_evaluates() {
    let empty = NormalizedPoly::<Fr>::new(vec![]);
    assert_eq!(empty.degree(), 0);
    assert_eq!(empty.evaluate_nonconstant_terms(Fr::one()), Fr::zero());
    let normalized = NormalizedPoly::from_q_coefficients(vec![
        Fr::from_u64(11),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::zero(),
    ]);
    assert_eq!(normalized.degree(), 3);
    assert_eq!(normalized.coefficients()[2], Fr::zero());
    assert_eq!(normalized.nonconstant_term_sum_at_one(), Fr::from_u64(5));
    assert_eq!(
        normalized.evaluate_nonconstant_terms(Fr::zero()),
        Fr::zero()
    );
    assert_eq!(
        normalized.evaluate_nonconstant_terms(Fr::one()),
        Fr::from_u64(5)
    );
    assert_eq!(
        normalized.evaluate_nonconstant_terms(Fr::from_u64(2)),
        Fr::from_u64(16)
    );
}

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
        let eval = id.evaluate(&bits);
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

    let eval = id.evaluate(&point);

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
    let b_plus_a = b + a;
    assert_eq!(ab.evaluations(), b_plus_a.evaluations());
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

    let bytes =
        bincode::serde::encode_to_vec(&poly, bincode::config::standard()).expect("serialize");
    let recovered: Polynomial<Fr> =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .expect("deserialize")
            .0;

    assert_eq!(poly.evaluations(), recovered.evaluations());
    assert_eq!(poly.num_vars(), recovered.num_vars());
}

/// bincode round-trip preserves UnivariatePoly<Fr>.
#[test]
fn univariate_bincode_round_trip() {
    let mut rng = ChaCha20Rng::seed_from_u64(9001);
    let coeffs: Vec<Fr> = (0..6).map(|_| Fr::random(&mut rng)).collect();
    let poly = UnivariatePoly::new(coeffs);

    let bytes =
        bincode::serde::encode_to_vec(&poly, bincode::config::standard()).expect("serialize");
    let recovered: UnivariatePoly<Fr> =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .expect("deserialize")
            .0;

    for i in 0..10 {
        let x = Fr::from_u64(i);
        assert_eq!(poly.evaluate(x), recovered.evaluate(x));
    }
}

use jolt_field::{Ext2, Field, Fr, One, Prime64Offset59, Ring, Zero};
use jolt_poly::{
    IdentityPolynomial, MultilinearEvaluation, MultilinearPoly, OmittedConstantPoly, Polynomial,
    RlcSource, UnivariatePoly, UnivariatePolynomial,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

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
fn omitted_constant_payload_preserves_shape_and_evaluates() {
    let empty = OmittedConstantPoly::<Fr>::new(vec![]);
    assert_eq!(empty.degree(), 0);
    assert_eq!(empty.evaluate_nonconstant_terms(Fr::one()), Fr::zero());
    let normalized = OmittedConstantPoly::from_q_coefficients(vec![
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

#[test]
fn identity_polynomial_random_point() {
    let mut rng = ChaCha20Rng::seed_from_u64(6000);
    let nv = 5;
    let id = IdentityPolynomial::new(nv);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let eval = id.evaluate(&point);

    let expected: Fr = point
        .iter()
        .enumerate()
        .map(|(i, r)| *r * Fr::from_u64(1u64 << (nv - 1 - i)))
        .sum();

    assert_eq!(eval, expected);
}

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

    let expected: Fr = polys
        .iter()
        .zip(scalars.iter())
        .map(|(p, s)| *s * p.evaluate(&point))
        .sum();

    let rlc = RlcSource::new(polys, scalars);
    let actual = rlc.evaluate(&point);

    assert_eq!(actual, expected);
}

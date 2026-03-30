use jolt_field::Fr;
use jolt_ir::{Expr, ExprBuilder};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

fn assert_formula_consistent(expr: &Expr, openings: &[Fr], challenges: &[Fr]) {
    let direct = expr.evaluate(openings, challenges);
    let formula = expr.to_composition_formula();
    let via_formula = formula.evaluate(openings, challenges);
    assert_eq!(direct, via_formula);
}

fn random_vals(rng: &mut ChaCha8Rng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| jolt_field::Field::random(rng)).collect()
}

#[test]
fn distribution_a_plus_b_times_c() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let c = b.opening(2);
    let expr = b.build((a + bv) * c);

    let formula = expr.to_composition_formula();
    assert_eq!(formula.len(), 2);

    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..100 {
        assert_formula_consistent(&expr, &random_vals(&mut rng, 3), &[]);
    }
}

#[test]
fn distribution_a_plus_b_times_c_plus_d() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let c = b.opening(2);
    let d = b.opening(3);
    let expr = b.build((a + bv) * (c + d));

    let formula = expr.to_composition_formula();
    assert_eq!(formula.len(), 4);

    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..100 {
        assert_formula_consistent(&expr, &random_vals(&mut rng, 4), &[]);
    }
}

#[test]
fn subtraction_a_minus_b_times_c() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let c = b.opening(2);
    let expr = b.build((a - bv) * c);

    let formula = expr.to_composition_formula();
    assert_eq!(formula.len(), 2);
    assert_eq!(formula.terms[1].coefficient, -1);

    let mut rng = ChaCha8Rng::seed_from_u64(3);
    for _ in 0..100 {
        assert_formula_consistent(&expr, &random_vals(&mut rng, 3), &[]);
    }
}

#[test]
fn negation_of_product() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let expr = b.build(-(a * bv));

    let formula = expr.to_composition_formula();
    assert_eq!(formula.len(), 1);
    assert_eq!(formula.terms[0].coefficient, -1);

    let mut rng = ChaCha8Rng::seed_from_u64(4);
    for _ in 0..100 {
        assert_formula_consistent(&expr, &random_vals(&mut rng, 2), &[]);
    }
}

#[test]
fn booleanity_formula() {
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let gamma = b.challenge(0);
    let expr = b.build(gamma * (h * h - h));

    let formula = expr.to_composition_formula();
    assert_eq!(formula.len(), 2);

    let mut rng = ChaCha8Rng::seed_from_u64(5);
    for _ in 0..100 {
        assert_formula_consistent(&expr, &random_vals(&mut rng, 1), &random_vals(&mut rng, 1));
    }
}

#[test]
fn deeply_nested_distribution() {
    // ((a + b) * (c + d)) * (e - f) -> 8 terms
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let c = b.opening(2);
    let d = b.opening(3);
    let e = b.opening(4);
    let f = b.opening(5);
    let expr = b.build((a + bv) * (c + d) * (e - f));

    let formula = expr.to_composition_formula();
    assert_eq!(formula.len(), 8);

    let mut rng = ChaCha8Rng::seed_from_u64(6);
    for _ in 0..100 {
        assert_formula_consistent(&expr, &random_vals(&mut rng, 6), &[]);
    }
}

#[test]
fn weighted_sum_with_challenges() {
    // alpha*a + beta*b + gamma*c
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let c = b.opening(2);
    let alpha = b.challenge(0);
    let beta = b.challenge(1);
    let gamma = b.challenge(2);
    let expr = b.build(alpha * a + beta * bv + gamma * c);

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for _ in 0..100 {
        assert_formula_consistent(&expr, &random_vals(&mut rng, 3), &random_vals(&mut rng, 3));
    }
}

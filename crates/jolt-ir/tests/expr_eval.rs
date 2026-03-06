use jolt_field::{Field, Fr};
use jolt_ir::ExprBuilder;

#[test]
fn all_binary_ops() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);

    let a_val = Fr::from_u64(10);
    let b_val = Fr::from_u64(3);

    let add: Fr = b.build(a + bv).evaluate(&[a_val, b_val], &[]);
    assert_eq!(add, Fr::from_u64(13));

    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let sub: Fr = b.build(a - bv).evaluate(&[a_val, b_val], &[]);
    assert_eq!(sub, Fr::from_u64(7));

    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let mul: Fr = b.build(a * bv).evaluate(&[a_val, b_val], &[]);
    assert_eq!(mul, Fr::from_u64(30));

    let b = ExprBuilder::new();
    let a = b.opening(0);
    let neg: Fr = b.build(-a).evaluate(&[a_val], &[]);
    assert_eq!(neg, -a_val);
}

#[test]
fn nested_arithmetic() {
    // ((a + b) * c) - (d * e) with a=2, b=3, c=4, d=5, e=6
    // = (5 * 4) - (5 * 6) = 20 - 30 = -10
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let c = b.opening(2);
    let d = b.opening(3);
    let e = b.opening(4);
    let expr = b.build((a + bv) * c - d * e);

    let vals: Vec<Fr> = (2..=6).map(Fr::from_u64).collect();
    let result: Fr = expr.evaluate(&vals, &[]);
    assert_eq!(result, Fr::from_u64(20) - Fr::from_u64(30));
}

#[test]
fn integer_literal_operations() {
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let h_val = Fr::from_u64(10);

    // h * 2 = 20
    let result: Fr = b.build(h * 2).evaluate(&[h_val], &[]);
    assert_eq!(result, Fr::from_u64(20));

    // 3 * h = 30
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let result: Fr = b.build(3i128 * h).evaluate(&[h_val], &[]);
    assert_eq!(result, Fr::from_u64(30));

    // h + 1 = 11
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let result: Fr = b.build(h + 1).evaluate(&[h_val], &[]);
    assert_eq!(result, Fr::from_u64(11));

    // h - 5 = 5
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let result: Fr = b.build(h - 5).evaluate(&[h_val], &[]);
    assert_eq!(result, Fr::from_u64(5));
}

#[test]
fn constant_promotion() {
    // Negative constant: -1
    let b = ExprBuilder::new();
    let expr = b.build(b.constant(-1));
    let result: Fr = expr.evaluate(&[], &[]);
    assert_eq!(result, -Fr::from_u64(1));

    // Large constant
    let b = ExprBuilder::new();
    let expr = b.build(b.constant(1_000_000_000));
    let result: Fr = expr.evaluate(&[], &[]);
    assert_eq!(result, Fr::from_u64(1_000_000_000));
}

#[test]
fn large_expression() {
    // Build a chain: x0 + x1 + x2 + ... + x99
    let b = ExprBuilder::new();
    let mut acc = b.opening(0);
    for i in 1..100u32 {
        acc = acc + b.opening(i);
    }
    let expr = b.build(acc);

    assert!(expr.len() >= 100);

    // Evaluate: each opening = its index + 1, so sum = 1+2+...+100 = 5050
    let vals: Vec<Fr> = (1..=100).map(Fr::from_u64).collect();
    let result: Fr = expr.evaluate(&vals, &[]);
    assert_eq!(result, Fr::from_u64(5050));
}

#[test]
fn mixed_openings_and_challenges() {
    // alpha * a + beta * b + gamma * c
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let c = b.opening(2);
    let alpha = b.challenge(0);
    let beta = b.challenge(1);
    let gamma = b.challenge(2);
    let expr = b.build(alpha * a + beta * bv + gamma * c);

    let openings = [Fr::from_u64(10), Fr::from_u64(20), Fr::from_u64(30)];
    let challenges = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
    // 2*10 + 3*20 + 5*30 = 20 + 60 + 150 = 230
    let result: Fr = expr.evaluate(&openings, &challenges);
    assert_eq!(result, Fr::from_u64(230));
}

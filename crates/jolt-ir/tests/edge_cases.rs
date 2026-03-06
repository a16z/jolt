use jolt_field::{Field, Fr};
use jolt_ir::ExprBuilder;

#[test]
fn constant_only_expression() {
    let b = ExprBuilder::new();
    let expr = b.build(b.constant(99));
    let result: Fr = expr.evaluate(&[], &[]);
    assert_eq!(result, Fr::from_u64(99));

    let sop = expr.to_sum_of_products();
    assert_eq!(sop.len(), 1);
    assert!(sop.terms[0].factors.is_empty());
}

#[test]
fn single_variable() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let expr = b.build(a);
    let val = Fr::from_u64(42);
    assert_eq!(expr.evaluate::<Fr>(&[val], &[]), val);
}

#[test]
fn same_variable_squared() {
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let expr = b.build(h * h);
    let val = Fr::from_u64(7);
    assert_eq!(expr.evaluate::<Fr>(&[val], &[]), Fr::from_u64(49));

    let sop = expr.to_sum_of_products();
    assert_eq!(sop.len(), 1);
    assert_eq!(sop.terms[0].factors.len(), 2);
}

#[test]
fn challenges_only() {
    let b = ExprBuilder::new();
    let alpha = b.challenge(0);
    let beta = b.challenge(1);
    let expr = b.build(alpha * beta);

    let a = Fr::from_u64(3);
    let bv = Fr::from_u64(7);
    assert_eq!(expr.evaluate::<Fr>(&[], &[a, bv]), Fr::from_u64(21));
}

#[test]
fn openings_only() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let expr = b.build(a + bv);
    assert_eq!(
        expr.evaluate::<Fr>(&[Fr::from_u64(5), Fr::from_u64(3)], &[]),
        Fr::from_u64(8)
    );
}

#[test]
fn cse_preserves_evaluation() {
    // (a + b) * (a + b) — shared subtree
    let b = ExprBuilder::new();
    let a1 = b.opening(0);
    let b1 = b.opening(1);
    let a2 = b.opening(0);
    let b2 = b.opening(1);
    let expr = b.build((a1 + b1) * (a2 + b2));

    let optimized = expr.eliminate_common_subexpressions();
    assert!(optimized.len() <= expr.len());

    let vals = [Fr::from_u64(3), Fr::from_u64(4)];
    let original: Fr = expr.evaluate(&vals, &[]);
    let cse_result: Fr = optimized.evaluate(&vals, &[]);
    assert_eq!(original, cse_result);
    assert_eq!(original, Fr::from_u64(49)); // (3+4)^2
}

#[test]
fn cse_on_nested_repeats() {
    // a*b + a*b — identical subtrees
    let b = ExprBuilder::new();
    let a1 = b.opening(0);
    let b1 = b.opening(1);
    let a2 = b.opening(0);
    let b2 = b.opening(1);
    let expr = b.build(a1 * b1 + a2 * b2);

    let optimized = expr.eliminate_common_subexpressions();
    assert!(optimized.len() < expr.len());

    let vals = [Fr::from_u64(5), Fr::from_u64(6)];
    assert_eq!(
        expr.evaluate::<Fr>(&vals, &[]),
        optimized.evaluate::<Fr>(&vals, &[])
    );
}

#[test]
fn negative_constant() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let expr = b.build(a + b.constant(-100));

    let result: Fr = expr.evaluate(&[Fr::from_u64(200)], &[]);
    assert_eq!(result, Fr::from_u64(100));
}

#[test]
fn zero_times_anything() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let expr = b.build(b.zero() * a);
    assert_eq!(
        expr.evaluate::<Fr>(&[Fr::from_u64(999)], &[]),
        Fr::from_u64(0)
    );
}

#[test]
fn fold_constants_chain() {
    // (2 + 3) * (4 - 1) → should fold to 15
    let b = ExprBuilder::new();
    let two = b.constant(2);
    let three = b.constant(3);
    let four = b.constant(4);
    let one = b.constant(1);
    let expr = b.build((two + three) * (four - one));

    let folded = expr.fold_constants();
    assert_eq!(folded.get(folded.root()), jolt_ir::ExprNode::Constant(15));
}

#[test]
fn fold_constants_preserves_variables() {
    // (2 + 3) * a → Constant(5) * Var(Opening(0))
    let b = ExprBuilder::new();
    let two = b.constant(2);
    let three = b.constant(3);
    let a = b.opening(0);
    let expr = b.build((two + three) * a);

    let folded = expr.fold_constants();
    let val = Fr::from_u64(7);
    assert_eq!(
        expr.evaluate::<Fr>(&[val], &[]),
        folded.evaluate::<Fr>(&[val], &[])
    );
    assert_eq!(folded.evaluate::<Fr>(&[val], &[]), Fr::from_u64(35));
}

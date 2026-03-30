use jolt_field::{Field, Fr};
use jolt_ir::{CompositionFormula, ExprBuilder, R1csVar};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

fn verify_r1cs_roundtrip(
    formula: &CompositionFormula,
    opening_vars: &[R1csVar],
    opening_vals: &[Fr],
    challenge_vals: &[Fr],
) {
    let mut next_var = opening_vars.iter().map(|v| v.0 + 1).max().unwrap_or(1);
    let emission = formula.emit_r1cs::<Fr>(opening_vars, challenge_vals, &mut next_var);

    // Build witness
    let witness_len = next_var as usize;
    let mut witness = vec![Fr::from_u64(0); witness_len];
    witness[0] = Fr::from_u64(1);
    for (var, val) in opening_vars.iter().zip(opening_vals) {
        witness[var.index()] = *val;
    }

    // Forward-evaluate aux vars
    for constraint in &emission.constraints {
        let a_val = constraint.a.evaluate(&witness);
        let b_val = constraint.b.evaluate(&witness);
        assert_eq!(constraint.c.terms.len(), 1);
        witness[constraint.c.terms[0].var.index()] = a_val * b_val;
    }

    // All constraints satisfied
    for (i, constraint) in emission.constraints.iter().enumerate() {
        assert!(constraint.is_satisfied(&witness), "Constraint {i} failed");
    }

    // Output matches direct evaluation
    let expected = formula.evaluate(opening_vals, challenge_vals);
    let actual = witness[emission.output_var.index()];
    assert_eq!(actual, expected, "Output mismatch");
}

#[test]
fn roundtrip_booleanity() {
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let gamma = b.challenge(0);
    let expr = b.build(gamma * (h * h - h));
    let formula = expr.to_composition_formula();

    let opening_vars = [R1csVar(1)];
    let mut rng = ChaCha8Rng::seed_from_u64(0xb001);
    for _ in 0..50 {
        let opening_vals = [Fr::random(&mut rng)];
        let challenge_vals = [Fr::random(&mut rng)];
        verify_r1cs_roundtrip(&formula, &opening_vars, &opening_vals, &challenge_vals);
    }
}

#[test]
fn roundtrip_weighted_sum() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let alpha = b.challenge(0);
    let beta = b.challenge(1);
    let expr = b.build(alpha * a + beta * bv);
    let formula = expr.to_composition_formula();

    let opening_vars = [R1csVar(1), R1csVar(2)];
    let mut rng = ChaCha8Rng::seed_from_u64(0xaced);
    for _ in 0..50 {
        let opening_vals: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
        let challenge_vals: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
        verify_r1cs_roundtrip(&formula, &opening_vars, &opening_vals, &challenge_vals);
    }
}

#[test]
fn roundtrip_foil() {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let c = b.opening(2);
    let d = b.opening(3);
    let expr = b.build((a + bv) * (c - d));
    let formula = expr.to_composition_formula();

    let opening_vars: Vec<R1csVar> = (0..4).map(|i| R1csVar(i + 1)).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(0xf011);
    for _ in 0..50 {
        let opening_vals: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        verify_r1cs_roundtrip(&formula, &opening_vars, &opening_vals, &[]);
    }
}

#[test]
fn roundtrip_constant() {
    let b = ExprBuilder::new();
    let expr = b.build(b.constant(42));
    let formula = expr.to_composition_formula();

    verify_r1cs_roundtrip(&formula, &[], &[], &[]);
}

#[test]
fn roundtrip_complex_ram_style() {
    // eq * ra * val + gamma * eq * ra * inc (RAM-style formula)
    let b = ExprBuilder::new();
    let eq = b.opening(0);
    let ra = b.opening(1);
    let val = b.opening(2);
    let inc = b.opening(3);
    let gamma = b.challenge(0);
    let expr = b.build(eq * ra * val + gamma * eq * ra * inc);
    let formula = expr.to_composition_formula();

    let opening_vars: Vec<R1csVar> = (0..4).map(|i| R1csVar(i + 1)).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(0x4a4d);
    for _ in 0..50 {
        let opening_vals: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        let challenge_vals = [Fr::random(&mut rng)];
        verify_r1cs_roundtrip(&formula, &opening_vars, &opening_vals, &challenge_vals);
    }
}

#![no_main]
use jolt_field::{Field, Fr};
use jolt_ir::{ExprBuilder, R1csVar};
use libfuzzer_sys::fuzz_target;

// Builds a random expression, normalizes to SoP, emits R1CS, then checks that
// all constraints are satisfied and the output matches direct evaluation.
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let b = ExprBuilder::new();
    let num_openings = 3u32;
    let num_challenges = 2u32;

    let mut handles = Vec::with_capacity(16);
    for i in 0..num_openings {
        handles.push(b.opening(i));
    }
    for i in 0..num_challenges {
        handles.push(b.challenge(i));
    }

    for &byte in data {
        if handles.is_empty() {
            break;
        }
        let op = byte % 5;
        let idx = (byte / 5) as usize;

        match op {
            0 => {
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a + bh);
            }
            1 => {
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a - bh);
            }
            2 => {
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a * bh);
            }
            3 => {
                let a = handles[idx % handles.len()];
                handles.push(-a);
            }
            _ => {
                handles.push(b.constant((byte as i128) - 128));
            }
        }

        // Keep expressions small to avoid long R1CS chains
        if handles.len() > 32 {
            break;
        }
    }

    if let Some(&root) = handles.last() {
        let expr = b.build(root);
        let sop = expr.to_sum_of_products();

        // R1CS variable 0 is always "one", openings start at 1
        let opening_vars: Vec<R1csVar> =
            (0..num_openings).map(|i| R1csVar(i + 1)).collect();
        let opening_vals: Vec<Fr> =
            (0..num_openings).map(|i| Fr::from_u64(i as u64 + 5)).collect();
        let challenge_vals: Vec<Fr> =
            (0..num_challenges).map(|i| Fr::from_u64(i as u64 + 50)).collect();

        let mut next_var = num_openings + 1;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

        // Build witness
        let witness_len = next_var as usize;
        let mut witness = vec![Fr::from_u64(0); witness_len];
        witness[0] = Fr::from_u64(1); // "one" variable
        for (var, val) in opening_vars.iter().zip(&opening_vals) {
            witness[var.index()] = *val;
        }

        // Forward-evaluate aux vars
        for constraint in &emission.constraints {
            let a_val = constraint.a.evaluate(&witness);
            let b_val = constraint.b.evaluate(&witness);
            assert_eq!(constraint.c.terms.len(), 1, "output LC must be single var");
            witness[constraint.c.terms[0].var.index()] = a_val * b_val;
        }

        // All constraints must be satisfied
        for (i, constraint) in emission.constraints.iter().enumerate() {
            assert!(
                constraint.is_satisfied(&witness),
                "Constraint {i} unsatisfied"
            );
        }

        // Output must match direct evaluation
        let expected: Fr = sop.evaluate(&opening_vals, &challenge_vals);
        let actual = witness[emission.output_var.index()];
        assert_eq!(actual, expected, "R1CS output mismatch");
    }
});

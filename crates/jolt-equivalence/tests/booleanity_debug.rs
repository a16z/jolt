//! Debug test: verify Booleanity sumcheck round polynomial structure.
//!
//! Confirms that Booleanity round polynomials are NOT all-zero even for
//! boolean RA values. The Gruen decomposition produces non-zero polynomials
//! because the MLE extension at non-boolean evaluation points (X=2,3) is
//! non-zero even when the hypercube evaluations (X=0,1) sum to zero.
//!
//! Stage 6 instance layout (muldiv):
//!   [0] BytecodeReadRaf  (22 rounds, offset=0)
//!   [1] Booleanity       (13 rounds, offset=9)   ← focus of this test
//!   [2] HammingBooleanity (9 rounds, offset=13)
//!   [3] RamRaVirtual      (9 rounds, offset=13)
//!   [4] LookupsRaVirtual  (9 rounds, offset=13)
//!   [5] IncClaimReduction  (9 rounds, offset=13)
#![allow(non_snake_case, clippy::print_stderr)]

use ark_bn254::Fr;
use ark_ff::{One, Zero};
use jolt_core::field::JoltField;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::poly::unipoly::UniPoly;

/// Verify that gruen_poly_deg_3 produces non-zero round polynomials
/// even when the inner quadratic coefficients come from boolean inputs
/// (where the total claim is zero).
#[test]
fn booleanity_gruen_round_poly_not_all_zero() {
    // Simulate a Booleanity Phase 1 scenario:
    // - The claim is zero (boolean RA → ra²-ra = 0 on hypercube)
    // - But the Gruen polynomial has non-zero evals at X=2,3

    // Create a GruenSplitEqPolynomial with some random-ish challenges
    let challenges: Vec<<Fr as JoltField>::Challenge> = (1..=4u128)
        .map(|v| <Fr as JoltField>::Challenge::from(v * 17 + 3))
        .collect();
    let B = GruenSplitEqPolynomial::<Fr>::new(&challenges, BindingOrder::LowToHigh);

    // For boolean RA values, the inner quadratic has:
    //   q_constant (eval_0 sum) ≠ 0 in general
    //   q_quadratic_coeff (eval_infty sum) ≠ 0 in general
    // But q_constant = q_quadratic_coeff (because G*F*(F-1) sum = G*F² sum - G*F sum
    // and they cancel in the total, giving claim = 0)
    //
    // For round 0 specifically: F=[1], so F_k=1 for all k.
    //   eval_0 = G_k * 1 * (1-1) = 0
    //   eval_infty = G_k * 1 * 1 = G_k
    // So q_constant = 0 but q_quadratic_coeff ≠ 0 (sum of G values).

    let q_constant = Fr::zero();
    let q_quadratic_coeff = Fr::from_u64(42); // non-zero
    let previous_claim = Fr::zero(); // booleanity claim is always 0

    let poly = B.gruen_poly_deg_3(q_constant, q_quadratic_coeff, previous_claim);

    let eval_0: Fr = poly.evaluate(&Fr::zero());
    let eval_1: Fr = poly.evaluate(&Fr::one());
    let eval_2: Fr = poly.evaluate(&Fr::from_u64(2));
    let eval_3: Fr = poly.evaluate(&Fr::from_u64(3));

    eprintln!("Booleanity round 0 (constructed example):");
    eprintln!("  eval[0] = {eval_0:?}");
    eprintln!("  eval[1] = {eval_1:?}");
    eprintln!("  eval[2] = {eval_2:?}");
    eprintln!("  eval[3] = {eval_3:?}");
    eprintln!("  sum(0+1) = {:?}", eval_0 + eval_1);

    // Claim is zero → H(0) + H(1) = 0
    assert_eq!(eval_0 + eval_1, Fr::zero(), "H(0)+H(1) should = 0 (claim)");
    // But H(0) = 0 specifically (because q_constant = 0 at round 0)
    assert_eq!(eval_0, Fr::zero(), "H(0) = 0 at round 0");
    assert_eq!(eval_1, Fr::zero(), "H(1) = 0 at round 0");
    // H(2) and H(3) are NOT zero
    assert_ne!(eval_2, Fr::zero(), "H(2) should be non-zero");
    assert_ne!(eval_3, Fr::zero(), "H(3) should be non-zero");

    // Compressed form: [c0, c2, c3] (drops c1)
    let compressed = poly.compress();
    eprintln!(
        "  compressed coeffs (c0, c2, c3): {:?}",
        compressed.coeffs_except_linear_term
    );
    // c0 = 0 but c2, c3 non-zero
    assert_eq!(
        compressed.coeffs_except_linear_term[0],
        Fr::zero(),
        "c0 should be 0"
    );
    assert_ne!(
        compressed.coeffs_except_linear_term[1],
        Fr::zero(),
        "c2 should be non-zero"
    );
    assert_ne!(
        compressed.coeffs_except_linear_term[2],
        Fr::zero(),
        "c3 should be non-zero"
    );
}

/// Verify that after binding the first challenge, subsequent Booleanity
/// round polynomials are fully non-zero (all 4 evals ≠ 0).
#[test]
fn booleanity_subsequent_rounds_fully_nonzero() {
    let challenges: Vec<<Fr as JoltField>::Challenge> = (1..=6u128)
        .map(|v| <Fr as JoltField>::Challenge::from(v * 13 + 7))
        .collect();
    let mut B = GruenSplitEqPolynomial::<Fr>::new(&challenges, BindingOrder::LowToHigh);

    // Round 0: claim = 0, q_constant = 0, q_quadratic_coeff ≠ 0
    let poly0 = B.gruen_poly_deg_3(Fr::zero(), Fr::from_u64(100), Fr::zero());
    let eval_0_at_0: Fr = poly0.evaluate(&Fr::zero());
    let eval_0_at_1: Fr = poly0.evaluate(&Fr::one());
    assert_eq!(eval_0_at_0 + eval_0_at_1, Fr::zero());

    // Bind first challenge
    let r_0 = <Fr as JoltField>::Challenge::from(999u128);
    B.bind(r_0);

    // The new "previous_claim" is poly0.evaluate(r_0), which is non-zero
    let new_claim: Fr = poly0.evaluate(&r_0);
    eprintln!("After round 0: new_claim = {new_claim:?}");
    assert_ne!(
        new_claim,
        Fr::zero(),
        "claim after round 0 should be non-zero"
    );

    // Round 1: with non-zero previous_claim and non-zero quadratic coeffs
    let poly1 = B.gruen_poly_deg_3(Fr::from_u64(50), Fr::from_u64(200), new_claim);
    let evals: Vec<Fr> = (0..4u64)
        .map(|t| poly1.evaluate(&Fr::from_u64(t)))
        .collect();

    eprintln!("Round 1 evals:");
    for (i, e) in evals.iter().enumerate() {
        eprintln!("  eval[{i}] = {e:?}");
    }

    // All 4 evals should be non-zero
    for (i, e) in evals.iter().enumerate() {
        assert_ne!(*e, Fr::zero(), "eval[{i}] should be non-zero in round 1");
    }
    // And they should sum correctly
    assert_eq!(
        evals[0] + evals[1],
        new_claim,
        "H(0)+H(1) should equal previous_claim"
    );
}

/// Verify that Dense iteration with eq-as-input produces correct round
/// polynomials: H(0)+H(1) = claim, and degree is 3 (eq × G²).
///
/// This confirms the kernel approach: formula = Σ_i γ^{2i} × eq × (G_i² - G_i),
/// Dense iteration with eq table as Input(0), G arrays as Input(1..D).
#[test]
fn dense_eq_input_produces_correct_round_poly() {
    // Small test: 3 address variables, 2 "G" polynomials
    let r_addr: Vec<<Fr as JoltField>::Challenge> = vec![
        <Fr as JoltField>::Challenge::from(11u128),
        <Fr as JoltField>::Challenge::from(22u128),
        <Fr as JoltField>::Challenge::from(33u128),
    ];

    // Build eq table for all 8 address vertices
    let eq_table: Vec<Fr> = EqPolynomial::<Fr>::evals(&r_addr);

    // Random G values (representing Σ_j eq(r_cycle, j) × ra_i(k, j))
    let G0: Vec<Fr> = (0..8u64).map(|k| Fr::from_u64(k * 7 + 1)).collect();
    let G1: Vec<Fr> = (0..8u64).map(|k| Fr::from_u64(k * 3 + 5)).collect();

    let gamma_sq_0 = Fr::one(); // γ^0
    let gamma_sq_1 = Fr::from_u64(49); // γ^2 (γ=7)

    // Compute the full claim: Σ_k eq(r, k) × Σ_i γ^{2i} × (G_i(k)² - G_i(k))
    let claim: Fr = (0..8usize)
        .map(|k| {
            let inner =
                gamma_sq_0 * (G0[k].square() - G0[k]) + gamma_sq_1 * (G1[k].square() - G1[k]);
            eq_table[k] * inner
        })
        .sum();

    eprintln!("Full claim = {claim:?}");

    // Evaluate round polynomial at {0, 1, 2, 3} for round 0 (LowToHigh, binding bit 0)
    let mut evals = vec![Fr::zero(); 4];
    for t in 0..4u64 {
        let t_f = Fr::from_u64(t);
        let mut sum = Fr::zero();
        for k_remaining in 0..4u64 {
            let k_lo = 2 * k_remaining as usize;
            let k_hi = 2 * k_remaining as usize + 1;

            let eq_val = eq_table[k_lo] + t_f * (eq_table[k_hi] - eq_table[k_lo]);
            let g0_val = G0[k_lo] + t_f * (G0[k_hi] - G0[k_lo]);
            let g1_val = G1[k_lo] + t_f * (G1[k_hi] - G1[k_lo]);

            let inner =
                gamma_sq_0 * (g0_val.square() - g0_val) + gamma_sq_1 * (g1_val.square() - g1_val);
            sum += eq_val * inner;
        }
        evals[t as usize] = sum;
    }

    eprintln!("Round 0 evals (Dense + eq-as-input):");
    for (i, e) in evals.iter().enumerate() {
        eprintln!("  eval[{i}] = {e:?}");
    }

    // Critical check: H(0) + H(1) = claim
    assert_eq!(
        evals[0] + evals[1],
        claim,
        "H(0)+H(1) should equal the full claim"
    );

    // Build UniPoly and verify degree 3 (4 coefficients)
    let poly = UniPoly::<Fr>::from_evals(&evals);
    let compressed = poly.compress();
    eprintln!(
        "Compressed coeffs: {:?}",
        compressed.coeffs_except_linear_term
    );
    // Should have 3 compressed coefficients [c0, c2, c3]
    assert_eq!(compressed.coeffs_except_linear_term.len(), 3);

    eprintln!("\nDense + eq-as-input approach produces valid degree-3 round polynomials.");
    eprintln!("This kernel formula: Σ_i γ^{{2i}} × eq(r, k) × (G_i(k)² - G_i(k))");
}

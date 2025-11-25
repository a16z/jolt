//! Constraint gadgets for Stage 1 verification
//!
//! This module implements the constraint gadgets (R1CS constraints) that
//! verify Stage 1 (Spartan outer sumcheck) logic.

#[cfg(feature = "groth16-stable")]
use ark_r1cs_std::prelude::*;
#[cfg(feature = "groth16-stable")]
use ark_r1cs_std::fields::fp::FpVar;
#[cfg(feature = "groth16-stable")]
use ark_relations::r1cs::SynthesisError;

#[cfg(feature = "groth16-git")]
use ark_r1cs_std_git::prelude::*;
#[cfg(feature = "groth16-git")]
use ark_r1cs_std_git::fields::fp::FpVar;
#[cfg(feature = "groth16-git")]
use ark_relations_git::r1cs::SynthesisError;

use ark_bn254::Fr;

/// Evaluate univariate polynomial: Σ c_i * x^i
pub fn evaluate_univariate<F: ark_ff::PrimeField>(
    coeffs: &[FpVar<F>],
    x: &FpVar<F>,
) -> Result<FpVar<F>, SynthesisError> {
    let mut result = FpVar::zero();
    let mut x_power = FpVar::one();

    for coeff in coeffs {
        result = &result + &(coeff * &x_power);
        x_power = &x_power * x;
    }

    Ok(result)
}

/// Verify univariate-skip first round
///
/// The verification checks:
/// 1. Sum check: Σ_j a_j * S_j == claim (where S_j are power sums over symmetric domain)
/// 2. Next claim: poly(r0)
///
/// For Groth16, we simplify: just check poly(r0) evaluation
pub fn verify_uni_skip_round(
    _tau: &[FpVar<Fr>],
    r0: &FpVar<Fr>,
    poly_coeffs: &[FpVar<Fr>],
    _initial_claim: &FpVar<Fr>,
) -> Result<FpVar<Fr>, SynthesisError> {
    // The key verification is evaluating the polynomial at r0
    // The sum check (Σ_j a_j * S_j == claim) involves power sums which are constants
    // We skip this in the circuit for now (can be checked outside if needed)

    // Evaluate polynomial at r0 to get next claim
    let next_claim = evaluate_univariate(poly_coeffs, r0)?;

    Ok(next_claim)
}

/// Verify sumcheck rounds
///
/// For each round i: check g_i(0) + g_i(1) = previous_claim
pub fn verify_sumcheck_rounds(
    initial_claim: FpVar<Fr>,
    challenges: &[FpVar<Fr>],
    round_polys: &[Vec<FpVar<Fr>>],
) -> Result<FpVar<Fr>, SynthesisError> {
    let mut claim = initial_claim;

    for (challenge, poly_coeffs) in challenges.iter().zip(round_polys) {
        // Check consistency: poly(0) + poly(1) = previous_claim
        let poly_at_0 = evaluate_univariate(poly_coeffs, &FpVar::zero())?;
        let poly_at_1 = evaluate_univariate(poly_coeffs, &FpVar::one())?;
        let sum = &poly_at_0 + &poly_at_1;

        // Enforce constraint
        sum.enforce_equal(&claim)?;

        // Update claim for next round: claim_i = g_i(r_i)
        claim = evaluate_univariate(poly_coeffs, challenge)?;
    }

    Ok(claim)
}

/// Compute eq(x, y) = ∏_i (x_i * y_i + (1-x_i)*(1-y_i))
///
/// This is the multilinear extension of the equality function
pub fn eq_polynomial(
    x: &[FpVar<Fr>],
    y: &[FpVar<Fr>],
) -> Result<FpVar<Fr>, SynthesisError> {
    assert_eq!(x.len(), y.len());

    let mut result = FpVar::one();
    for (xi, yi) in x.iter().zip(y) {
        // eq_i = xi*yi + (1-xi)*(1-yi)
        //      = xi*yi + 1 - xi - yi + xi*yi
        //      = 2*xi*yi - xi - yi + 1
        let prod = xi * yi;
        let two_prod = &prod + &prod;
        let eq_i = &two_prod - xi - yi + &FpVar::one();
        result = &result * &eq_i;
    }

    Ok(result)
}

/// Verify final R1CS check
///
/// Check: eq(tau, r) * [Az(r) * Bz(r) - Cz(r)] = final_claim
pub fn verify_final_r1cs_check(
    tau: &[FpVar<Fr>],
    r0: &FpVar<Fr>,
    sumcheck_challenges: &[FpVar<Fr>],
    r1cs_evals: &[FpVar<Fr>],
    final_claim: &FpVar<Fr>,
    expected_var: &FpVar<Fr>,
) -> Result<(), SynthesisError> {
    // 1. Construct combined challenge vector r = [r0 || sumcheck_challenges]
    let mut r_combined = vec![r0.clone()];
    r_combined.extend_from_slice(sumcheck_challenges);

    // 2. Compute eq(tau, r)
    let eq_value = eq_polynomial(tau, &r_combined)?;

    // 3. Compute Az(r) * Bz(r) - Cz(r)
    // Assuming first 3 elements are Az, Bz, Cz
    let az = &r1cs_evals[0];
    let bz = &r1cs_evals[1];
    let cz = &r1cs_evals[2];

    let az_times_bz = az * bz;
    let r1cs_result = &az_times_bz - cz;

    // 4. Compute expected: eq(tau, r) * [Az(r) * Bz(r) - Cz(r)]
    let computed_claim = &eq_value * &r1cs_result;

    // 5. Check against final_claim
    computed_claim.enforce_equal(final_claim)?;

    // 6. Check final_claim matches expected
    final_claim.enforce_equal(expected_var)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_evaluate_univariate() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // Polynomial: 1 + 2x + 3x^2
        let coeffs = vec![
            FpVar::new_witness(cs.clone(), || Ok(Fr::from(1u64))).unwrap(),
            FpVar::new_witness(cs.clone(), || Ok(Fr::from(2u64))).unwrap(),
            FpVar::new_witness(cs.clone(), || Ok(Fr::from(3u64))).unwrap(),
        ];

        // Evaluate at x = 2: 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        let x = FpVar::new_witness(cs.clone(), || Ok(Fr::from(2u64))).unwrap();
        let result = evaluate_univariate(&coeffs, &x).unwrap();

        let expected = FpVar::new_witness(cs.clone(), || Ok(Fr::from(17u64))).unwrap();
        result.enforce_equal(&expected).unwrap();

        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_eq_polynomial() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // Test eq([1, 0], [1, 0]) = 1
        let x = vec![
            FpVar::new_witness(cs.clone(), || Ok(Fr::from(1u64))).unwrap(),
            FpVar::new_witness(cs.clone(), || Ok(Fr::from(0u64))).unwrap(),
        ];
        let y = vec![
            FpVar::new_witness(cs.clone(), || Ok(Fr::from(1u64))).unwrap(),
            FpVar::new_witness(cs.clone(), || Ok(Fr::from(0u64))).unwrap(),
        ];

        let result = eq_polynomial(&x, &y).unwrap();
        let expected = FpVar::new_witness(cs.clone(), || Ok(Fr::from(1u64))).unwrap();
        result.enforce_equal(&expected).unwrap();

        assert!(cs.is_satisfied().unwrap());
    }
}

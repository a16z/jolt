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

/// Power sums S_k = Σ_t t^k over symmetric domain [-4, -3, ..., 5] (10 values)
/// Used for uni-skip first round verification.
/// Domain size N = 10 (OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE)
/// Polynomial has 28 coefficients (OUTER_FIRST_ROUND_POLY_NUM_COEFFS)
pub const POWER_SUMS: [i128; 28] = [
    10,
    5,
    85,
    125,
    1333,
    3125,
    25405,
    78125,
    535333,
    1953125,
    11982925,
    48828125,
    278766133,
    1220703125,
    6649985245,
    30517578125,
    161264049733,
    762939453125,
    3952911584365,
    19073486328125,
    97573430562133,
    476837158203125,
    2419432933612285,
    11920928955078125,
    60168159621439333,
    298023223876953125,
    1499128402505381005,
    7450580596923828125,
];

/// Convert i128 to Fr field element
/// Handles both positive and negative values, and values larger than u64::MAX
fn i128_to_fr(val: i128) -> Fr {
    if val >= 0 {
        // For positive values, split into high and low 64-bit parts
        let low = (val as u128 & 0xFFFFFFFFFFFFFFFF) as u64;
        let high = ((val as u128) >> 64) as u64;
        if high == 0 {
            Fr::from(low)
        } else {
            // 2^64 in the field
            let two_64 = Fr::from(1u64 << 32) * Fr::from(1u64 << 32);
            Fr::from(high) * two_64 + Fr::from(low)
        }
    } else {
        // For negative values, compute -(-val)
        let abs_val = (-val) as u128;
        let low = (abs_val & 0xFFFFFFFFFFFFFFFF) as u64;
        let high = (abs_val >> 64) as u64;
        if high == 0 {
            -Fr::from(low)
        } else {
            let two_64 = Fr::from(1u64 << 32) * Fr::from(1u64 << 32);
            -(Fr::from(high) * two_64 + Fr::from(low))
        }
    }
}

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
pub fn verify_uni_skip_round(
    _tau: &[FpVar<Fr>],
    r0: &FpVar<Fr>,
    poly_coeffs: &[FpVar<Fr>],
    initial_claim: &FpVar<Fr>,
) -> Result<FpVar<Fr>, SynthesisError> {
    // 1. Power sum check: Σ_j a_j * S_j == claim
    // For Stage 1, claim should be 0
    // S_j are precomputed constants from POWER_SUMS
    let mut domain_sum = FpVar::zero();
    for (j, coeff) in poly_coeffs.iter().enumerate() {
        if j < POWER_SUMS.len() {
            // Multiply coefficient by power sum constant
            let power_sum = POWER_SUMS[j];
            // Convert i128 to field element (handle large values)
            let power_sum_fr = i128_to_fr(power_sum);
            let power_sum_var = FpVar::constant(power_sum_fr);
            domain_sum = &domain_sum + &(coeff * &power_sum_var);
        }
    }

    // Enforce: domain_sum == initial_claim (which should be 0)
    domain_sum.enforce_equal(initial_claim)?;

    // 2. Evaluate polynomial at r0 to get next claim
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
/// This is the multilinear extension of the equality function.
/// Note: Currently unused but kept for future full verification implementation.
#[allow(dead_code)]
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

/// Verify final claim matches expected value
///
/// This is a simplified check - just verify the final sumcheck claim equals
/// the expected value (which should be computed off-circuit).
///
/// For a complete Stage 1 verification, the expected value would be:
///   tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
///
/// Where inner_sum_prod = A(rx_constr, r) * B(rx_constr, r) computed from
/// all 36 R1CS input evaluations. This is too complex for the initial circuit,
/// so we just verify the final claim matches a known expected value.
///
/// Note: Currently unused but kept for future full verification implementation.
#[allow(dead_code)]
pub fn verify_final_claim(
    final_claim: &FpVar<Fr>,
    expected_var: &FpVar<Fr>,
) -> Result<(), SynthesisError> {
    // Simply check final claim matches expected
    // The expected value is computed off-circuit by the witness extractor
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

    #[test]
    fn test_i128_to_fr() {
        // Test small positive
        assert_eq!(i128_to_fr(10), Fr::from(10u64));
        assert_eq!(i128_to_fr(0), Fr::from(0u64));

        // Test negative
        assert_eq!(i128_to_fr(-5), -Fr::from(5u64));

        // Test large positive (but within u64)
        assert_eq!(i128_to_fr(1_000_000_000_000i128), Fr::from(1_000_000_000_000u64));

        // Test very large positive (>= u64::MAX)
        // 7450580596923828125 fits in i128 and u64
        let large = 7450580596923828125i128;
        assert_eq!(i128_to_fr(large), Fr::from(large as u64));
    }

    #[test]
    fn test_power_sums_domain() {
        // The domain is [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        // S_0 = sum of 1s = 10
        // S_1 = sum of t = -4-3-2-1+0+1+2+3+4+5 = 5
        // S_2 = sum of t^2 = 16+9+4+1+0+1+4+9+16+25 = 85

        assert_eq!(POWER_SUMS[0], 10);
        assert_eq!(POWER_SUMS[1], 5);
        assert_eq!(POWER_SUMS[2], 85);

        // Verify by computation
        let domain: Vec<i64> = vec![-4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
        let s0: i128 = domain.iter().map(|_| 1i128).sum();
        let s1: i128 = domain.iter().map(|&t| t as i128).sum();
        let s2: i128 = domain.iter().map(|&t| (t as i128) * (t as i128)).sum();

        assert_eq!(s0, 10);
        assert_eq!(s1, 5);
        assert_eq!(s2, 85);
    }
}

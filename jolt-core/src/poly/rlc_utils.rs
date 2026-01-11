//! Shared utilities for RLC (Random Linear Combination) computations
//!
//! This module provides common functionality for computing RLC coefficients
//! used in various parts of the polynomial commitment scheme.

use crate::{field::JoltField, zkvm::witness::CommittedPolynomial};
use std::collections::BTreeMap;

/// Computes RLC coefficients for a set of polynomials given gamma powers.
///
/// For each polynomial that appears in `polynomial_claims`, this function
/// accumulates the corresponding gamma powers to compute the final RLC coefficient.
///
/// # Arguments
/// * `gamma_powers` - Powers of gamma (γ^i for i = 0, 1, ...)
/// * `polynomial_claims` - Iterator of (polynomial, claim) pairs
///
/// # Returns
/// A BTreeMap from polynomial to its RLC coefficient
pub fn compute_rlc_coefficients<F: JoltField>(
    gamma_powers: &[F],
    polynomial_claims: impl IntoIterator<Item = (CommittedPolynomial, F)>,
) -> BTreeMap<CommittedPolynomial, F> {
    let mut rlc_map = BTreeMap::new();

    for (gamma, (poly, _claim)) in gamma_powers.iter().zip(polynomial_claims) {
        *rlc_map.entry(poly).or_insert(F::zero()) += *gamma;
    }

    rlc_map
}

/// Computes the joint claim for RLC given gamma powers and individual claims.
///
/// Computes: Σ γ^i · claim_i
///
/// # Arguments
/// * `gamma_powers` - Powers of gamma (γ^i for i = 0, 1, ...)
/// * `claims` - Individual claims for each polynomial
///
/// # Returns
/// The joint claim value
pub fn compute_joint_claim<F: JoltField>(gamma_powers: &[F], claims: &[F]) -> F {
    gamma_powers
        .iter()
        .zip(claims.iter())
        .map(|(gamma, claim)| *gamma * *claim)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_rlc_coefficients() {
        // Test with simple values
        let gamma_powers = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(4u64)];
        let polynomial_claims = vec![
            (CommittedPolynomial::InstructionRa(0), Fr::from(10u64)),
            (CommittedPolynomial::InstructionRa(0), Fr::from(20u64)), // Same polynomial
            (CommittedPolynomial::InstructionRa(1), Fr::from(30u64)),
        ];

        let rlc_map = compute_rlc_coefficients(&gamma_powers, polynomial_claims);

        // InstructionRa(0) should have coefficient 1 + 2 = 3
        assert_eq!(
            rlc_map.get(&CommittedPolynomial::InstructionRa(0)),
            Some(&Fr::from(3u64))
        );
        // InstructionRa(1) should have coefficient 4
        assert_eq!(
            rlc_map.get(&CommittedPolynomial::InstructionRa(1)),
            Some(&Fr::from(4u64))
        );
    }

    #[test]
    fn test_joint_claim() {
        let gamma_powers = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(4u64)];
        let claims = vec![Fr::from(10u64), Fr::from(20u64), Fr::from(30u64)];

        let joint_claim = compute_joint_claim(&gamma_powers, &claims);

        // 1*10 + 2*20 + 4*30 = 10 + 40 + 120 = 170
        assert_eq!(joint_claim, Fr::from(170u64));
    }
}

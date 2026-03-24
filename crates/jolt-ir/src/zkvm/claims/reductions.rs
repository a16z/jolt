//! Claim reduction definitions.
//!
//! Claim reductions batch multiple opening claims from earlier stages
//! into fewer claims via eq-weighted sumchecks. Most follow a common
//! pattern: γ-weighted sum of openings at the output point.

use crate::builder::ExprBuilder;
use crate::claim::{ClaimDefinition, OpeningBinding};
use crate::PolynomialId;

// Verified against jolt-core/src/zkvm/claim_reductions/registers.rs
// Formula: Σ eq(r,j) · (rd_wv(j) + γ·rs1_v(j) + γ²·rs2_v(j))
// Degree: 2 (challenge × opening)
/// Registers claim reduction output claim.
///
/// Reduces rd_write_value, rs1_value, and rs2_value claims from
/// SpartanOuter into a single opening point via eq-weighted sumcheck.
///
/// Output claim: `eq·rd_wv + eq·γ·rs1_v + eq·γ²·rs2_v`
///
/// Challenge layout: `[eq_eval, γ, γ²]`
pub fn registers_claim_reduction() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let rd_wv = b.opening(0);
    let rs1_v = b.opening(1);
    let rs2_v = b.opening(2);
    let eq = b.challenge(0);
    let gamma = b.challenge(1);
    let gamma_sq = b.challenge(2);

    let expr = b.build(eq * rd_wv + eq * gamma * rs1_v + eq * gamma_sq * rs2_v);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial: PolynomialId::RdWriteValue,
            },
            OpeningBinding {
                var_id: 1,
                polynomial: PolynomialId::Rs1Value,
            },
            OpeningBinding {
                var_id: 2,
                polynomial: PolynomialId::Rs2Value,
            },
        ],
        num_challenges: 3,
    }
}

// Verified against jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs
// Formula: Σ eq(r,j) · (lookup + γ·left_op + γ²·right_op + γ³·left_input + γ⁴·right_input)
// Degree: 2 (challenge × opening)
/// Instruction lookups claim reduction output claim.
///
/// Reduces lookup output, left/right operands, and left/right instruction
/// inputs from SpartanOuter into a single opening point.
///
/// Output claim: `Σ_i c_i · opening_i` (5 openings, γ-weighted)
///
/// Challenge layout: `[c0, c1, c2, c3, c4]` where `c_i = eq_eval · γ^i`.
pub fn instruction_lookups_claim_reduction() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let lookup_out = b.opening(0);
    let left_op = b.opening(1);
    let right_op = b.opening(2);
    let left_input = b.opening(3);
    let right_input = b.opening(4);

    let c0 = b.challenge(0);
    let c1 = b.challenge(1);
    let c2 = b.challenge(2);
    let c3 = b.challenge(3);
    let c4 = b.challenge(4);

    let expr = b
        .build(c0 * lookup_out + c1 * left_op + c2 * right_op + c3 * left_input + c4 * right_input);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial: PolynomialId::LookupOutput,
            },
            OpeningBinding {
                var_id: 1,
                polynomial: PolynomialId::LeftLookupOperand,
            },
            OpeningBinding {
                var_id: 2,
                polynomial: PolynomialId::RightLookupOperand,
            },
            OpeningBinding {
                var_id: 3,
                polynomial: PolynomialId::LeftInstructionInput,
            },
            OpeningBinding {
                var_id: 4,
                polynomial: PolynomialId::RightInstructionInput,
            },
        ],
        num_challenges: 5,
    }
}

// Verified against jolt-core/src/zkvm/claim_reductions/ram_ra.rs
// Formula: Σ (eq_raf(c) + γ·eq_rw(c) + γ²·eq_val(c)) · ra(r_addr, c)
// Degree: 2 (challenge × opening)
/// RAM RA claim reduction output claim.
///
/// Reduces RA claims from three sources (RAF evaluation, RW checking,
/// val check) into a single opening point.
///
/// Output claim: `Σ_i c_i · ra`  (single RA opening, γ-weighted eq sums)
///
/// Challenge layout: `[c0]` where `c0 = eq_raf + γ·eq_rw + γ²·eq_val`.
pub fn ram_ra_claim_reduction() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let ra = b.opening(0);
    let c0 = b.challenge(0);

    let expr = b.build(c0 * ra);

    ClaimDefinition {
        expr,
        opening_bindings: vec![OpeningBinding {
            var_id: 0,
            polynomial: PolynomialId::RamAddress,
        }],
        num_challenges: 1,
    }
}

// Verified against jolt-core/src/zkvm/claim_reductions/increments.rs
// Formula: Σ ram_inc·(eq(r2,·) + γ·eq(r4,·)) + γ²·rd_inc·(eq(s4,·) + γ·eq(s5,·))
// Degree: 2 (challenge × opening)
/// Increment claim reduction output claim.
///
/// Reduces RAM inc and Rd inc claims from multiple sumcheck instances
/// into a single opening point.
///
/// Output claim: `c0·ram_inc + c1·rd_inc`
///
/// Challenge layout: `[c0, c1]` where weights combine γ-powers and eq evals.
pub fn increment_claim_reduction() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let ram_inc = b.opening(0);
    let rd_inc = b.opening(1);
    let c0 = b.challenge(0);
    let c1 = b.challenge(1);

    let expr = b.build(c0 * ram_inc + c1 * rd_inc);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial: PolynomialId::RamInc,
            },
            OpeningBinding {
                var_id: 1,
                polynomial: PolynomialId::RdInc,
            },
        ],
        num_challenges: 2,
    }
}

// Verified against jolt-core/src/zkvm/claim_reductions/hamming_weight.rs
// Formula: Σ_i G_i(k) · (γ^{3i} + γ^{3i+1}·eq_bool(k) + γ^{3i+2}·eq_virt_i(k))
// Degree: 2 (challenge × opening)
/// Hamming weight claim reduction output claim.
///
/// Reduces all RA polynomial opening claims from Booleanity, RA virtual, and
/// Hamming weight sumchecks into a single (address) opening point.
///
/// `polynomials` maps each polynomial index to its [`PolynomialId`]
/// (e.g., `InstructionRa(i)`, `BytecodeRa(j)`, `RamRa(k)`).
///
/// Output claim: `Σ_i c_i · poly_i`
///
/// Challenge layout: `[c_0, ..., c_{n-1}]` where each `c_i` encodes
/// `γ^{3i}·1 + γ^{3i+1}·eq_bool + γ^{3i+2}·eq_virt` (three claim types
/// per RA polynomial, combined into one coefficient).
///
/// Verified against jolt-core/src/zkvm/claim_reductions/hamming_weight.rs.
pub fn hamming_weight_claim_reduction(polynomials: &[PolynomialId]) -> ClaimDefinition {
    let n = polynomials.len();
    let b = ExprBuilder::new();

    let mut terms = b.zero();
    for i in 0..n {
        let poly_i = b.opening(i as u32);
        let c_i = b.challenge(i as u32);
        terms = terms + c_i * poly_i;
    }

    let expr = b.build(terms);

    let opening_bindings = (0..n)
        .map(|i| OpeningBinding {
            var_id: i as u32,
            polynomial: polynomials[i],
        })
        .collect();

    ClaimDefinition {
        expr,
        opening_bindings,
        num_challenges: n as u32,
    }
}

// Verified against jolt-core/src/zkvm/claim_reductions/advice.rs
// Formula: Σ advice(y) · eq(r_val, y) → output: eq_combined · advice
// Degree: 2 (challenge × opening)
/// Advice claim reduction output claim (address phase).
///
/// The advice two-phase reduction has two stages:
/// - **Cycle phase:** Output is a direct polynomial opening (no formula needed).
/// - **Address phase:** Reduces the cycle-phase intermediate claim to a
///   final opening point.
///
/// Output claim: `c0 · advice`
///
/// where `c0 = eq_combined · scale`.
pub fn advice_claim_reduction_address() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let advice = b.opening(0);
    let c0 = b.challenge(0);

    let expr = b.build(c0 * advice);

    ClaimDefinition {
        expr,
        opening_bindings: vec![OpeningBinding {
            var_id: 0,
            polynomial: PolynomialId::TrustedAdvice,
        }],
        num_challenges: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};

    #[test]
    fn registers_reduction_formula() {
        let claim = registers_claim_reduction();
        let rd_wv = Fr::from_u64(2);
        let rs1_v = Fr::from_u64(3);
        let rs2_v = Fr::from_u64(5);
        let eq = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let gamma_sq = gamma * gamma;

        let expected = eq * rd_wv + eq * gamma * rs1_v + eq * gamma_sq * rs2_v;
        let result = claim.evaluate::<Fr>(&[rd_wv, rs1_v, rs2_v], &[eq, gamma, gamma_sq]);
        assert_eq!(result, expected);
    }

    #[test]
    fn instruction_lookups_reduction_formula() {
        let claim = instruction_lookups_claim_reduction();
        let openings: Vec<Fr> = (1..=5).map(Fr::from_u64).collect();
        let challenges: Vec<Fr> = (10..=14).map(Fr::from_u64).collect();

        let expected: Fr = openings
            .iter()
            .zip(challenges.iter())
            .map(|(o, c)| *c * *o)
            .sum();
        let result = claim.evaluate::<Fr>(&openings, &challenges);
        assert_eq!(result, expected);
    }

    #[test]
    fn ram_ra_reduction_formula() {
        let claim = ram_ra_claim_reduction();
        let ra = Fr::from_u64(5);
        let c0 = Fr::from_u64(13);

        let result = claim.evaluate::<Fr>(&[ra], &[c0]);
        assert_eq!(result, Fr::from_u64(65));
    }

    #[test]
    fn increment_reduction_formula() {
        let claim = increment_claim_reduction();
        let ram_inc = Fr::from_u64(3);
        let rd_inc = Fr::from_u64(7);
        let c0 = Fr::from_u64(11);
        let c1 = Fr::from_u64(13);

        // 11*3 + 13*7 = 33 + 91 = 124
        let result = claim.evaluate::<Fr>(&[ram_inc, rd_inc], &[c0, c1]);
        assert_eq!(result, Fr::from_u64(124));
    }

    #[test]
    fn hamming_weight_reduction_formula() {
        let polynomials = vec![
            PolynomialId::InstructionRa(0),
            PolynomialId::BytecodeRa(0),
            PolynomialId::RamRa(0),
        ];
        let claim = hamming_weight_claim_reduction(&polynomials);
        let polys: Vec<Fr> = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let challenges: Vec<Fr> = vec![Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];

        // 7*2 + 11*3 + 13*5 = 14 + 33 + 65 = 112
        let result = claim.evaluate::<Fr>(&polys, &challenges);
        assert_eq!(result, Fr::from_u64(112));
    }

    #[test]
    fn sop_equivalence_registers_reduction() {
        let claim = registers_claim_reduction();
        let openings: Vec<Fr> = (1..=3).map(Fr::from_u64).collect();
        let challenges: Vec<Fr> = (5..=7).map(Fr::from_u64).collect();

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_sop = claim
            .expr
            .to_sum_of_products()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_sop);
    }

    #[test]
    fn sop_equivalence_hamming_weight_reduction() {
        let polynomials = vec![
            PolynomialId::InstructionRa(0),
            PolynomialId::InstructionRa(1),
            PolynomialId::BytecodeRa(0),
            PolynomialId::RamRa(0),
        ];
        let claim = hamming_weight_claim_reduction(&polynomials);
        let openings: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let challenges: Vec<Fr> = (10..=13).map(Fr::from_u64).collect();

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_sop = claim
            .expr
            .to_sum_of_products()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_sop);
    }
}

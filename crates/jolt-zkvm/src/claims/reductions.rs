//! Claim reduction definitions.
//!
//! Claim reductions batch multiple opening claims from earlier stages
//! into fewer claims via eq-weighted sumchecks. Most follow a common
//! pattern: γ-weighted sum of openings at the output point.

use jolt_ir::{
    ChallengeBinding, ChallengeSource, ClaimDefinition, ExprBuilder, OpeningBinding,
};

use crate::tags::{poly, sumcheck};

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
                polynomial_tag: poly::RD_WRITE_VALUE,
                sumcheck_tag: sumcheck::REGISTERS_CLAIM_REDUCTION,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::RS1_VALUE,
                sumcheck_tag: sumcheck::REGISTERS_CLAIM_REDUCTION,
            },
            OpeningBinding {
                var_id: 2,
                polynomial_tag: poly::RS2_VALUE,
                sumcheck_tag: sumcheck::REGISTERS_CLAIM_REDUCTION,
            },
        ],
        challenge_bindings: vec![
            ChallengeBinding {
                var_id: 0,
                source: ChallengeSource::Derived,
            },
            ChallengeBinding {
                var_id: 1,
                source: ChallengeSource::Derived,
            },
            ChallengeBinding {
                var_id: 2,
                source: ChallengeSource::Derived,
            },
        ],
    }
}

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

    let expr = b.build(
        c0 * lookup_out + c1 * left_op + c2 * right_op + c3 * left_input + c4 * right_input,
    );

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial_tag: poly::LOOKUP_OUTPUT,
                sumcheck_tag: sumcheck::INSTRUCTION_CLAIM_REDUCTION,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::LEFT_LOOKUP_OPERAND,
                sumcheck_tag: sumcheck::INSTRUCTION_CLAIM_REDUCTION,
            },
            OpeningBinding {
                var_id: 2,
                polynomial_tag: poly::RIGHT_LOOKUP_OPERAND,
                sumcheck_tag: sumcheck::INSTRUCTION_CLAIM_REDUCTION,
            },
            OpeningBinding {
                var_id: 3,
                polynomial_tag: poly::LEFT_INSTRUCTION_INPUT,
                sumcheck_tag: sumcheck::INSTRUCTION_CLAIM_REDUCTION,
            },
            OpeningBinding {
                var_id: 4,
                polynomial_tag: poly::RIGHT_INSTRUCTION_INPUT,
                sumcheck_tag: sumcheck::INSTRUCTION_CLAIM_REDUCTION,
            },
        ],
        challenge_bindings: (0..5)
            .map(|i| ChallengeBinding {
                var_id: i,
                source: ChallengeSource::Derived,
            })
            .collect(),
    }
}

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
    let c0 = b.challenge(0); // combined eq weight

    let expr = b.build(c0 * ra);

    ClaimDefinition {
        expr,
        opening_bindings: vec![OpeningBinding {
            var_id: 0,
            polynomial_tag: poly::RAM_RA,
            sumcheck_tag: sumcheck::RAM_RA_CLAIM_REDUCTION,
        }],
        challenge_bindings: vec![ChallengeBinding {
            var_id: 0,
            source: ChallengeSource::Derived,
        }],
    }
}

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
                polynomial_tag: poly::RAM_INC,
                sumcheck_tag: sumcheck::INC_CLAIM_REDUCTION,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::RD_INC,
                sumcheck_tag: sumcheck::INC_CLAIM_REDUCTION,
            },
        ],
        challenge_bindings: vec![
            ChallengeBinding {
                var_id: 0,
                source: ChallengeSource::Derived,
            },
            ChallengeBinding {
                var_id: 1,
                source: ChallengeSource::Derived,
            },
        ],
    }
}

/// Hamming weight claim reduction output claim.
///
/// Reduces Hamming weight polynomials from multiple chunk types into
/// a single opening point. `n_poly_types` is the number of polynomial types.
///
/// Output claim: `Σ_i c_i · poly_i`
///
/// Challenge layout: `[c_0, ..., c_{n-1}]` with combined eq·γ-power weights.
pub fn hamming_weight_claim_reduction(n_poly_types: usize) -> ClaimDefinition {
    let b = ExprBuilder::new();

    let mut terms = b.zero();
    for i in 0..n_poly_types {
        let poly_i = b.opening(i as u32);
        let c_i = b.challenge(i as u32);
        terms = terms + c_i * poly_i;
    }

    let expr = b.build(terms);

    let opening_bindings = (0..n_poly_types)
        .map(|i| OpeningBinding {
            var_id: i as u32,
            polynomial_tag: 0, // Resolved at runtime per polynomial type
            sumcheck_tag: sumcheck::HAMMING_WEIGHT_CLAIM_REDUCTION,
        })
        .collect();

    let challenge_bindings = (0..n_poly_types)
        .map(|i| ChallengeBinding {
            var_id: i as u32,
            source: ChallengeSource::Derived,
        })
        .collect();

    ClaimDefinition {
        expr,
        opening_bindings,
        challenge_bindings,
    }
}

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
    let c0 = b.challenge(0); // eq_combined * scale

    let expr = b.build(c0 * advice);

    ClaimDefinition {
        expr,
        opening_bindings: vec![OpeningBinding {
            var_id: 0,
            polynomial_tag: 0, // Resolved at runtime (trusted or untrusted advice)
            sumcheck_tag: sumcheck::ADVICE_CLAIM_REDUCTION,
        }],
        challenge_bindings: vec![ChallengeBinding {
            var_id: 0,
            source: ChallengeSource::Derived,
        }],
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
        let claim = hamming_weight_claim_reduction(3);
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
        let claim = hamming_weight_claim_reduction(4);
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

//! Spartan-related claim definitions.
//!
//! The Spartan outer sumcheck and its follow-up sumchecks (product
//! virtualization, shift, instruction input) have specialized claim
//! structures. The outer sumcheck uses univariate skip; the remaining
//! sumchecks are standard.

use crate::builder::ExprBuilder;
use crate::claim::{ChallengeBinding, ChallengeSource, ClaimDefinition, OpeningBinding};
use crate::zkvm::tags::{poly, sumcheck};

/// Shift sumcheck output claim.
///
/// Verifies PC, unexpanded PC, is_virtual, is_first_in_sequence, and is_noop
/// shifted-by-one relationships.
///
/// Output claim:
/// ```text
///   c0·unexpanded_pc + c1·pc + c2·is_virtual
///   + c3·is_first_in_seq + c4·noop_shifted + c5
/// ```
///
/// The challenges encode eq-plus-one evaluations weighted by γ-powers.
/// `c5` is a constant term from `(1 − noop)` expansion.
pub fn shift() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let unexpanded_pc = b.opening(0);
    let pc = b.opening(1);
    let is_virtual = b.opening(2);
    let is_first_in_seq = b.opening(3);
    let noop_shifted = b.opening(4);

    let c0 = b.challenge(0);
    let c1 = b.challenge(1);
    let c2 = b.challenge(2);
    let c3 = b.challenge(3);
    let c4 = b.challenge(4);
    let c5 = b.challenge(5);

    let expr = b.build(
        c0 * unexpanded_pc
            + c1 * pc
            + c2 * is_virtual
            + c3 * is_first_in_seq
            + c4 * noop_shifted
            + c5,
    );

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial_tag: poly::NEXT_UNEXPANDED_PC,
                sumcheck_tag: sumcheck::SHIFT,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::NEXT_PC,
                sumcheck_tag: sumcheck::SHIFT,
            },
            OpeningBinding {
                var_id: 2,
                polynomial_tag: poly::NEXT_IS_VIRTUAL,
                sumcheck_tag: sumcheck::SHIFT,
            },
            OpeningBinding {
                var_id: 3,
                polynomial_tag: poly::NEXT_IS_FIRST_IN_SEQUENCE,
                sumcheck_tag: sumcheck::SHIFT,
            },
            OpeningBinding {
                var_id: 4,
                polynomial_tag: poly::NEXT_IS_NOOP,
                sumcheck_tag: sumcheck::SHIFT,
            },
        ],
        challenge_bindings: (0..=5)
            .map(|i| ChallengeBinding {
                var_id: i,
                source: ChallengeSource::Derived,
            })
            .collect(),
    }
}

/// Instruction input virtualization output claim.
///
/// Batches left and right instruction inputs via γ.
///
/// Output claim: `eq·(right_input + γ·left_input)`
///
/// Challenge layout: `[eq_eval, γ]`
pub fn instruction_input() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let right = b.opening(0);
    let left = b.opening(1);
    let eq = b.challenge(0);
    let gamma = b.challenge(1);

    let expr = b.build(eq * (right + gamma * left));

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial_tag: poly::RIGHT_INSTRUCTION_INPUT,
                sumcheck_tag: sumcheck::INSTRUCTION_INPUT_VIRTUAL,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::LEFT_INSTRUCTION_INPUT,
                sumcheck_tag: sumcheck::INSTRUCTION_INPUT_VIRTUAL,
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

/// Product virtual remainder output claim.
///
/// Batches five R1CS product-virtual constraints via γ-power coefficients:
///
/// | Index | Constraint           | Left factor           | Right factor                  |
/// |-------|----------------------|-----------------------|-------------------------------|
/// | 0     | Product              | left_instruction_input| right_instruction_input       |
/// | 1     | WriteLookupOutputToRD| is_rd_not_zero        | write_lookup_output_to_rd_flag|
/// | 2     | WritePCtoRD          | is_rd_not_zero        | jump_flag                     |
/// | 3     | ShouldBranch         | lookup_output         | branch_flag                   |
/// | 4     | ShouldJump           | jump_flag             | (1 − next_is_noop)            |
///
/// Output claim (evaluated after eq is factored out):
/// ```text
///   c0·left·right + c1·rd_nz·wl + c2·rd_nz·jump
///   + c3·lookup·branch + c4·jump + c5·jump·noop
/// ```
///
/// Challenge layout: `[c0, c1, c2, c3, c4, c5]` where `c_i = γ^i`
/// for i < 5 and `c5 = −γ^4` (from expanding `γ^4 · jump · (1 − noop)`).
pub fn product_virtual_remainder() -> ClaimDefinition {
    let b = ExprBuilder::new();

    let left_inst = b.opening(0);
    let right_inst = b.opening(1);
    let is_rd_nz = b.opening(2);
    let wl_flag = b.opening(3);
    let jump_flag = b.opening(4);
    let lookup_out = b.opening(5);
    let branch_flag = b.opening(6);
    let next_noop = b.opening(7);

    let c0 = b.challenge(0);
    let c1 = b.challenge(1);
    let c2 = b.challenge(2);
    let c3 = b.challenge(3);
    let c4 = b.challenge(4);
    let c5 = b.challenge(5); // −γ^4

    let expr = b.build(
        c0 * left_inst * right_inst
            + c1 * is_rd_nz * wl_flag
            + c2 * is_rd_nz * jump_flag
            + c3 * lookup_out * branch_flag
            + c4 * jump_flag
            + c5 * jump_flag * next_noop,
    );

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial_tag: poly::LEFT_INSTRUCTION_INPUT,
                sumcheck_tag: sumcheck::SPARTAN_PRODUCT_VIRTUAL,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::RIGHT_INSTRUCTION_INPUT,
                sumcheck_tag: sumcheck::SPARTAN_PRODUCT_VIRTUAL,
            },
            OpeningBinding {
                var_id: 2,
                polynomial_tag: poly::IS_RD_NOT_ZERO,
                sumcheck_tag: sumcheck::SPARTAN_PRODUCT_VIRTUAL,
            },
            OpeningBinding {
                var_id: 3,
                polynomial_tag: poly::WRITE_LOOKUP_OUTPUT_TO_RD_FLAG,
                sumcheck_tag: sumcheck::SPARTAN_PRODUCT_VIRTUAL,
            },
            OpeningBinding {
                var_id: 4,
                polynomial_tag: poly::JUMP_FLAG,
                sumcheck_tag: sumcheck::SPARTAN_PRODUCT_VIRTUAL,
            },
            OpeningBinding {
                var_id: 5,
                polynomial_tag: poly::LOOKUP_OUTPUT,
                sumcheck_tag: sumcheck::SPARTAN_PRODUCT_VIRTUAL,
            },
            OpeningBinding {
                var_id: 6,
                polynomial_tag: poly::BRANCH_FLAG,
                sumcheck_tag: sumcheck::SPARTAN_PRODUCT_VIRTUAL,
            },
            OpeningBinding {
                var_id: 7,
                polynomial_tag: poly::NEXT_IS_NOOP,
                sumcheck_tag: sumcheck::SPARTAN_PRODUCT_VIRTUAL,
            },
        ],
        challenge_bindings: (0..=5)
            .map(|i| ChallengeBinding {
                var_id: i,
                source: ChallengeSource::Derived,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};

    #[test]
    fn shift_formula() {
        let claim = shift();
        let openings: Vec<Fr> = (1..=5).map(Fr::from_u64).collect();
        let challenges: Vec<Fr> = (10..=15).map(Fr::from_u64).collect();

        let expected = challenges[0] * openings[0]
            + challenges[1] * openings[1]
            + challenges[2] * openings[2]
            + challenges[3] * openings[3]
            + challenges[4] * openings[4]
            + challenges[5];

        let result = claim.evaluate::<Fr>(&openings, &challenges);
        assert_eq!(result, expected);
    }

    #[test]
    fn instruction_input_formula() {
        let claim = instruction_input();
        let right = Fr::from_u64(3);
        let left = Fr::from_u64(5);
        let eq = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);

        // eq*(right + γ*left) = 7*(3 + 11*5) = 7*58 = 406
        let result = claim.evaluate::<Fr>(&[right, left], &[eq, gamma]);
        assert_eq!(result, Fr::from_u64(406));
    }

    #[test]
    fn sop_equivalence_shift() {
        let claim = shift();
        let openings: Vec<Fr> = (1..=5).map(Fr::from_u64).collect();
        let challenges: Vec<Fr> = (10..=15).map(Fr::from_u64).collect();

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_sop = claim
            .expr
            .to_sum_of_products()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_sop);
    }

    #[test]
    fn product_virtual_remainder_formula() {
        let claim = product_virtual_remainder();

        // openings: [left, right, rd_nz, wl, jump, lookup, branch, noop]
        let left = Fr::from_u64(2);
        let right = Fr::from_u64(3);
        let rd_nz = Fr::from_u64(1);
        let wl = Fr::from_u64(5);
        let jump = Fr::from_u64(7);
        let lookup = Fr::from_u64(11);
        let branch = Fr::from_u64(13);
        let noop = Fr::from_u64(0);
        let openings = vec![left, right, rd_nz, wl, jump, lookup, branch, noop];

        // challenges: [γ^0, γ^1, γ^2, γ^3, γ^4, −γ^4]
        let gamma = Fr::from_u64(17);
        let g0 = Fr::from_u64(1);
        let g1 = gamma;
        let g2 = gamma * gamma;
        let g3 = g2 * gamma;
        let g4 = g3 * gamma;
        let neg_g4 = -g4;
        let challenges = vec![g0, g1, g2, g3, g4, neg_g4];

        // Expected: c0*left*right + c1*rd_nz*wl + c2*rd_nz*jump
        //         + c3*lookup*branch + c4*jump + c5*jump*noop
        let expected = g0 * left * right
            + g1 * rd_nz * wl
            + g2 * rd_nz * jump
            + g3 * lookup * branch
            + g4 * jump
            + neg_g4 * jump * noop;

        let result = claim.evaluate::<Fr>(&openings, &challenges);
        assert_eq!(result, expected);
    }

    #[test]
    fn sop_equivalence_product_virtual() {
        let claim = product_virtual_remainder();
        let openings: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let gamma = Fr::from_u64(5);
        let g: Vec<Fr> = (0..5)
            .scan(Fr::from_u64(1), |acc, _| {
                let v = *acc;
                *acc *= gamma;
                Some(v)
            })
            .collect();
        let challenges = vec![g[0], g[1], g[2], g[3], g[4], -g[4]];

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_sop = claim
            .expr
            .to_sum_of_products()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_sop);
    }
}

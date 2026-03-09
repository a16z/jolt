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
}

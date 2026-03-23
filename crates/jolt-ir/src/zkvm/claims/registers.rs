//! Register subsystem claim definitions.
//!
//! Registers use a similar read-write checking pattern to RAM, but with
//! three register files (rd, rs1, rs2) batched via γ-powers.

use crate::builder::ExprBuilder;
use crate::claim::{ChallengeBinding, ChallengeSource, ClaimDefinition, OpeningBinding};
use crate::zkvm::tags::{poly, sumcheck};

// Verified against jolt-core/src/zkvm/registers/read_write_checking.rs
// Formula: Σ eq(r,j) · (rd_wa·(inc+val) + γ·rs1_ra·val + γ²·rs2_ra·val)
// Degree: 3 (challenge × opening × opening)
/// Register read-write checking output claim.
///
/// Batches rd write, rs1 read, and rs2 read into a single sumcheck via γ.
///
/// Output claim:
/// ```text
///   eq·rd_wa·inc + eq·rd_wa·val + eq·γ·rs1_ra·val + eq·γ²·rs2_ra·val
/// ```
///
/// Challenge layout: `[eq_eval, γ, γ²]`
pub fn registers_read_write_checking() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let val = b.opening(0);
    let rs1_ra = b.opening(1);
    let rs2_ra = b.opening(2);
    let rd_wa = b.opening(3);
    let inc = b.opening(4);

    let eq = b.challenge(0);
    let gamma = b.challenge(1);
    let gamma_sq = b.challenge(2);

    let expr = b.build(
        eq * rd_wa * inc
            + eq * rd_wa * val
            + eq * gamma * rs1_ra * val
            + eq * gamma_sq * rs2_ra * val,
    );

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial_tag: poly::REGISTERS_VAL,
                sumcheck_tag: sumcheck::REGISTERS_READ_WRITE_CHECKING,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::RS1_RA,
                sumcheck_tag: sumcheck::REGISTERS_READ_WRITE_CHECKING,
            },
            OpeningBinding {
                var_id: 2,
                polynomial_tag: poly::RS2_RA,
                sumcheck_tag: sumcheck::REGISTERS_READ_WRITE_CHECKING,
            },
            OpeningBinding {
                var_id: 3,
                polynomial_tag: poly::RD_WA,
                sumcheck_tag: sumcheck::REGISTERS_READ_WRITE_CHECKING,
            },
            OpeningBinding {
                var_id: 4,
                polynomial_tag: poly::RD_INC,
                sumcheck_tag: sumcheck::REGISTERS_READ_WRITE_CHECKING,
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

// Verified against jolt-core/src/zkvm/registers/val_evaluation.rs
// Formula: Σ inc(j) · wa(r_addr,j) · LT(r_cycle,j) → output: inc · wa · lt_eval
// Degree: 3 (challenge × opening × opening, via Toom-Cook at {0,1,2,∞})
/// Register value evaluation output claim.
///
/// Output claim: `c0·inc·wa`
///
/// where `c0` is a combined eq/LT evaluation challenge (same structure as
/// RAM val check).
pub fn registers_val_evaluation() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let inc = b.opening(0);
    let wa = b.opening(1);
    let c0 = b.challenge(0);

    let expr = b.build(c0 * inc * wa);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial_tag: poly::RD_INC,
                sumcheck_tag: sumcheck::REGISTERS_VAL_EVALUATION,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::RD_WA,
                sumcheck_tag: sumcheck::REGISTERS_VAL_EVALUATION,
            },
        ],
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
    fn registers_rw_checking_formula() {
        let claim = registers_read_write_checking();
        let val = Fr::from_u64(3);
        let rs1_ra = Fr::from_u64(2);
        let rs2_ra = Fr::from_u64(5);
        let rd_wa = Fr::from_u64(4);
        let inc = Fr::from_u64(7);
        let eq = Fr::from_u64(11);
        let gamma = Fr::from_u64(13);
        let gamma_sq = gamma * gamma;

        let expected = eq * rd_wa * inc
            + eq * rd_wa * val
            + eq * gamma * rs1_ra * val
            + eq * gamma_sq * rs2_ra * val;

        let result =
            claim.evaluate::<Fr>(&[val, rs1_ra, rs2_ra, rd_wa, inc], &[eq, gamma, gamma_sq]);
        assert_eq!(result, expected);
    }

    #[test]
    fn registers_val_eval_formula() {
        let claim = registers_val_evaluation();
        let inc = Fr::from_u64(3);
        let wa = Fr::from_u64(5);
        let c0 = Fr::from_u64(7);

        let result = claim.evaluate::<Fr>(&[inc, wa], &[c0]);
        assert_eq!(result, Fr::from_u64(105));
    }

    #[test]
    fn sop_equivalence_registers_rw() {
        let claim = registers_read_write_checking();
        let openings: Vec<Fr> = (1..=5).map(Fr::from_u64).collect();
        let challenges: Vec<Fr> = (6..=8).map(Fr::from_u64).collect();

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_sop = claim
            .expr
            .to_sum_of_products()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_sop);
    }
}

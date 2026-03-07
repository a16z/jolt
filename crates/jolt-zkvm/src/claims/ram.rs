//! RAM subsystem claim definitions.
//!
//! These encode the output claim formulas for the five RAM-related sumcheck
//! instances. Each function returns a [`ClaimDefinition`] whose expression
//! captures the exact polynomial identity, and whose bindings map symbolic
//! variables to concrete Jolt polynomials and sumcheck stages.

use jolt_ir::{
    ChallengeBinding, ChallengeSource, ClaimDefinition, ExprBuilder, OpeningBinding,
};

use crate::tags::{poly, sumcheck};

/// Hamming booleanity: `H² − H`, scaled by eq evaluation challenges.
///
/// Proves that the Hamming weight polynomial H is boolean (0 or 1) at every
/// point. The sumcheck input claim is zero (pure zero-check).
///
/// Output claim: `challenge[0]·H² + challenge[1]·H`
///
/// where `challenge[0] = eq_eval` and `challenge[1] = −eq_eval`.
pub fn hamming_booleanity() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let eq = b.challenge(0);
    let neg_eq = b.challenge(1);

    // eq·H² + (−eq)·H = eq·(H² − H)
    let expr = b.build(eq * h * h + neg_eq * h);

    ClaimDefinition {
        expr,
        opening_bindings: vec![OpeningBinding {
            var_id: 0,
            polynomial_tag: poly::RAM_HAMMING_WEIGHT,
            sumcheck_tag: sumcheck::RAM_HAMMING_BOOLEANITY,
        }],
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

/// RAM read-write checking output claim.
///
/// Verifies consistency between read values, write values, and increments
/// across the RAM address/cycle space.
///
/// Output claim: `c0·ra·val + c1·ra·inc`
///
/// where `c0 = eq_eval·(1+γ)` and `c1 = eq_eval·γ`.
pub fn ram_read_write_checking() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let ra = b.opening(0);
    let val = b.opening(1);
    let inc = b.opening(2);
    let c0 = b.challenge(0); // eq*(1+γ)
    let c1 = b.challenge(1); // eq*γ

    let expr = b.build(c0 * ra * val + c1 * ra * inc);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial_tag: poly::RAM_RA,
                sumcheck_tag: sumcheck::RAM_READ_WRITE_CHECKING,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::RAM_VAL,
                sumcheck_tag: sumcheck::RAM_READ_WRITE_CHECKING,
            },
            OpeningBinding {
                var_id: 2,
                polynomial_tag: poly::RAM_INC,
                sumcheck_tag: sumcheck::RAM_READ_WRITE_CHECKING,
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

/// RAM output check.
///
/// Verifies that final RAM values match expected I/O values in the
/// designated memory region.
///
/// Output claim: `c0·val_final + c1`
///
/// where `c0 = eq_eval·io_mask_eval` and `c1 = −eq_eval·io_mask_eval·val_io_eval`.
pub fn ram_output_check() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let val_final = b.opening(0);
    let c0 = b.challenge(0); // eq*io_mask
    let c1 = b.challenge(1); // -eq*io_mask*val_io (constant term)

    let expr = b.build(c0 * val_final + c1);

    ClaimDefinition {
        expr,
        opening_bindings: vec![OpeningBinding {
            var_id: 0,
            polynomial_tag: poly::RAM_VAL_FINAL,
            sumcheck_tag: sumcheck::RAM_OUTPUT_CHECK,
        }],
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

/// RAM value check output claim.
///
/// Relates increment polynomials, write-address polynomials, and a
/// less-than evaluation to verify RAM value consistency across cycles.
///
/// Output claim: `c0·inc·wa`
///
/// where `c0 = LT(r_cycle', r_cycle) + γ`.
pub fn ram_val_check() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let inc = b.opening(0);
    let wa = b.opening(1);
    let c0 = b.challenge(0); // LT + γ

    let expr = b.build(c0 * inc * wa);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial_tag: poly::RAM_INC,
                sumcheck_tag: sumcheck::RAM_VAL_CHECK,
            },
            OpeningBinding {
                var_id: 1,
                polynomial_tag: poly::RAM_RA,
                sumcheck_tag: sumcheck::RAM_VAL_CHECK,
            },
        ],
        challenge_bindings: vec![ChallengeBinding {
            var_id: 0,
            source: ChallengeSource::Derived,
        }],
    }
}

/// RAM value check input claim.
///
/// The input claim for the RAM value check sumcheck has a complex structure
/// involving advice polynomial contributions.
///
/// Input claim: `val_rw + γ·val_final + c1 + Σ(c_{i+2}·advice_i)`
///
/// where `c1 = −(1+γ)·init_eval_public` and `c_{i+2} = −(1+γ)·selector_i`.
///
/// `n_advice` specifies the number of advice polynomial contributions.
pub fn ram_val_check_input(n_advice: usize) -> ClaimDefinition {
    let b = ExprBuilder::new();
    let val_rw = b.opening(0);
    let val_final = b.opening(1);
    let gamma = b.challenge(0);
    let neg_init = b.challenge(1); // -(1+γ)*init_eval_public

    let mut result = val_rw + gamma * val_final + neg_init;

    for i in 0..n_advice {
        let advice_i = b.opening(2 + i as u32);
        let selector_i = b.challenge(2 + i as u32);
        result = result + selector_i * advice_i;
    }

    let expr = b.build(result);

    let mut opening_bindings = vec![
        OpeningBinding {
            var_id: 0,
            polynomial_tag: poly::RAM_VAL,
            sumcheck_tag: sumcheck::RAM_READ_WRITE_CHECKING,
        },
        OpeningBinding {
            var_id: 1,
            polynomial_tag: poly::RAM_VAL_FINAL,
            sumcheck_tag: sumcheck::RAM_OUTPUT_CHECK,
        },
    ];

    let mut challenge_bindings = vec![
        ChallengeBinding {
            var_id: 0,
            source: ChallengeSource::Derived,
        },
        ChallengeBinding {
            var_id: 1,
            source: ChallengeSource::Derived,
        },
    ];

    for i in 0..n_advice {
        opening_bindings.push(OpeningBinding {
            var_id: 2 + i as u32,
            polynomial_tag: 0, // Advice polynomial tag resolved at runtime
            sumcheck_tag: sumcheck::RAM_VAL_CHECK,
        });
        challenge_bindings.push(ChallengeBinding {
            var_id: 2 + i as u32,
            source: ChallengeSource::Derived,
        });
    }

    ClaimDefinition {
        expr,
        opening_bindings,
        challenge_bindings,
    }
}

/// RAM RAF evaluation output claim.
///
/// Relates the read-address polynomial to the address unmapping polynomial.
///
/// Output claim: `c0·ra`
///
/// where `c0 = unmap_eval` (evaluation of the address unmapping polynomial).
pub fn ram_raf_evaluation() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let ra = b.opening(0);
    let unmap = b.challenge(0);

    let expr = b.build(unmap * ra);

    ClaimDefinition {
        expr,
        opening_bindings: vec![OpeningBinding {
            var_id: 0,
            polynomial_tag: poly::RAM_RA,
            sumcheck_tag: sumcheck::RAM_RAF_EVALUATION,
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
    use num_traits::{One, Zero};

    #[test]
    fn hamming_booleanity_zero_for_boolean_input() {
        let claim = hamming_booleanity();
        let eq_eval = Fr::from_u64(7);

        // H = 0: eq*0*0 + (-eq)*0 = 0
        let result = claim.evaluate::<Fr>(&[Fr::zero()], &[eq_eval, -eq_eval]);
        assert_eq!(result, Fr::zero());

        // H = 1: eq*1*1 + (-eq)*1 = eq - eq = 0
        let result = claim.evaluate::<Fr>(&[Fr::one()], &[eq_eval, -eq_eval]);
        assert_eq!(result, Fr::zero());
    }

    #[test]
    fn hamming_booleanity_nonzero_for_non_boolean() {
        let claim = hamming_booleanity();
        let eq_eval = Fr::from_u64(7);
        let h = Fr::from_u64(3);

        // H = 3: eq*(9 - 3) = 7*6 = 42
        let result = claim.evaluate::<Fr>(&[h], &[eq_eval, -eq_eval]);
        assert_eq!(result, Fr::from_u64(42));
    }

    #[test]
    fn ram_rw_checking_formula() {
        let claim = ram_read_write_checking();
        let ra = Fr::from_u64(2);
        let val = Fr::from_u64(3);
        let inc = Fr::from_u64(5);
        let gamma = Fr::from_u64(11);
        let eq_eval = Fr::from_u64(7);

        let c0 = eq_eval * (Fr::one() + gamma); // 7*12 = 84
        let c1 = eq_eval * gamma; // 7*11 = 77

        // c0*ra*val + c1*ra*inc = 84*2*3 + 77*2*5 = 504 + 770 = 1274
        let result = claim.evaluate::<Fr>(&[ra, val, inc], &[c0, c1]);
        assert_eq!(result, Fr::from_u64(1274));
    }

    #[test]
    fn ram_output_check_formula() {
        let claim = ram_output_check();
        let val_final = Fr::from_u64(10);
        let eq_io = Fr::from_u64(3);
        let val_io = Fr::from_u64(4);

        let c0 = eq_io;
        let c1 = -eq_io * val_io; // -12

        // c0*val_final + c1 = 3*10 + (-12) = 18
        let result = claim.evaluate::<Fr>(&[val_final], &[c0, c1]);
        assert_eq!(result, Fr::from_u64(18));
    }

    #[test]
    fn ram_val_check_formula() {
        let claim = ram_val_check();
        let inc = Fr::from_u64(2);
        let wa = Fr::from_u64(3);
        let lt_plus_gamma = Fr::from_u64(7);

        // c0*inc*wa = 7*2*3 = 42
        let result = claim.evaluate::<Fr>(&[inc, wa], &[lt_plus_gamma]);
        assert_eq!(result, Fr::from_u64(42));
    }

    #[test]
    fn ram_val_check_input_no_advice() {
        let claim = ram_val_check_input(0);
        let val_rw = Fr::from_u64(10);
        let val_final = Fr::from_u64(20);
        let gamma = Fr::from_u64(3);
        let neg_init = -Fr::from_u64(16); // -(1+3)*4 = -16

        // val_rw + γ*val_final + neg_init = 10 + 3*20 + (-16) = 54
        let result = claim.evaluate::<Fr>(&[val_rw, val_final], &[gamma, neg_init]);
        assert_eq!(result, Fr::from_u64(54));
    }

    #[test]
    fn ram_raf_evaluation_formula() {
        let claim = ram_raf_evaluation();
        let ra = Fr::from_u64(5);
        let unmap = Fr::from_u64(3);

        let result = claim.evaluate::<Fr>(&[ra], &[unmap]);
        assert_eq!(result, Fr::from_u64(15));
    }

    #[test]
    fn sop_equivalence_hamming() {
        let claim = hamming_booleanity();
        let h = Fr::from_u64(5);
        let eq = Fr::from_u64(3);
        let neg_eq = -eq;

        let direct = claim.evaluate::<Fr>(&[h], &[eq, neg_eq]);
        let sop = claim.expr.to_sum_of_products();
        let via_sop = sop.evaluate::<Fr>(&[h], &[eq, neg_eq]);
        assert_eq!(direct, via_sop);
    }

    #[test]
    fn sop_equivalence_rw_checking() {
        let claim = ram_read_write_checking();
        let ra = Fr::from_u64(2);
        let val = Fr::from_u64(7);
        let inc = Fr::from_u64(11);
        let c0 = Fr::from_u64(13);
        let c1 = Fr::from_u64(17);

        let direct = claim.evaluate::<Fr>(&[ra, val, inc], &[c0, c1]);
        let sop = claim.expr.to_sum_of_products();
        let via_sop = sop.evaluate::<Fr>(&[ra, val, inc], &[c0, c1]);
        assert_eq!(direct, via_sop);
    }
}

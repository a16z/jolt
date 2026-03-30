//! RAM subsystem claim definitions.
//!
//! These encode the output claim formulas for the five RAM-related sumcheck
//! instances. Each function returns a [`ClaimDefinition`]
//! whose expression captures the exact polynomial identity, and whose bindings
//! map symbolic variables to concrete Jolt polynomials.

use crate::builder::ExprBuilder;
use crate::claim::{ClaimDefinition, OpeningBinding};
use crate::PolynomialId;

// Verified against jolt-core/src/zkvm/ram/hamming_booleanity (via booleanity.rs pattern)
// Formula: eq · (H² − H)
// Degree: 3 (challenge × opening × opening)
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
            polynomial: PolynomialId::HammingWeight,
        }],
        num_challenges: 2,
    }
}

// Verified against jolt-core/src/zkvm/ram/read_write_checking.rs
// Formula: Σ eq(r,x) · ra(x) · (val(x) + γ·(inc(x) + val(x)))
//        = eq·(1+γ)·ra·val + eq·γ·ra·inc
// Degree: 3 (challenge × opening × opening)
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
    let c0 = b.challenge(0);
    let c1 = b.challenge(1);

    let expr = b.build(c0 * ra * val + c1 * ra * inc);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial: PolynomialId::RamAddress,
            },
            OpeningBinding {
                var_id: 1,
                polynomial: PolynomialId::RamVal,
            },
            OpeningBinding {
                var_id: 2,
                polynomial: PolynomialId::RamInc,
            },
        ],
        num_challenges: 2,
    }
}

// Verified against jolt-core/src/zkvm/ram/output_check.rs
// Formula: Σ eq(r,k) · io_mask(k) · (val_final(k) − val_io(k))
// Degree: 3 in jolt-core (eq × io_mask × val), but 2 in IR since eq·io_mask is pre-combined
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
    let c0 = b.challenge(0);
    let c1 = b.challenge(1);

    let expr = b.build(c0 * val_final + c1);

    ClaimDefinition {
        expr,
        opening_bindings: vec![OpeningBinding {
            var_id: 0,
            polynomial: PolynomialId::RamValFinal,
        }],
        num_challenges: 2,
    }
}

// Verified against jolt-core/src/zkvm/ram/val_check.rs
// Formula: Σ inc(j) · wa(r_addr, j) · (LT(j, r_cycle) + γ)
// Degree: 3 (challenge × opening × opening, via Toom-Cook at {0,1,2,∞})
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
    let c0 = b.challenge(0);

    let expr = b.build(c0 * inc * wa);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding {
                var_id: 0,
                polynomial: PolynomialId::RamInc,
            },
            OpeningBinding {
                var_id: 1,
                polynomial: PolynomialId::RamAddress,
            },
        ],
        num_challenges: 1,
    }
}

// Verified against jolt-core/src/zkvm/ram/val_check.rs (input_claim + input_claim_constraint)
// Formula: (val_rw − init_eval) + γ·(val_final − init_eval)
//   where init_eval = init_eval_public + Σ(selector_i · advice_i)
// Degree: 2 (challenge × opening)
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
    let neg_init = b.challenge(1);

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
            polynomial: PolynomialId::RamVal,
        },
        OpeningBinding {
            var_id: 1,
            polynomial: PolynomialId::RamValFinal,
        },
    ];

    for i in 0..n_advice {
        opening_bindings.push(OpeningBinding {
            var_id: 2 + i as u32,
            polynomial: PolynomialId::TrustedAdvice,
        });
    }

    ClaimDefinition {
        expr,
        opening_bindings,
        num_challenges: 2 + n_advice as u32,
    }
}

// Verified against jolt-core/src/zkvm/ram/ (uses same pattern as instruction RA virtual)
// Formula: eq · Π_{i=0}^{d-1} ra_committed_i
// Degree: d + 1 (challenge × d openings)
/// RAM RA virtual sumcheck output claim.
///
/// The virtual RAM RA polynomial is decomposed into `d` committed chunks.
/// The output claim is the product of all chunk evaluations scaled by eq:
///
/// Output claim: `c_0 · Π_{i=0}^{d-1} ra_i`
///
/// where `c_0 = eq_eval` (single gamma power for 1 virtual polynomial).
///
/// `d` is the number of committed RA chunks per the one-hot decomposition.
pub fn ram_ra_virtual(d: usize) -> ClaimDefinition {
    let b = ExprBuilder::new();

    let c0 = b.challenge(0);
    let mut product = c0;
    for i in 0..d {
        product = product * b.opening(i as u32);
    }

    let expr = b.build(product);

    let opening_bindings = (0..d)
        .map(|idx| OpeningBinding {
            var_id: idx as u32,
            polynomial: PolynomialId::RamRa(idx),
        })
        .collect();

    ClaimDefinition {
        expr,
        opening_bindings,
        num_challenges: 1,
    }
}

// Verified against jolt-core/src/zkvm/ram/raf_evaluation.rs
// Formula: Σ ra(k) · unmap(k) → output: unmap_eval · ra(r)
// Degree: 2 (challenge × opening)
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
    fn ram_ra_virtual_formula() {
        let claim = ram_ra_virtual(3);
        let ra = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let eq_eval = Fr::from_u64(7);

        // eq_eval * 2 * 3 * 5 = 7*30 = 210
        let result = claim.evaluate::<Fr>(&ra, &[eq_eval]);
        assert_eq!(result, Fr::from_u64(210));
    }

    #[test]
    fn ram_ra_virtual_composition_equivalence() {
        let claim = ram_ra_virtual(4);
        let openings: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let challenges = vec![Fr::from_u64(10)];

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_formula = claim
            .to_composition_formula()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_formula);
    }

    #[test]
    fn composition_equivalence_hamming() {
        let claim = hamming_booleanity();
        let h = Fr::from_u64(5);
        let eq = Fr::from_u64(3);
        let neg_eq = -eq;

        let direct = claim.evaluate::<Fr>(&[h], &[eq, neg_eq]);
        let via_formula = claim
            .to_composition_formula()
            .evaluate::<Fr>(&[h], &[eq, neg_eq]);
        assert_eq!(direct, via_formula);
    }

    #[test]
    fn composition_equivalence_rw_checking() {
        let claim = ram_read_write_checking();
        let ra = Fr::from_u64(2);
        let val = Fr::from_u64(7);
        let inc = Fr::from_u64(11);
        let c0 = Fr::from_u64(13);
        let c1 = Fr::from_u64(17);

        let direct = claim.evaluate::<Fr>(&[ra, val, inc], &[c0, c1]);
        let via_formula = claim
            .to_composition_formula()
            .evaluate::<Fr>(&[ra, val, inc], &[c0, c1]);
        assert_eq!(direct, via_formula);
    }
}

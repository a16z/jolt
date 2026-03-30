//! Bytecode subsystem claim definitions.
//!
//! Bytecode read-RAF checking is the most complex sumcheck instance in Jolt.
//! It folds multiple stages' bytecode lookups and RAF evaluations into a
//! single multi-phase sumcheck.
//!
//! Verified against jolt-core/src/zkvm/bytecode/read_raf_checking.rs.

use crate::builder::ExprBuilder;
use crate::claim::{ClaimDefinition, OpeningBinding};
use crate::PolynomialId;

/// Bytecode RA virtual sumcheck output claim.
///
/// Analogous to [`ram_ra_virtual`](super::ram::ram_ra_virtual) and
/// [`instruction_ra_virtual`](super::instruction::instruction_ra_virtual),
/// but for bytecode one-hot decomposition chunks.
///
/// Output claim: `c_0 · Π_{i=0}^{d-1} ra_i`
///
/// where `c_0 = eq_eval` and `d` is the number of bytecode RA chunks.
///
/// Verified against jolt-core bytecode read-RAF checking RA product structure.
pub fn bytecode_ra_virtual(d: usize) -> ClaimDefinition {
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
            polynomial: PolynomialId::BytecodeRa(idx),
        })
        .collect();

    ClaimDefinition {
        expr,
        opening_bindings,
        num_challenges: 1,
    }
}

/// Bytecode read-RAF checking output claim.
///
/// This is a multi-stage folded sumcheck that verifies bytecode lookups
/// from 5 prior stages plus RAF evaluation. The output claim for each
/// per-stage sub-instance follows the pattern: `c · ra_product · val`.
///
/// Because the formula depends on runtime per-stage challenge values and
/// polynomials from multiple stages, we parameterize by the number of
/// folded stages (`n_stages`) rather than hard-coding the structure.
///
/// Output claim: `Σ_{s=0}^{n_stages-1} c_s · val_s`
///
/// where `c_s = eq_s_eval · γ^s` and `val_s` is the per-stage address-only
/// polynomial value. The RA product is already factored into the challenges
/// via the Toom-Cook eq decomposition.
///
/// Verified against jolt-core/src/zkvm/bytecode/read_raf_checking.rs.
pub fn bytecode_read_raf(n_stages: usize) -> ClaimDefinition {
    let b = ExprBuilder::new();

    let mut terms = b.zero();
    for s in 0..n_stages {
        let val_s = b.opening(s as u32);
        let c_s = b.challenge(s as u32);
        terms = terms + c_s * val_s;
    }

    let expr = b.build(terms);

    let opening_bindings = (0..n_stages)
        .map(|s| OpeningBinding {
            var_id: s as u32,
            polynomial: PolynomialId::BytecodeReadRafVal(s),
        })
        .collect();

    ClaimDefinition {
        expr,
        opening_bindings,
        num_challenges: n_stages as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};

    #[test]
    fn bytecode_ra_virtual_formula() {
        let claim = bytecode_ra_virtual(3);
        let ra = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let eq_eval = Fr::from_u64(7);

        // eq * 2 * 3 * 5 = 210
        let result = claim.evaluate::<Fr>(&ra, &[eq_eval]);
        assert_eq!(result, Fr::from_u64(210));
    }

    #[test]
    fn bytecode_read_raf_formula() {
        let claim = bytecode_read_raf(3);
        let vals: Vec<Fr> = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let challenges: Vec<Fr> = vec![Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];

        // 7*2 + 11*3 + 13*5 = 14 + 33 + 65 = 112
        let result = claim.evaluate::<Fr>(&vals, &challenges);
        assert_eq!(result, Fr::from_u64(112));
    }

    #[test]
    fn bytecode_ra_virtual_composition_equivalence() {
        let claim = bytecode_ra_virtual(4);
        let openings: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let challenges = vec![Fr::from_u64(10)];

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_formula = claim
            .to_composition_formula()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_formula);
    }
}

//! Instruction lookup claim definitions.
//!
//! Instruction lookups use RA (read-address) decomposition where a virtual
//! RA polynomial is decomposed into a product of committed RA chunks.

use crate::builder::ExprBuilder;
use crate::claim::{ChallengeBinding, ChallengeSource, ClaimDefinition, OpeningBinding};
use crate::zkvm::tags::{poly, sumcheck};

// Verified against jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs
// Formula: Σ_x eq(r_cycle,x) · Σ_i γ^i · Π_{j=0}^{m-1} ra_{i·m+j}(x)
// Degree: n_committed_per_virtual + 1 (challenge × m openings)
/// Instruction RA virtual sumcheck output claim.
///
/// Each virtual RA polynomial is decomposed into `m` committed chunks.
/// The output claim is a γ-weighted sum of products of chunk evaluations.
///
/// Output claim: `Σ_i c_i · Π_{j=0}^{m-1} ra_{i·m+j}`
///
/// where `c_i = eq_eval · γ^i`.
///
/// `n_virtual` is the number of virtual RA polynomials and `n_committed_per_virtual`
/// is the number of committed chunks per virtual polynomial.
pub fn instruction_ra_virtual(n_virtual: usize, n_committed_per_virtual: usize) -> ClaimDefinition {
    let b = ExprBuilder::new();
    let m = n_committed_per_virtual;

    let mut terms = b.zero();
    for i in 0..n_virtual {
        let c_i = b.challenge(i as u32);
        let mut product = c_i;
        for j in 0..m {
            let ra_ij = b.opening((i * m + j) as u32);
            product = product * ra_ij;
        }
        terms = terms + product;
    }

    let expr = b.build(terms);

    let opening_bindings = (0..n_virtual * m)
        .map(|idx| OpeningBinding {
            var_id: idx as u32,
            polynomial_tag: poly::instruction_ra(idx),
            sumcheck_tag: sumcheck::INSTRUCTION_RA_VIRTUAL,
        })
        .collect();

    let challenge_bindings = (0..n_virtual)
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};

    #[test]
    fn ra_virtual_single_poly_two_chunks() {
        // 1 virtual, 2 committed: c0 * ra0 * ra1
        let claim = instruction_ra_virtual(1, 2);
        let ra0 = Fr::from_u64(3);
        let ra1 = Fr::from_u64(5);
        let c0 = Fr::from_u64(7);

        let result = claim.evaluate::<Fr>(&[ra0, ra1], &[c0]);
        assert_eq!(result, Fr::from_u64(105)); // 7*3*5
    }

    #[test]
    fn ra_virtual_two_polys_two_chunks() {
        // 2 virtual, 2 committed each: c0*ra0*ra1 + c1*ra2*ra3
        let claim = instruction_ra_virtual(2, 2);
        let ra = [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ];
        let c = [Fr::from_u64(11), Fr::from_u64(13)];

        // 11*2*3 + 13*5*7 = 66 + 455 = 521
        let result = claim.evaluate::<Fr>(&ra, &c);
        assert_eq!(result, Fr::from_u64(521));
    }

    #[test]
    fn ra_virtual_sop_equivalence() {
        let claim = instruction_ra_virtual(2, 3);
        let openings: Vec<Fr> = (1..=6).map(Fr::from_u64).collect();
        let challenges: Vec<Fr> = (10..=11).map(Fr::from_u64).collect();

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_sop = claim
            .expr
            .to_sum_of_products()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_sop);
    }
}

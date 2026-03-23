//! One-hot RA booleanity claim definition.
//!
//! Verified against jolt-core/src/subprotocols/booleanity.rs.
//! Proves that every RA polynomial is boolean: ra_i(x) ∈ {0, 1}.
//!
//! Formula: `Σ_i γ^i · eq(r, x) · (ra_i(x)² − ra_i(x)) = 0`
//!
//! This is a zero-check (input claim = 0). The sumcheck operates over
//! the combined (address || cycle) domain with `log_k_chunk + log_T` rounds.

use crate::builder::ExprBuilder;
use crate::claim::{ChallengeBinding, ChallengeSource, ClaimDefinition, OpeningBinding};
use crate::zkvm::tags::sumcheck;

/// RA booleanity output claim (γ-batched zero-check).
///
/// Batches `n_polys` RA polynomials into a single zero-check via γ-powers.
/// Each RA polynomial's booleanity is checked simultaneously.
///
/// Output claim: `Σ_i c_i · (ra_i² − ra_i)`
///
/// where `c_i = eq_eval · γ^i`.
///
/// `n_polys` is `instruction_d + bytecode_d + ram_d` (total RA polynomial count).
/// `poly_tags` maps each index to its polynomial tag (from `poly::instruction_ra(i)`, etc.).
pub fn ra_booleanity(n_polys: usize, poly_tags: &[u64]) -> ClaimDefinition {
    assert_eq!(poly_tags.len(), n_polys);

    let b = ExprBuilder::new();

    let mut terms = b.zero();
    for i in 0..n_polys {
        let ra_i = b.opening(i as u32);
        let c_i = b.challenge(i as u32);
        // c_i · (ra_i² − ra_i) = c_i · ra_i · ra_i − c_i · ra_i
        terms = terms + c_i * ra_i * ra_i - c_i * ra_i;
    }

    let expr = b.build(terms);

    let opening_bindings = (0..n_polys)
        .map(|i| OpeningBinding {
            var_id: i as u32,
            polynomial_tag: poly_tags[i],
            sumcheck_tag: sumcheck::BOOLEANITY,
        })
        .collect();

    let challenge_bindings = (0..n_polys)
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
    use crate::zkvm::tags::poly;
    use jolt_field::{Field, Fr};
    use num_traits::Zero;

    #[test]
    fn booleanity_zero_for_boolean_inputs() {
        let tags = vec![poly::instruction_ra(0), poly::instruction_ra(1)];
        let claim = ra_booleanity(2, &tags);

        let eq = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let c0 = eq;
        let c1 = eq * gamma;

        // ra0=1, ra1=0: c0*(1-1) + c1*(0-0) = 0
        let result = claim.evaluate::<Fr>(&[Fr::from_u64(1), Fr::from_u64(0)], &[c0, c1]);
        assert_eq!(result, Fr::zero());

        // ra0=0, ra1=1: 0 + c1*(1-1) = 0
        let result = claim.evaluate::<Fr>(&[Fr::from_u64(0), Fr::from_u64(1)], &[c0, c1]);
        assert_eq!(result, Fr::zero());
    }

    #[test]
    fn booleanity_nonzero_for_non_boolean() {
        let tags = vec![poly::instruction_ra(0)];
        let claim = ra_booleanity(1, &tags);

        let eq = Fr::from_u64(5);
        let ra = Fr::from_u64(3);

        // c0*(3²-3) = 5*(9-3) = 30
        let result = claim.evaluate::<Fr>(&[ra], &[eq]);
        assert_eq!(result, Fr::from_u64(30));
    }

    #[test]
    fn booleanity_sop_equivalence() {
        let tags = vec![poly::instruction_ra(0), poly::bytecode_ra(0)];
        let claim = ra_booleanity(2, &tags);
        let openings: Vec<Fr> = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let challenges: Vec<Fr> = vec![Fr::from_u64(7), Fr::from_u64(11)];

        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_sop = claim
            .expr
            .to_sum_of_products()
            .evaluate::<Fr>(&openings, &challenges);
        assert_eq!(direct, via_sop);
    }
}

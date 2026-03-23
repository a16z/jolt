//! PCS opening proof generation for the typed DAG pipeline.
//!
//! [`prove_opening`] collects all committed polynomial claims from S6 and S7,
//! normalizes dense polynomials to the unified point via Lagrange zero-selectors,
//! and produces a batch Dory opening proof through RLC reduction.
//!
//! This is the final proving step — it runs after all sumcheck stages and before
//! proof assembly.

use jolt_field::Field;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, OpeningReduction, ProverClaim, RlcReduction,
};
use jolt_transcript::Transcript;

use jolt_verifier::protocol::types::{S6Evals, S7Evals};
use crate::tables::PolynomialTables;

/// Result of the PCS opening phase.
///
/// Contains the reduced opening proofs and the RLC reduction proof (trivial
/// for [`RlcReduction`], which produces `()`).
pub struct OpeningResult<PCS: CommitmentScheme> {
    /// Batch opening proofs — one per distinct evaluation point after RLC.
    /// With the unified-point architecture, this should be exactly one proof.
    pub proofs: Vec<PCS::Proof>,
}

/// Collects all committed polynomial claims, normalizes to the unified point,
/// and produces batch PCS opening proofs.
///
/// # Claim collection
///
/// Two categories of committed polynomials:
///
/// 1. **Dense** (`ram_inc`, `rd_inc`) — evaluated at `r_cycle_s6` (length `log_T`).
///    Zero-padded to the full `(r_addr || r_cycle)` space and Lagrange-normalized:
///    `eval_unified = eval_at_r_cycle × ∏(1 − r_addr_i)`.
///
/// 2. **RA** (`instruction_ra`, `bytecode_ra`, `ram_ra`) — already at the unified
///    point `(r_addr_s7 || r_cycle_s6)` from S7's HammingWeightCR.
///
/// # Reduction
///
/// All claims share the same unified point, so RLC combines them into a single
/// claim, yielding exactly one PCS opening proof.
pub fn prove_opening<PCS, T>(
    s6: &S6Evals<PCS::Field>,
    s7: &S7Evals<PCS::Field>,
    tables: &PolynomialTables<PCS::Field>,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> OpeningResult<PCS>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript<Challenge = PCS::Field>,
{
    let unified = &s7.unified_point;
    let log_t = tables.log_num_cycles();
    let log_k_chunk = unified.len() - log_t;
    let r_addr = &unified[..log_k_chunk];

    // Lagrange zero-selector: eq(r_addr, 0) = ∏(1 - r_addr_i)
    let lagrange = {
        let one = PCS::Field::from_u64(1);
        r_addr.iter().fold(one, |acc, &r| acc * (one - r))
    };

    let padded_len = 1usize << unified.len();
    let total_d = tables.total_d();
    // 2 dense + total_d RA polynomials
    let mut claims: Vec<ProverClaim<PCS::Field>> = Vec::with_capacity(2 + total_d);

    // ── Dense polynomials (Lagrange-normalized) ─────────────────────
    claims.push(make_dense_claim(
        &tables.ram_inc,
        padded_len,
        s6.ram_inc_reduced.eval * lagrange,
        unified,
    ));
    claims.push(make_dense_claim(
        &tables.rd_inc,
        padded_len,
        s6.rd_inc_reduced.eval * lagrange,
        unified,
    ));

    // ── RA polynomials (already at unified point from S7) ───────────
    for (i, eval) in s7.instruction_ra.iter().enumerate() {
        claims.push(ProverClaim {
            evaluations: tables.instruction_ra[i].clone(),
            point: unified.clone(),
            eval: eval.eval,
        });
    }
    for (i, eval) in s7.bytecode_ra.iter().enumerate() {
        claims.push(ProverClaim {
            evaluations: tables.bytecode_ra[i].clone(),
            point: unified.clone(),
            eval: eval.eval,
        });
    }
    for (i, eval) in s7.ram_ra.iter().enumerate() {
        claims.push(ProverClaim {
            evaluations: tables.ram_ra[i].clone(),
            point: unified.clone(),
            eval: eval.eval,
        });
    }

    // ── RLC reduction → single claim → PCS open ────────────────────
    let (reduced, ()) =
        <RlcReduction as OpeningReduction<PCS>>::reduce_prover(claims, transcript);

    debug_assert!(
        reduced.len() == 1,
        "all claims at unified point should RLC to 1 claim, got {}",
        reduced.len()
    );

    let proofs = reduced
        .into_iter()
        .map(|claim| {
            let poly: PCS::Polynomial = claim.evaluations.into();
            PCS::open(
                &poly,
                &claim.point,
                claim.eval,
                pcs_setup,
                None,
                transcript,
            )
        })
        .collect();

    OpeningResult { proofs }
}

/// Creates a `ProverClaim` for a dense polynomial zero-padded to the unified point.
///
/// The evaluation table is extended from `2^log_T` to `padded_len = 2^(log_k + log_T)`
/// with trailing zeros. The evaluation at the padded point is pre-computed by
/// the caller as `eval_at_r_cycle × zero_selector(r_addr)`.
fn make_dense_claim<F: Field>(
    table: &[F],
    padded_len: usize,
    eval_at_unified: F,
    unified_point: &[F],
) -> ProverClaim<F> {
    debug_assert!(
        padded_len >= table.len(),
        "padded_len ({padded_len}) < table.len() ({})",
        table.len()
    );
    let mut padded = vec![F::zero(); padded_len];
    padded[..table.len()].copy_from_slice(table);

    ProverClaim {
        evaluations: padded,
        point: unified_point.to_vec(),
        eval: eval_at_unified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    #[test]
    fn lagrange_zero_selector_all_zeros() {
        let r = [Fr::zero(); 3];
        let selector: Fr = r.iter().fold(Fr::from_u64(1), |acc, &ri| acc * (Fr::from_u64(1) - ri));
        assert_eq!(selector, Fr::one());
    }

    #[test]
    fn lagrange_zero_selector_nonzero() {
        let r = [Fr::one(), Fr::zero()];
        let selector: Fr = r.iter().fold(Fr::from_u64(1), |acc, &ri| acc * (Fr::from_u64(1) - ri));
        assert_eq!(selector, Fr::zero());
    }

    #[test]
    fn make_dense_claim_pads_correctly() {
        let table = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let point = vec![Fr::zero(); 3];
        let claim = make_dense_claim(&table, 8, Fr::from_u64(42), &point);

        assert_eq!(claim.evaluations.len(), 8);
        assert_eq!(claim.evaluations[0], Fr::from_u64(1));
        assert_eq!(claim.evaluations[1], Fr::from_u64(2));
        for i in 2..8 {
            assert_eq!(claim.evaluations[i], Fr::zero());
        }
        assert_eq!(claim.eval, Fr::from_u64(42));
        assert_eq!(claim.point, point);
    }
}

//! Multilinear polynomial commitmnets as a matrix
use crate::arithmetic::{Field, Group, MultiScalarMul, Pairing};
use crate::poly::Polynomial;
use crate::setup::ProverSetup;

/// Dory's 2-tier homomorphic commitment to multilinear polynomial arranged as matrix
/// Tier 1: Row commitments in G1, Tier 2: Multi-pairing to GT
/// See page 12 of the paper.
pub fn compute_polynomial_commitment<
    E: Pairing<G1 = G1>,
    M1: MultiScalarMul<G1>,
    P: Polynomial<F, G1> + ?Sized,
    F: Field,
    G1: Group<Scalar = F>,
>(
    poly: &P,      // Polynomial
    offset: usize, // Starting position in matrix
    sigma: usize,  // log₂(matrix_width)
    prover_setup: &ProverSetup<E>,
) -> (E::GT, Vec<G1>) {
    let num_columns = 1 << sigma;

    let rows_offset = offset / num_columns; // Row start position

    // TODO(moodlezoup): handle offset
    let row_len = num_columns;
    let row_commitments = poly.commit_rows::<M1>(&prover_setup.g1_vec()[..row_len], row_len);

    // --- TIER 2: Multi-pairing to combine row commitments ---

    // Use cached multi-pairing if G2 cache is available, otherwise fall back to regular multi-pairing
    let commitment = if prover_setup.g2_cache.is_some() {
        // Use cached G2 values from prover setup
        E::multi_pair_cached(
            Some(&row_commitments),
            None,
            None, // G1: use runtime points row_commitments
            None,
            Some(row_commitments.len()),
            prover_setup.g2_cache.as_ref(), // G2: use cached elements from rows_offset
        )
    } else {
        // Fall back to regular multi-pairing
        let g2_elements = &prover_setup.g2_vec()[rows_offset..rows_offset + row_commitments.len()];
        E::multi_pair(&row_commitments, g2_elements)
    };

    // Return `row_commitments` because they will come in handy for the opening proof
    (commitment, row_commitments)
}

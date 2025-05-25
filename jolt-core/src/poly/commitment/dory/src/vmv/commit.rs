//! Multilinear polynomial commitmnets as a matrix
use crate::arithmetic::{Field, Group, MultiScalarMul, Pairing};
use crate::poly::compute_polynomial_evaluation;
use crate::setup::ProverSetup;

/// Dory's 2-tier homomorphic commitment to multilinear polynomial arranged as matrix
/// Tier 1: Row commitments in G1, Tier 2: Multi-pairing to GT
/// See page 12 of the paper.
pub fn compute_polynomial_commitment<E: Pairing, M1: MultiScalarMul<E::G1>>(
    coeffs: &[<E::G1 as Group>::Scalar], // Polynomial coefficients
    offset: usize,                       // Starting position in matrix
    sigma: usize,                        // log₂(matrix_width)
    prover_setup: &ProverSetup<E>,
) -> E::GT {
    let num_columns = 1 << sigma;

    // Handle arbitrary offset within the matrix
    let first_row_offset = offset % num_columns; // Column start position
    let rows_offset = offset / num_columns; // Row start position
    let first_row_len = coeffs.len().min(num_columns - first_row_offset);

    let (first_row_coeffs, remaining_coeffs) = coeffs.split_at(first_row_len);
    let remaining_row_count = (remaining_coeffs.len() + num_columns - 1) / num_columns;

    // --- TIER 1: Compute row commitments in G1 ---

    let first_row_commit = if first_row_len > 0 {
        M1::msm(
            &prover_setup.g1_vec[first_row_offset..first_row_offset + first_row_len],
            first_row_coeffs,
        )
    } else {
        E::G1::identity()
    };

    let mut g1_row_commitments = Vec::with_capacity(1 + remaining_row_count);
    g1_row_commitments.push(first_row_commit);

    // Remaining row commitments (full rows)
    for row_coeffs in remaining_coeffs.chunks(num_columns) {
        let row_commit = M1::msm(&prover_setup.g1_vec[0..row_coeffs.len()], row_coeffs);
        g1_row_commitments.push(row_commit);
    }

    // --- TIER 2: Multi-pairing to combine row commitments ---

    let g2_elements = &prover_setup.g2_vec[rows_offset..rows_offset + g1_row_commitments.len()];
    E::multi_pair(&g1_row_commitments, g2_elements) // Final commitment in GT
}

/// Create commitment batch, batching factors, and evaluations for verification
/// This provides the values needed for verify_evaluation_proof
pub fn commit_and_evaluate_batch<E: Pairing, M1: MultiScalarMul<E::G1>>(
    coeffs: &[<E::G1 as Group>::Scalar],
    point: &[<E::G1 as Group>::Scalar],
    offset: usize,
    sigma: usize,
    prover_setup: &ProverSetup<E>,
) -> (
    Vec<E::GT>,                    // commitment_batch
    Vec<<E::G1 as Group>::Scalar>, // batching_factors
    Vec<<E::G1 as Group>::Scalar>, // evaluations
)
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    <E::G1 as Group>::Scalar: Field + Clone,
{
    // Compute the commitment to the polynomial
    let commitment = compute_polynomial_commitment::<E, M1>(coeffs, offset, sigma, prover_setup);

    // Compute the evaluation of the polynomial at the point
    let evaluation = compute_polynomial_evaluation(coeffs, point);

    // For a single polynomial, we use a single batching factor of 1
    let commitment_batch = vec![commitment];

    // @TODO(markosg04): support batching
    let batching_factors = vec![<E::G1 as Group>::Scalar::one()];
    let evaluations = vec![evaluation]; // for now just one evaluation

    (commitment_batch, batching_factors, evaluations)
}

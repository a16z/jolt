//! Spartan proof structure.
//!
//! Contains the witness commitment, sumcheck proof, and evaluation claims
//! needed for the verifier to check the Spartan argument.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;

/// A Spartan proof for an R1CS instance.
///
/// The proof demonstrates that the prover knows a witness $z$ satisfying
/// $Az \circ Bz = Cz$ without revealing $z$. It contains:
///
/// 1. A commitment to the witness polynomial $\tilde{z}$.
/// 2. An outer sumcheck proof for
///    $\sum_x \tilde{eq}(x,\tau) \cdot (\tilde{Az}(x) \cdot \tilde{Bz}(x) - \tilde{Cz}(x)) = 0$.
/// 3. Evaluation claims $\widetilde{Az}(r_x)$, $\widetilde{Bz}(r_x)$, $\widetilde{Cz}(r_x)$.
/// 4. An inner sumcheck proof verifying the evaluation claims against the matrix MLEs:
///    $\sum_y M(r_x, y) \cdot \tilde{z}(y) = \rho_A \cdot \widetilde{Az}(r_x) + \rho_B \cdot \widetilde{Bz}(r_x) + \rho_C \cdot \widetilde{Cz}(r_x)$.
/// 5. An opening proof for the witness polynomial at the inner challenge point $r_y$.
pub struct SpartanProof<F: Field, PCS: CommitmentScheme> {
    /// Commitment to the multilinear extension of the witness vector.
    pub witness_commitment: PCS::Output,
    /// Sumcheck proof for the outer Spartan sumcheck.
    pub outer_sumcheck_proof: SumcheckProof<F>,
    /// Evaluation of $\widetilde{Az}$ at the outer sumcheck challenge $r_x$.
    pub az_eval: F,
    /// Evaluation of $\widetilde{Bz}$ at the outer sumcheck challenge $r_x$.
    pub bz_eval: F,
    /// Evaluation of $\widetilde{Cz}$ at the outer sumcheck challenge $r_x$.
    pub cz_eval: F,
    /// Sumcheck proof for the inner Spartan sumcheck, binding evaluation claims
    /// to the matrix MLEs and witness polynomial.
    pub inner_sumcheck_proof: SumcheckProof<F>,
    /// Evaluation of the witness polynomial $\tilde{z}(r_y)$ at the inner challenge point.
    pub witness_eval: F,
    /// Opening proof for the witness polynomial at the inner challenge point $r_y$.
    pub witness_opening_proof: PCS::Proof,
}

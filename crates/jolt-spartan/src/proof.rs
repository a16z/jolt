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
/// 2. A sumcheck proof for
///    $\sum_x \tilde{eq}(x,\tau) \cdot (\tilde{Az}(x) \cdot \tilde{Bz}(x) - \tilde{Cz}(x)) = 0$.
/// 3. Evaluation claims at the sumcheck challenge point.
/// 4. An opening proof for the witness polynomial.
pub struct SpartanProof<F: Field, PCS: CommitmentScheme> {
    /// Commitment to the multilinear extension of the witness vector.
    pub witness_commitment: PCS::Commitment,
    /// Sumcheck proof for the outer Spartan sumcheck.
    pub sumcheck_proof: SumcheckProof<F>,
    /// Evaluation of the witness polynomial $\tilde{z}(r_y)$ at the inner challenge point.
    pub witness_eval: F,
    /// Evaluation of $\widetilde{Az}$ at the sumcheck challenge $r_x$.
    pub az_eval: F,
    /// Evaluation of $\widetilde{Bz}$ at the sumcheck challenge $r_x$.
    pub bz_eval: F,
    /// Evaluation of $\widetilde{Cz}$ at the sumcheck challenge $r_x$.
    pub cz_eval: F,
    /// Opening proof for the witness polynomial at the claimed evaluation point.
    pub witness_opening_proof: PCS::Proof,
}

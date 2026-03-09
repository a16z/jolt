//! Spartan proof structures.
//!
//! Spartan is a pure PIOP — it reduces R1CS satisfiability to sumcheck
//! arguments and polynomial evaluation claims. Commitment and opening
//! are the caller's responsibility.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;
use serde::{Deserialize, Serialize};

/// A Spartan proof for an R1CS instance.
///
/// Contains the two sumcheck proofs (outer and inner) and the
/// evaluation claims needed for the verifier to check the Spartan PIOP.
/// The witness commitment and opening proof are NOT included — the
/// caller handles PCS operations externally.
///
/// The proof demonstrates that the prover knows a witness $z$ satisfying
/// $Az \circ Bz = Cz$ via:
///
/// 1. An outer sumcheck proof for
///    $\sum_x \tilde{eq}(x,\tau) \cdot (\tilde{Az}(x) \cdot \tilde{Bz}(x) - \tilde{Cz}(x)) = 0$.
/// 2. Evaluation claims $\widetilde{Az}(r_x)$, $\widetilde{Bz}(r_x)$, $\widetilde{Cz}(r_x)$.
/// 3. An inner sumcheck proof reducing to a single witness evaluation.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProof<F: Field> {
    pub outer_sumcheck_proof: SumcheckProof<F>,
    pub az_eval: F,
    pub bz_eval: F,
    pub cz_eval: F,
    pub inner_sumcheck_proof: SumcheckProof<F>,
    /// Evaluation of the witness polynomial $\tilde{z}(r_y)$ at the inner
    /// challenge point. The caller verifies this via a PCS opening proof.
    pub witness_eval: F,
}

/// A Spartan proof for a **relaxed** R1CS instance: $Az \circ Bz = u \cdot Cz + E$.
///
/// Used by the BlindFold protocol after Nova folding. The relaxed equation
/// introduces a scalar $u$ and error vector $E$. The outer sumcheck becomes:
/// $$\sum_x \widetilde{eq}(x,\tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - u \cdot \widetilde{Cz}(x) - \widetilde{E}(x)) = 0$$
///
/// Commitments to the witness and error polynomials are **not** included in the
/// proof — they are passed as separate parameters by the caller (BlindFold).
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
#[allow(clippy::type_complexity)]
pub struct RelaxedSpartanProof<F: Field, PCS: CommitmentScheme> {
    pub outer_sumcheck_proof: SumcheckProof<F>,
    pub az_eval: F,
    pub bz_eval: F,
    pub cz_eval: F,
    pub e_eval: F,
    pub inner_sumcheck_proof: SumcheckProof<F>,
    pub witness_eval: F,
    pub witness_opening_proof: PCS::Proof,
    pub error_opening_proof: PCS::Proof,
}

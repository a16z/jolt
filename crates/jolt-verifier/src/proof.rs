//! Jolt proof and associated types.
//!
//! [`JoltProof`] is the complete proof artifact sent from prover to verifier.
//! It carries the Spartan R1CS proof, per-stage sumcheck proofs with claimed
//! polynomial evaluations, and batch PCS opening proofs.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::UniformSpartanProof;
use jolt_sumcheck::SumcheckProof;
use serde::{Deserialize, Serialize};

/// Per-stage sumcheck proof with claimed polynomial evaluations.
///
/// After the batched sumcheck completes for one stage, the prover reveals the
/// individual polynomial evaluations at the sumcheck challenge point. The
/// verifier checks that these evaluations are consistent with the sumcheck
/// final evaluation (via the stage's claim formula) and records them as
/// opening claims for the batch opening phase.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SumcheckStageProof<F: Field> {
    /// Batched sumcheck proof for this stage.
    pub sumcheck_proof: SumcheckProof<F>,
    /// Claimed evaluations of the stage's polynomials at the sumcheck
    /// challenge point. Ordering is stage-defined and must be consistent
    /// between prover and verifier.
    pub evaluations: Vec<F>,
}

/// Batch PCS opening proofs.
///
/// After all sumcheck stages complete, the prover RLC-reduces opening claims
/// by evaluation point and produces one PCS opening proof per distinct
/// reduced claim.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchOpeningProofs<PCS: CommitmentScheme> {
    pub proofs: Vec<PCS::Proof>,
}

/// Complete Jolt proof for one program execution.
///
/// Produced by stages S1 (uniform Spartan) through S8 (batch openings). The
/// verifier replays the Fiat-Shamir transcript and checks each component
/// in sequence.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltProof<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// Uniform Spartan R1CS proof (outer + inner sumcheck + witness opening).
    pub spartan_proof: UniformSpartanProof<F, PCS>,
    /// Per-stage sumcheck proofs (S2–S7) with claimed evaluations.
    pub stage_proofs: Vec<SumcheckStageProof<F>>,
    /// Batch PCS opening proofs from the opening reduction phase.
    pub opening_proofs: BatchOpeningProofs<PCS>,
    /// Commitments to all committed polynomials.
    pub commitments: Vec<PCS::Output>,
    /// Number of execution cycles in the trace.
    pub trace_length: usize,
}

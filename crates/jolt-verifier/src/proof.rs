//! [`JoltProof`] is the complete proof artifact sent from prover to verifier.
//! It carries the Spartan R1CS proof, per-stage sumcheck proofs with claimed
//! polynomial evaluations, batch PCS opening proofs, and a [`ProverConfig`]
//! so the verifier can reconstruct claim structure without external context.

use crate::config::ProverConfig;
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
    pub sumcheck_proof: SumcheckProof<F>,
    /// Claimed evaluations of the stage's polynomials at the sumcheck
    /// challenge point. Ordering is stage-defined and must be consistent
    /// between prover and verifier.
    pub evaluations: Vec<F>,
}

/// Complete Jolt proof for one program execution.
///
/// Self-contained: the [`config`](Self::config) field carries all parameters
/// the verifier needs to reconstruct claim structure. The verifier only needs
/// the proof and a [`JoltVerifyingKey`](crate::key::JoltVerifyingKey).
///
/// Produced by stages S1 (uniform Spartan) through S8 (batch openings). The
/// verifier replays the Fiat-Shamir transcript and checks each component
/// in sequence.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltProof<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// Prover configuration — trace dimensions, one-hot config, RW config.
    pub config: ProverConfig,
    /// Uniform Spartan R1CS proof (PIOP only — no PCS).
    pub spartan_proof: UniformSpartanProof<F>,
    /// Per-stage sumcheck proofs (S2–S7) with claimed evaluations.
    pub stage_proofs: Vec<SumcheckStageProof<F>>,
    /// Batch PCS opening proofs from the opening reduction phase.
    pub opening_proofs: Vec<PCS::Proof>,
    /// Commitment to the R1CS witness polynomial.
    ///
    /// Appended to the Fiat-Shamir transcript before Spartan verification.
    /// Used by the verifier to construct the witness opening claim.
    pub witness_commitment: PCS::Output,
    /// Commitments to stage polynomials (RA chunks, inc, etc.).
    pub commitments: Vec<PCS::Output>,
}
